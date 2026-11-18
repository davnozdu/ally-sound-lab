import argparse
import fcntl
import json
import os
from pathlib import Path
import select
from queue import Empty, Full, Queue
import signal
import subprocess
import sys
from threading import Event, Thread
from time import monotonic, perf_counter

import numpy as np
import onnxruntime as ort

from spatial import CHANNELS, DEVICES, HOP, PROFILES, RATE, Delay, Renderer

SINK = "ally_sound_lab"
RUNTIME_DIR = Path.home() / ".local" / "share" / "ally-sound-npu"
MODELS = RUNTIME_DIR / "models"
ROUTE_STATE = RUNTIME_DIR / "live-route.json"


def emit(**fields):
    print(json.dumps(fields), flush=True)


def pulse(*args):
    return subprocess.check_output(["/usr/bin/pactl", *args], text=True, timeout=4).strip()


def sinks():
    return {item["name"]: item for item in json.loads(pulse("--format=json", "list", "sinks"))}


def runtime(name, threads):
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.add_session_config_entry("session.use_kleidiai", "0")
    return ort.InferenceSession(str(MODELS / name), options, providers=["CPUExecutionProvider"])


class Vocals:
    latency = 1  # hops: the graph returns the previous input hop

    def __init__(self):
        self.session = runtime("vocals.onnx", 1)
        self.state = {
            item.name: np.zeros(item.shape, dtype=np.float32)
            for item in self.session.get_inputs()[1:]
        }
        self.previous = np.zeros((2, HOP), dtype=np.float32)

    def process(self, audio, tick=None):
        vocals, *states = self.session.run(
            None, {"audio_chunk": np.ascontiguousarray(audio[None]), **self.state}
        )
        self.state = dict(zip(self.state, states))
        # The graph emits the previous input hop after its initial pre-roll.
        dry = self.previous
        self.previous = audio
        return dry, dry - vocals[0, 0]


class PipelinedVocals:
    """Runs StemgenRT on its own thread, one block behind the audio thread.

    ONNX Runtime releases the GIL, so inference of block N overlaps with the
    spatial rendering of block N-1 on another core. Both queues are kept full
    but never waited on: when inference falls behind (it runs at ~90% of one
    core, i.e. right at the 2.9 ms block budget) the audio thread carries on
    with its own delayed dry signal instead of stalling. Blocking here cost
    dropouts -- audible as crackle -- and every stall pushed the whole stream
    back in time.
    """

    latency = Vocals.latency + 1

    def __init__(self, vocals):
        self.vocals = vocals
        self.inbox = Queue(maxsize=1)
        self.outbox = Queue(maxsize=1)
        self.pending = False
        self.last_tick = None
        self.fallback = Delay(2, self.latency * HOP)  # dry signal kept in step
        Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            audio = self.inbox.get()
            self.outbox.put(self.vocals.process(audio))

    def process(self, audio, tick):
        if self.pending and self.last_tick != tick - 1:
            # This group was idle for a while: its queued result is stale.
            try:
                self.outbox.get_nowait()
            except Empty:
                pass
            self.pending = False
        self.last_tick = tick
        dry = self.fallback.process(np.asarray(audio, dtype=np.float32))
        try:
            self.inbox.put_nowait(np.array(audio, dtype=np.float32, copy=True))
            self.pending = True
        except Full:
            pass                      # model is behind: this block goes without it
        try:
            return self.outbox.get_nowait()
        except Empty:
            return dry, dry           # no voice reference this block, but no stall


def realtime_priority():
    """Ask for SCHED_RR like PipeWire clients do; harmless if not permitted."""
    try:
        os.sched_setscheduler(0, os.SCHED_RR, os.sched_param(10))
        return True
    except (AttributeError, PermissionError, OSError):
        return False


def interrupt(_number, _frame):
    raise KeyboardInterrupt


def stop_process(child):
    if child is not None and child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=1)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()


ALLY_SPEAKER_FIR = Path("/usr/share/cachyos-handheld/rog-ally/game.wav")


def is_virtual(item):
    """Filter-chain or null sinks. Their output follows the default sink, which
    is ours while live, so routing to one would feed our own input."""
    properties = item.get("properties", {})
    return properties.get("node.virtual") == "true" or properties.get("factory.name") == "support.null-audio-sink"


def ally_model():
    try:
        return Path("/sys/devices/virtual/dmi/id/product_name").read_text().strip()
    except OSError:
        return ""


def device_kind(item):
    """Best guess of what a sink is: (kind, speaker correction FIR or None)."""
    properties = item.get("properties", {})
    text = " ".join((item.get("name", ""), item.get("description", ""),
                     properties.get("device.description", ""), properties.get("device.product.name", ""))).lower()
    port = (item.get("active_port") or "").lower()
    form = properties.get("device.form_factor", "").lower()
    if "xreal" in text or "nreal" in text:
        return "glasses", None
    if "speaker" in port:
        internal = item.get("name", "").startswith("alsa_output.pci") and "ally" in ally_model().lower()
        return "speakers", str(ALLY_SPEAKER_FIR) if internal and ALLY_SPEAKER_FIR.is_file() else None
    if "headphone" in port or "headset" in port:
        return "headphones", None
    if form in ("speaker", "computer", "hifi", "tv", "car", "portable"):
        return "speakers", None
    if form in ("headphone", "headset", "handset", "hands-free", "earbuds"):
        return "headphones", None
    if "hdmi" in text or "displayport" in text:
        return "speakers", None  # TV or monitor
    return "headphones", None


def best_sink(available):
    """Highest-priority real output, the same pick WirePlumber would make."""
    candidates = [item for name, item in available.items() if name != SINK and not is_virtual(item)]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: int(item.get("properties", {}).get("priority.session") or 0),
    )["name"]


FULL_VOLUME = 65536


def sink_volume(item):
    """Raw volume (65536 = 100%) of a sink from `pactl --format=json`, or None."""
    try:
        return max(channel["value"] for channel in item["volume"].values())
    except (KeyError, TypeError, ValueError):
        return None


def set_volume(sink, raw):
    pulse("set-sink-volume", sink, str(int(raw)))


def recover_route():
    state = json.loads(ROUTE_STATE.read_text()) if ROUTE_STATE.exists() else {}
    target = state.get("target")
    available = sinks()
    # While live, the volume slider drives our sink and the device sits at
    # 100%. Hand the device the slider's current volume back.
    volume = sink_volume(available.get(SINK)) or state.get("volume")
    if target in available and volume:
        set_volume(target, volume)
    if pulse("get-default-sink") == SINK:
        fallback = target if target in available else best_sink(available)
        if fallback:
            pulse("set-default-sink", fallback)
    # PipeWire's pactl omits "index" from JSON module listings, so use the
    # tab-separated short form: index, name, argument.
    for line in pulse("list", "short", "modules").splitlines():
        index, name, argument = (line.split("\t") + ["", ""])[:3]
        if name == "module-null-sink" and f"sink_name={SINK}" in argument.split():
            pulse("unload-module", index)
    ROUTE_STATE.unlink(missing_ok=True)


def run_live(profile, neural):
    with (RUNTIME_DIR / "live.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        recover_route()
        process_live(profile, neural)


SURROUND_MAP = ",".join(CHANNELS)
# These streams go through pipewire-pulse, which by default accepts up to 4 MB
# before it starts dropping -- about 2.5 s of 8-channel audio. Whatever the DSP
# does not consume in time waits in that buffer, and since we then read and play
# at exactly real time, the offset never shrinks: one stall (first inference, a
# page fault, a scene hitch at game start) leaves the sound seconds behind the
# picture for the rest of the session. Cap the buffered audio instead, so a
# stall costs a short dropout rather than permanent lag.
RECORD_CAP = 49152   # bytes: 35 ms of 8-channel float32 at 44100
PLAYBACK_CAP = 24576  # bytes: 70 ms of stereo float32 at 44100
# A single block takes 2.9 ms, so a read that waits this long means the process
# or the graph stalled. Only re-open the ends if audio actually queued up behind
# us (see the backlog check in the loop) -- a quiet menu must not trigger it.
STALL_SECONDS = 1.0
COMMON = [
    "--raw", "--format=float32le", f"--rate={RATE}",
    "--latency-msec=20", "--process-time-msec=4", "--client-name=Ally Sound Lab",
]


def open_playback(target):
    return subprocess.Popen(
        ["/usr/bin/pacat", "--playback", f"--device={target}", "--volume=65536",
         "--property=node.dont-reconnect=true", "--property=node.dont-move=true",
         f"--property=pulse.attr.maxlength={PLAYBACK_CAP}",
         "--channels=2", "--channel-map=front-left,front-right", *COMMON],
        stdin=subprocess.PIPE,
        bufsize=0,
    )


def open_record():
    return subprocess.Popen(
        ["/usr/bin/parec", f"--device={SINK}.monitor",
         f"--property=pulse.attr.maxlength={RECORD_CAP}",
         f"--channels={len(CHANNELS)}", f"--channel-map={SURROUND_MAP}", *COMMON],
        stdout=subprocess.PIPE,
    )


def capture_streams(source, own_pid):
    """Move application streams from a physical output into the plugin sink."""
    available = sinks()
    if source not in available:
        return
    source_index = available[source]["index"]
    for stream in json.loads(pulse("--format=json", "list", "sink-inputs")):
        properties = stream["properties"]
        if (
            stream["sink"] == source_index
            and properties.get("application.process.id")
            and properties.get("application.name") != "Ally Sound Lab"
            and properties.get("application.process.id") != str(own_pid)
        ):
            try:
                pulse("move-sink-input", str(stream["index"]), SINK)
            except subprocess.CalledProcessError:
                pass


def read_commands(pending):
    """Control channel from the plugin backend: one JSON object per line."""
    for line in sys.stdin:
        try:
            command = json.loads(line)
        except ValueError:
            continue
        if command.get("profile") in PROFILES:
            pending["profile"] = command["profile"]
        for flag in ("bypass", "neural"):
            if isinstance(command.get(flag), bool):
                pending[flag] = command[flag]
        if isinstance(command.get("devices"), dict):
            pending["overrides"] = {
                str(name): kind for name, kind in command["devices"].items() if kind in DEVICES
            }


def process_live(profile, neural):
    realtime = realtime_priority()  # threads started below inherit it
    renderer = Renderer(
        MODELS / "ku100.sofa", make_vocals=lambda: PipelinedVocals(Vocals()), profile=profile, neural=neural,
    )
    pending = {}
    Thread(target=read_commands, args=(pending,), daemon=True).start()
    target = pulse("get-default-sink")
    if target == SINK or SINK in sinks():
        raise RuntimeError("Аудиовыход Ally Sound Lab уже существует.")
    module_id = None
    pipes = {"record": None, "playback": None}
    measured = missed = 0
    total = 0.0
    last_resync = 0.0
    health_check = monotonic()
    finished = Event()
    disconnect = Event()
    retarget = Queue()
    chunk_bytes = HOP * len(CHANNELS) * 4

    def route_to(new_target, initial=False):
        nonlocal target
        # Restart both ends so no stale audio piles up in the capture buffer.
        stop_process(pipes["playback"])
        stop_process(pipes["record"])
        # The system volume slider follows the default sink, which is ours.
        # Move the device's own volume onto our sink and run the device at
        # 100%, so the slider keeps its full range; give it back on leaving.
        available = sinks()
        slider = sink_volume(available.get(SINK))
        if not initial and target in available and slider:
            set_volume(target, slider)
        device_volume = sink_volume(available.get(new_target))
        if device_volume:
            set_volume(SINK, device_volume)
            set_volume(new_target, FULL_VOLUME)
        target = new_target
        ROUTE_STATE.write_text(json.dumps({"target": target, "volume": device_volume}))
        pipes["playback"] = open_playback(target)
        pipes["record"] = open_record()
        capture_streams(target, pipes["playback"].pid)
        apply_device(sinks().get(target, {}))

    def apply_device(item):
        auto, correction = device_kind(item)
        kind = overrides.get(target, auto)
        renderer.set_device(kind, correction if kind == "speakers" else None)
        device_state["current"] = (auto, correction)
        emit(phase="live", output=item.get("description", target), sink=target,
             device=kind, device_auto=auto, correction=bool(renderer.correction_enabled))

    overrides = {}
    device_state = {"current": None}

    def check_route():
        # Follow output changes like the stock audio stack would: a sink the
        # user selects, a newly connected device (Bluetooth, USB, HDMI), or a
        # fallback when the current device disappears.
        known = set(sinks())
        failures = 0
        while not finished.wait(1.5):
            try:
                available = sinks()
                default = pulse("get-default-sink")
                failures = 0
            except (subprocess.SubprocessError, OSError, ValueError):
                failures += 1
                if failures < 4:
                    continue
                available, default = {}, None
            if SINK not in available:
                disconnect.set()
                record = pipes["record"]
                if record is not None and record.poll() is None:
                    record.terminate()
                return
            added = {name: item for name, item in available.items() if name not in known}
            known = set(available)
            desired = None
            if target in available and device_kind(available[target]) != device_state["current"]:
                pending["redetect"] = True  # e.g. a 3.5 mm jack was plugged in
            if default != SINK and default in available and not is_virtual(available[default]):
                desired = default
            elif added and best_sink(added):
                desired = best_sink(added)
            elif target not in available:
                desired = best_sink(available)
            try:
                if default != SINK:
                    pulse("set-default-sink", SINK)
            except (subprocess.SubprocessError, OSError):
                pass
            if desired and desired != target:
                retarget.put(desired)

    try:
        module_id = pulse(
            "load-module", "module-null-sink", f"sink_name={SINK}",
            f"rate={RATE}", f"channels={len(CHANNELS)}", f"channel_map={SURROUND_MAP}",
            "sink_properties=device.description=Ally_Sound_Lab",
        )
        pulse("set-default-sink", SINK)
        route_to(target, initial=True)
        Thread(target=check_route, daemon=True).start()
        while True:
            if not retarget.empty():
                new_target = None
                while not retarget.empty():
                    new_target = retarget.get_nowait()
                if new_target != target:
                    route_to(new_target)
            wait_started = monotonic()
            chunk = pipes["record"].stdout.read(chunk_bytes)
            if monotonic() - wait_started > STALL_SECONDS and monotonic() - last_resync > 3:
                if select.select([pipes["record"].stdout.fileno()], [], [], 0)[0]:
                    # Starved and more audio is already waiting: that backlog is stale,
                    # so start over instead of replaying it seconds late.
                    last_resync = monotonic()
                    route_to(target)
                    continue
            if disconnect.is_set():
                raise RuntimeError("PipeWire удалил аудиовыход Ally Sound Lab.")
            if len(chunk) != chunk_bytes:
                raise RuntimeError("PipeWire прервал аудиопоток.")
            if pending:
                if "profile" in pending:
                    renderer.set_profile(pending.pop("profile"))
                if "bypass" in pending:
                    renderer.bypass = pending.pop("bypass")
                if "neural" in pending:
                    renderer.neural = pending.pop("neural")
                if "overrides" in pending:
                    overrides.clear()
                    overrides.update(pending.pop("overrides"))
                    pending["redetect"] = True
                if pending.pop("redetect", False):
                    apply_device(sinks().get(target, {}))
                health_check = 0
            tick = perf_counter()
            block = np.frombuffer(chunk, dtype="<f4").reshape(HOP, len(CHANNELS)).T
            result = renderer.process(block).T.copy().tobytes()
            elapsed_ms = (perf_counter() - tick) * 1000
            measured += 1
            missed += elapsed_ms > HOP / RATE * 1000
            total += elapsed_ms
            write_started = monotonic()
            try:
                written = pipes["playback"].stdin.write(result)
            except BrokenPipeError:
                written = None
            if written == len(result) and monotonic() - write_started > STALL_SECONDS:
                # The player is sitting on a queue we cannot drain: start over.
                route_to(target)
                continue
            if written != len(result):
                # The physical output vanished (e.g. headphones turned off).
                # Give the route checker a moment to pick the fallback device.
                try:
                    new_target = retarget.get(timeout=5)
                except Empty:
                    raise RuntimeError("Аудиовыход пропал, замены не нашлось.") from None
                route_to(new_target)
                continue
            if monotonic() - health_check >= 2:
                emit(
                    phase="live", avg_ms=round(total / measured, 2), slow_blocks=missed, blocks=measured,
                    source=renderer.source, profile=renderer.profile, bypass=renderer.bypass,
                    neural=renderer.neural, realtime=realtime,
                )
                health_check = monotonic()
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        finished.set()
        stop_process(pipes["record"])
        stop_process(pipes["playback"])
        if module_id is not None:
            recover_route()


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    live = commands.add_parser("live")
    live.add_argument("profile", choices=tuple(PROFILES))
    live.add_argument("--no-neural", dest="neural", action="store_false")
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    try:
        run_live(args.profile, args.neural)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        emit(phase="error", error=str(exc))
        raise


if __name__ == "__main__":
    main()
