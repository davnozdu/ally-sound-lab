import argparse
import fcntl
import json
from pathlib import Path
from queue import Empty, Queue
import signal
import subprocess
from threading import Event, Thread
from time import monotonic, perf_counter

import h5py
import numpy as np
import onnxruntime as ort
from scipy import signal as dsp

RATE = 44100
HOP = 128
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


def impulse_responses():
    with h5py.File(MODELS / "ku100.sofa", "r") as file:
        positions = file["SourcePosition"][:]
        indices = [
            np.argmin(np.abs((positions[:, 0] - angle + 180) % 360 - 180) + np.abs(positions[:, 1]))
            for angle in (30, 330, 110, 250)
        ]
        responses = np.stack([file["Data.IR"][index] for index in indices])
    return dsp.resample_poly(responses, 147, 160, axis=-1).astype(np.float32)


def surround_inputs(stereo, mode, auxiliary=None):
    if mode == "hrtf":
        side = (stereo[0] - stereo[1]) * 0.325
        surround = np.stack((side, -side))
    else:
        surround = auxiliary * 0.22
    return np.vstack((stereo, surround))


class Headphones:
    def __init__(self):
        self.responses = impulse_responses()
        self.state = np.zeros((4, 2, self.responses.shape[-1] - 1), dtype=np.float32)
        self.denominator = np.ones(1, dtype=np.float32)

    def render(self, sources):
        ears = np.zeros((2, sources.shape[-1]), dtype=np.float32)
        for source in range(4):
            for ear in range(2):
                result, self.state[source, ear] = dsp.lfilter(
                    self.responses[source, ear],
                    self.denominator,
                    sources[source],
                    zi=self.state[source, ear],
                )
                ears[ear] += result
        return ears


class Vocals:
    def __init__(self):
        self.session = runtime("vocals.onnx", 1)
        self.state = {
            item.name: np.zeros(item.shape, dtype=np.float32)
            for item in self.session.get_inputs()[1:]
        }
        self.previous = np.zeros((2, HOP), dtype=np.float32)

    def process(self, audio):
        vocals, *states = self.session.run(
            None, {"audio_chunk": np.ascontiguousarray(audio[None]), **self.state}
        )
        self.state = dict(zip(self.state, states))
        # The graph emits the previous input hop after its initial pre-roll.
        dry = self.previous
        self.previous = audio
        return dry, dry - vocals[0, 0]


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


def best_sink(available):
    """Highest-priority real output, the same pick WirePlumber would make."""
    candidates = [item for name, item in available.items() if name != SINK]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: int(item.get("properties", {}).get("priority.session") or 0),
    )["name"]


def recover_route():
    state = json.loads(ROUTE_STATE.read_text()) if ROUTE_STATE.exists() else {}
    target = state.get("target")
    available = sinks()
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


def run_live(mode):
    with (RUNTIME_DIR / "live.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        recover_route()
        process_live(mode)


COMMON = [
    "--raw", "--format=float32le", f"--rate={RATE}", "--channels=2",
    "--latency-msec=28", "--process-time-msec=4", "--client-name=Ally Sound Lab",
]


def open_playback(target):
    return subprocess.Popen(
        ["/usr/bin/pacat", "--playback", f"--device={target}", "--volume=65536",
         "--property=node.dont-reconnect=true", "--property=node.dont-move=true", *COMMON],
        stdin=subprocess.PIPE,
        bufsize=0,
    )


def open_record():
    return subprocess.Popen(
        ["/usr/bin/parec", f"--device={SINK}.monitor", *COMMON],
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


def process_live(mode):
    renderer = Headphones()
    model = Vocals() if mode == "stemgen" else None
    target = pulse("get-default-sink")
    if target == SINK or SINK in sinks():
        raise RuntimeError("Аудиовыход Ally Sound Lab уже существует.")
    module_id = None
    pipes = {"record": None, "playback": None}
    measured = missed = 0
    total = 0.0
    health_check = monotonic()
    finished = Event()
    disconnect = Event()
    retarget = Queue()
    chunk_bytes = HOP * 2 * 4

    def route_to(new_target):
        nonlocal target
        # Restart both ends so no stale audio piles up in the capture buffer.
        stop_process(pipes["playback"])
        stop_process(pipes["record"])
        target = new_target
        ROUTE_STATE.write_text(json.dumps({"target": target}))
        pipes["playback"] = open_playback(target)
        pipes["record"] = open_record()
        capture_streams(target, pipes["playback"].pid)
        emit(phase="live", output=sinks().get(target, {}).get("description", target))

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
            if default != SINK and default in available:
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
            f"rate={RATE}", "channels=2", "channel_map=front-left,front-right",
            "sink_properties=device.description=Ally_Sound_Lab",
        )
        pulse("set-default-sink", SINK)
        route_to(target)
        Thread(target=check_route, daemon=True).start()
        while True:
            if not retarget.empty():
                new_target = None
                while not retarget.empty():
                    new_target = retarget.get_nowait()
                if new_target != target:
                    route_to(new_target)
            chunk = pipes["record"].stdout.read(chunk_bytes)
            if disconnect.is_set():
                raise RuntimeError("PipeWire удалил аудиовыход Ally Sound Lab.")
            if len(chunk) != chunk_bytes:
                raise RuntimeError("PipeWire прервал аудиопоток.")
            tick = perf_counter()
            dry = np.frombuffer(chunk, dtype="<f4").reshape(HOP, 2).T
            auxiliary = None
            if model is not None:
                dry, auxiliary = model.process(dry)
            rendered = renderer.render(surround_inputs(dry, mode, auxiliary))
            result = np.clip(rendered * 0.70, -0.98, 0.98).T.copy().tobytes()
            elapsed_ms = (perf_counter() - tick) * 1000
            measured += 1
            missed += elapsed_ms > HOP / RATE * 1000
            total += elapsed_ms
            try:
                written = pipes["playback"].stdin.write(result)
            except BrokenPipeError:
                written = None
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
                emit(phase="live", avg_ms=round(total / measured, 2), slow_blocks=missed, blocks=measured)
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
    live.add_argument("mode", choices=("hrtf", "stemgen"))
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    try:
        run_live(args.mode)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        emit(phase="error", error=str(exc))
        raise


if __name__ == "__main__":
    main()
