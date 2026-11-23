"""Background audio classification and gentle EQ for the live renderer."""

import math
import os
import sys
from collections import deque
from queue import Empty, Full, Queue
from threading import Thread

import numpy as np
import onnxruntime as ort
from scipy import fft, signal
from scipy.special import expit

MODEL_RATE = 16000
PATCH_SAMPLES = 15600
# YAMNet AudioSet class indices in the Qualcomm ONNX export's labels.txt.
SPEECH = (0, 1, 2, 3, 5)
MUSIC = (24, 132)
GENRES = {
    "metal": (215,),
    "rock": (214, 216, 217, 218, 219, 220),
    "jazz": (225, 230),
    "classical": (232, 233),
    "electronic": (234, 235, 236, 237, 238, 239, 240, 241, 242),
    "hiphop": (212,),
}
# Discogs-EffNet (Essentia, ISMIR 2022): 400 Discogs styles from MusiCNN-style mel
# patches of 128 frames. Each style maps to an EQ group, in the model's class order:
# 0 other music, 1 metal, 2 rock, 3 jazz, 4 classical/score/soundtrack, 5 electronic,
# 6 hip hop. Same table and thresholds as the phone module (dsp/auto_eq.cpp).
STYLE_GROUPS = np.frombuffer(
    b"0000000000000000004444444444444555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555500000000000000000000000000000000000000000066666666666666666666666666333333333333333333333333300000000000000000000000000000000000000000000000000000000000000000000000000022222212212222221121212222121122121122212222221212222221222222212122212222222212221222112124440", dtype=np.uint8) - ord("0")
GROUP_PROFILES = (None, "metal", "rock", "jazz", "classical", "electronic", "hiphop")
GENRE_MINIMUM = 0.15
GENRE_MARGIN = 1.25
GENRE_PATCHES = 3
GENRE_PATCH_FRAMES = 128
GENRE_PATCH_HOP = 56

# Battery: classify in short probes. Once the result is stable (and, for music, its
# genre has been checked) the classifier rests this many windows (~20 s).
REST_WINDOWS = 20
# Release the ONNX sessions after this much silence.
IDLE_RELEASE_SECONDS = 10.0

# Gains in dB for the low, presence and high bands, relative to the rendered output.
PROFILE_GAINS_DB = {
    "neutral": (0.0, 0.0, 0.0),
    "speech": (-1.8, 2.0, 0.0),
    "music": (0.0, 0.0, 0.0),
    "metal": (0.5, 0.2, -1.0),
    "rock": (0.5, 0.8, -0.2),
    "jazz": (0.0, -0.4, 0.8),
    "classical": (0.0, 0.0, 0.5),
    "electronic": (1.0, -0.4, 0.3),
    "hiphop": (1.2, 0.4, 0.0),
}


def select_profile(scores):
    speech = max(scores[i] for i in SPEECH)
    ranked = sorted(
        ((max(scores[i] for i in ids), genre) for genre, ids in GENRES.items()),
        reverse=True,
    )
    genre_score, genre = ranked[0]
    music = max(genre_score, *(scores[i] for i in MUSIC))
    if speech >= 0.55 and speech > music + 0.14:
        return "speech"
    if music >= 0.50:
        if genre_score >= 0.42 and genre_score > ranked[1][0] + 0.07 and speech < 0.55:
            return genre
        return "music"
    return "neutral"


def style_groups(scores):
    """Average the patches, then score each EQ group by its strongest style."""
    mean = np.asarray(scores, dtype=np.float32).reshape(GENRE_PATCHES, -1).mean(axis=0)
    return np.array([mean[STYLE_GROUPS == g].max() for g in range(len(GROUP_PROFILES))], dtype=np.float32)


def genre_profile(groups, fallback):
    """Decide on group scores averaged over a probe; unsure results keep `fallback`."""
    if not np.isfinite(groups).all():
        return fallback
    ranked = sorted(((groups[g], GROUP_PROFILES[g]) for g in range(1, len(GROUP_PROFILES))), reverse=True)
    score, genre = ranked[0]
    # Metal styles also light up rock; let a clear metal score win over its parent.
    if genre == "rock" and groups[1] >= GENRE_MINIMUM and groups[1] >= 0.6 * groups[2]:
        return "metal"
    competitor = max(ranked[1][0], groups[0])
    return genre if score >= GENRE_MINIMUM and score > competitor * GENRE_MARGIN else fallback


def cpu_session(model):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return ort.InferenceSession(str(model), opts, providers=["CPUExecutionProvider"])


class Yamnet:
    def __init__(self, model, rate):
        self.session = cpu_session(model)
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if len(inputs) != 1 or inputs[0].shape != [1, 1, 96, 64] or len(outputs) != 1 or outputs[0].shape != [1, 521]:
            raise ValueError("Unexpected YAMNet tensor shapes")
        self.input = inputs[0].name
        divisor = math.gcd(rate, MODEL_RATE)
        self.ratio = MODEL_RATE // divisor, rate // divisor
        self.window = signal.windows.hann(400, sym=False).astype(np.float32)
        bins = 1127 * np.log1p(np.fft.rfftfreq(512, 1 / MODEL_RATE) / 700)
        limits = np.linspace(1127 * np.log1p(125 / 700), 1127 * np.log1p(7500 / 700), 66)
        rise = (bins[:, None] - limits[:-2]) / (limits[1:-1] - limits[:-2])
        fall = (limits[2:] - bins[:, None]) / (limits[2:] - limits[1:-1])
        self.mel = np.maximum(0, np.minimum(rise, fall)).astype(np.float32)

    def resample(self, audio):
        return signal.resample_poly(audio, *self.ratio)[:PATCH_SAMPLES]

    def features(self, mono):
        frames = np.lib.stride_tricks.sliding_window_view(mono, 400)[::160]
        magnitude = np.abs(fft.rfft(frames * self.window, n=512, axis=-1))
        return np.log(magnitude @ self.mel + 0.001)[None, None]

    def classify(self, audio):
        rms = np.sqrt(np.mean(audio ** 2))
        if not np.isfinite(rms) or rms < 1e-3:
            return "neutral"
        # The export exposes the logits before the AudioSet sigmoid.
        logits = self.session.run(None, {self.input: self.features(audio)})[0][0]
        return select_profile(expit(logits)) if np.isfinite(logits).all() else "neutral"


class DiscogsEffNet:
    """Discogs-EffNet on MusiCNN-style features: three overlapping 128-frame patches."""

    FRAME_COUNT = GENRE_PATCH_FRAMES + (GENRE_PATCHES - 1) * GENRE_PATCH_HOP
    SAMPLES = 512 + (FRAME_COUNT - 1) * 256

    def __init__(self, model):
        self.session = cpu_session(model)
        self.window = signal.windows.hann(512).astype(np.float32)
        # Essentia uses Slaney-spaced boundaries with unit-area triangles in Hz.
        bins = np.fft.rfftfreq(512, 1 / MODEL_RATE)
        transition = 15
        log_step = np.log(6.4) / 27
        upper = transition + np.log(8) / log_step
        points = np.linspace(0, upper, 98)
        limits = np.where(points <= transition, points * 1000 / transition,
                          1000 * np.exp(log_step * (points - transition)))
        rise = (bins[:, None] - limits[:-2]) / (limits[1:-1] - limits[:-2])
        fall = (limits[2:] - bins[:, None]) / (limits[2:] - limits[1:-1])
        self.mel = (2 * np.maximum(0, np.minimum(rise, fall)) / (limits[2:] - limits[:-2])).astype(np.float32)

    def features(self, audio):
        frames = np.lib.stride_tricks.sliding_window_view(audio[-self.SAMPLES:], 512)[::256]
        power = np.abs(fft.rfft(frames * self.window, axis=-1)) ** 2
        mel = np.log10(1 + 10000 * (power @ self.mel)).astype(np.float32)
        return np.stack([mel[i * GENRE_PATCH_HOP:i * GENRE_PATCH_HOP + GENRE_PATCH_FRAMES]
                         for i in range(GENRE_PATCHES)])

    def groups(self, audio):
        scores = self.session.run(["activations"], {"melspectrogram": self.features(audio)})[0]
        return style_groups(scores)


class AutoEqualizer:
    def __init__(self, model, rate, hop):
        self.enabled = False
        self._model = model
        self._rate = rate
        self._generation = 0
        self._result = (0, "neutral", None)
        self._jobs = Queue(maxsize=1)
        self._thread = None
        self._samples = np.empty(math.ceil(PATCH_SAMPLES * rate / MODEL_RATE), dtype=np.float32)
        self._offset = 0
        self._rest = 0           # windows to skip before the next probe (set by the worker)
        self._release = False    # drop the models at the next worker wake-up

        # Parallel filtered corrections keep the signal path at unity when neutral.
        self._filters = [
            tuple(coef.astype(np.float32) for coef in signal.butter(2, cutoff, btype=kind, fs=rate))
            for cutoff, kind in ((180, "lowpass"), ((1000, 4200), "bandpass"), (5200, "highpass"))
        ]
        self._state = [np.zeros((2, len(a) - 1), dtype=np.float32) for _, a in self._filters]
        self._gains = np.zeros(3, dtype=np.float32)
        self._correcting = False
        self._targets = {name: np.float32(10 ** (np.asarray(db) / 20) - 1) for name, db in PROFILE_GAINS_DB.items()}
        self._ramp = np.arange(1, hop + 1, dtype=np.float32) / hop
        self._slew = np.float32(1 - math.exp(-hop / (rate * 0.65)))

    @property
    def profile(self):
        generation, profile, _ = self._result
        return profile if generation == self._generation else "neutral"

    @property
    def error(self):
        generation, _, error = self._result
        return error if generation == self._generation else None

    def set_enabled(self, enabled):
        if enabled == self.enabled:
            return
        self._generation += 1
        self.enabled = enabled
        self._offset = 0
        self._rest = 0
        if enabled and self._thread is None:
            self._thread = Thread(target=self._run, name="Ally Auto EQ", daemon=True)
            self._thread.start()
        elif not enabled:
            # Switched off: give the model memory back.
            self._release = True

    def observe(self, stereo):
        if not self.enabled or self.error is not None:
            return
        mono = (stereo[0] + stereo[1]) * 0.5
        take = min(len(mono), len(self._samples) - self._offset)
        self._samples[self._offset:self._offset + take] = mono[:take]
        self._offset += take
        if self._offset == len(self._samples):
            if self._rest > 0:
                self._rest -= 1        # resting between probes: drop the window
            else:
                try:
                    self._jobs.put_nowait((self._generation, self._samples.copy()))
                except Full:
                    pass  # playback takes priority over an outstanding classification
            self._samples[:len(mono) - take] = mono[take:]
            self._offset = len(mono) - take

    def _run(self):
        # The renderer can use SCHED_RR. The classifier must stay below it.
        if hasattr(os, "sched_setscheduler"):
            try:
                os.sched_setscheduler(0, os.SCHED_OTHER, os.sched_param(0))
            except OSError:
                pass
        model = genre_model = None
        recent_music = deque(maxlen=4)
        generation = -1
        vote, count = "neutral", 0
        group_sum, group_runs, genre_ran = np.zeros(len(GROUP_PROFILES), np.float32), 0, False
        silent = 0
        idle_windows = int(IDLE_RELEASE_SECONDS * self._rate / len(self._samples)) + 1
        while True:
            try:
                job = self._jobs.get(timeout=IDLE_RELEASE_SECONDS)
            except Empty:
                job = None             # no audio at all for a while (paused/stopped)
            if job is None or self._release:
                self._release = False
                model = genre_model = None
                recent_music.clear()
                if job is None:
                    continue
            token, audio = job
            if token != self._generation:
                continue
            if generation != token:
                generation = token
                vote, count = "neutral", 0
                recent_music.clear()
                group_sum[:], group_runs, genre_ran = 0, 0, False
            rms = float(np.sqrt(np.mean(audio ** 2)))
            audible = np.isfinite(rms) and rms >= 1e-3
            silent = 0 if audible else silent + 1
            if not audible and silent >= idle_windows:
                model = genre_model = None
            candidate = "neutral"
            try:
                if audible:
                    if model is None:
                        model = Yamnet(self._model, self._rate)
                    mono = model.resample(audio)
                    candidate = model.classify(mono)
                    if candidate == "music" or candidate in GENRES:
                        recent_music.append(mono)
                        if len(recent_music) == recent_music.maxlen:
                            if genre_model is None:
                                genre_model = DiscogsEffNet(
                                    self._model.with_name("discogs-effnet-bsdynamic-1.onnx"))
                            group_sum += genre_model.groups(np.concatenate(recent_music))
                            group_runs += 1
                            genre_ran = True
                            candidate = genre_profile(group_sum / group_runs, candidate)
                    else:
                        recent_music.clear()
                else:
                    recent_music.clear()
            except Exception as exc:
                # A model or runtime failure leaves playback on the neutral curve.
                print(f"Auto EQ: {exc}", file=sys.stderr, flush=True)
                self._result = (token, "neutral", str(exc))
                continue
            if token != self._generation:
                continue
            if candidate == vote:
                count += 1
            else:
                vote, count = candidate, 1
            musical = candidate == "music" or candidate in GENRES
            # Music only counts once its genre has been checked in this probe.
            if count >= 2 and (not musical or genre_ran):
                # An unsure probe keeps the genre an earlier probe found.
                previous = self.profile
                if not (candidate == "music" and previous in GENRES):
                    self._result = (token, candidate, None)
                self._rest = REST_WINDOWS
                recent_music.clear()       # the next probe needs fresh, contiguous audio
                group_sum[:], group_runs, genre_ran = 0, 0, False
                count = 0

    def apply(self, stereo):
        profile = self.profile
        flat = profile in ("neutral", "music")
        if flat and not self._correcting:
            return stereo
        if flat and np.max(np.abs(self._gains)) < 1e-5:
            self._correcting = False
            self._gains.fill(0)
            for state in self._state:
                state.fill(0)
            return stereo
        self._correcting = True
        target = self._targets[profile]
        change = (target - self._gains) * self._slew
        output = stereo.copy()
        for i, (b, a) in enumerate(self._filters):
            band, self._state[i] = signal.lfilter(b, a, stereo, axis=-1, zi=self._state[i])
            output += band * (self._gains[i] + change[i] * self._ramp)
        self._gains += change
        return output
