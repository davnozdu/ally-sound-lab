"""Binaural virtual-surround renderer for headphones.

Input: one block of 7.1 audio (FL FR FC LFE RL RR SL SR), HOP samples.
Output: one block of binaural stereo.

Multichannel sources are placed as discrete virtual speakers. Stereo sources
are upmixed first: StemgenRT pulls the voice into the centre speaker and the
ambience of the remaining mix is spread to the sides and rear.
"""

import h5py
import numpy as np
from scipy import signal as dsp

RATE = 44100
HOP = 128
CHANNELS = (
    "front-left", "front-right", "front-center", "lfe",
    "rear-left", "rear-right", "side-left", "side-right",
)
FL, FR, FC, LFE, RL, RR, SL, SR = range(8)

# Virtual speakers fed to the HRTF stage, with SOFA azimuths (degrees,
# counter-clockwise from straight ahead, so positive is to the left).
SPEAKERS = ((FL, 30), (FR, 330), (FC, 0), (RL, 145), (RR, 215), (SL, 100), (SR, 260))

PROFILES = {
    # Precise localisation: small dry room, untouched dynamics.
    "game": {"dialogue": 1.0, "room": 0.10, "rt60": 0.22, "compress": False},
    # Clear dialogue over effects, bigger room, tamed loud scenes.
    "movie": {"dialogue": 10 ** (5 / 20), "room": 0.16, "rt60": 0.40, "compress": True},
}
ROOM_SECONDS = 0.45
SILENCE = 1e-5          # RMS below this counts as an unused channel (-100 dBFS)
SURROUND_HOLD = 1.0     # seconds a surround channel stays "seen" after going quiet
AMBIENCE_DELAY = 0.015  # rear ambience delay, decorrelates it from the front


def load_hrirs(path):
    with h5py.File(path, "r") as file:
        positions = file["SourcePosition"][:]
        rate = float(np.asarray(file["Data.SamplingRate"][:]).ravel()[0])
        chosen = []
        for _, azimuth in SPEAKERS:
            distance = np.abs((positions[:, 0] - azimuth + 180) % 360 - 180) + 2 * np.abs(positions[:, 1])
            chosen.append(int(np.argmin(distance)))
        responses = np.stack([file["Data.IR"][index] for index in chosen])
        picked = positions[chosen, :2]
    if round(rate) != RATE:
        responses = dsp.resample_poly(responses, RATE, round(rate), axis=-1)
    return responses.astype(np.float32), picked


def room_response(rt60, seed):
    """Synthetic small-room tail: early reflections plus a decaying diffuse tail.

    Shape (2 sends, 2 ears, samples); each path uses independent noise so the
    room is decorrelated between the ears.
    """
    length = int(ROOM_SECONDS * RATE)
    rng = np.random.default_rng(seed)
    time = np.arange(length) / RATE
    low_b, low_a = dsp.butter(2, 2500 / (RATE / 2))
    air_b, air_a = dsp.butter(1, 9000 / (RATE / 2))
    onset = np.clip((time - 0.008) / 0.02, 0, 1)  # tail builds up after 8 ms
    responses = np.zeros((2, 2, length), dtype=np.float64)
    for send in range(2):
        for ear in range(2):
            noise = rng.standard_normal(length)
            low = dsp.lfilter(low_b, low_a, noise)
            high = noise - low
            # Highs die out faster than lows, as in a furnished room.
            tail = low * np.exp(-6.91 * time / rt60) + high * np.exp(-6.91 * time / (rt60 * 0.45))
            tail = dsp.lfilter(air_b, air_a, tail) * onset
            same_side = send == ear
            for _ in range(10):
                delay = rng.uniform(0.003, 0.03)
                gain = rng.uniform(0.3, 0.7) * (1.0 if same_side else 0.55) * (1 - delay / 0.04)
                tail[int(delay * RATE)] += gain * rng.choice((-1.0, 1.0)) * 6
            responses[send, ear] = tail
    responses /= np.sqrt(np.sum(responses ** 2, axis=-1, keepdims=True).mean())
    return responses.astype(np.float32)


class Convolver:
    """Uniformly partitioned overlap-save convolution, S inputs to 2 ears."""

    def __init__(self, responses):
        sources, ears, length = responses.shape
        self.parts = -(-length // HOP)
        self.spectra = self._spectra(responses)
        self.history = np.zeros((self.parts, sources, HOP + 1), dtype=np.complex64)
        self.previous = np.zeros((sources, HOP), dtype=np.float32)

    def _spectra(self, responses):
        sources, ears, length = responses.shape
        padded = np.zeros((sources, ears, self.parts * HOP), dtype=np.float32)
        padded[..., :length] = responses
        parts = padded.reshape(sources, ears, self.parts, HOP).transpose(2, 0, 1, 3)
        return np.fft.rfft(parts, n=2 * HOP, axis=-1).astype(np.complex64)

    def set_responses(self, responses):
        self.spectra = self._spectra(responses)

    def process(self, block):
        frame = np.concatenate((self.previous, block), axis=-1)
        self.previous = block
        self.history[1:] = self.history[:-1]
        self.history[0] = np.fft.rfft(frame, axis=-1)
        # Plain multiply-and-sum is ~5x faster than einsum for these shapes.
        spectrum = (self.history[:, :, None, :] * self.spectra).sum(axis=(0, 1))
        return np.fft.irfft(spectrum, n=2 * HOP, axis=-1)[:, HOP:].astype(np.float32)


class Delay:
    def __init__(self, channels, samples):
        self.buffer = np.zeros((channels, samples), dtype=np.float32)

    def process(self, block):
        joined = np.concatenate((self.buffer, block), axis=-1)
        self.buffer = joined[:, block.shape[-1]:]
        return joined[:, :block.shape[-1]]


class Dynamics:
    """Optional speech-friendly compressor followed by a peak limiter."""

    THRESHOLD_DB = -30.0
    RATIO = 3.0
    MAKEUP_DB = 3.0  # keeps "movie" loudness-matched to bypass for a fair A/B
    CEILING = 0.89

    def __init__(self):
        self.level_db = -90.0
        self.gain = 1.0
        self.attack = np.exp(-HOP / (RATE * 0.010))
        self.release = np.exp(-HOP / (RATE * 0.300))

    def process(self, block, compress):
        target = 1.0
        if compress:
            level = 10 * np.log10(np.mean(block ** 2) + 1e-12)
            coefficient = self.attack if level > self.level_db else self.release
            self.level_db = coefficient * self.level_db + (1 - coefficient) * level
            over = max(0.0, self.level_db - self.THRESHOLD_DB)
            target = 10 ** ((self.MAKEUP_DB - over * (1 - 1 / self.RATIO)) / 20)
        peak = float(np.max(np.abs(block))) * target
        if peak > self.CEILING:
            target *= self.CEILING / peak
        if target < self.gain:
            ramp = np.full(HOP, target, dtype=np.float32)  # limiter: act at once
        else:
            target = self.gain + (target - self.gain) * 0.05  # smooth recovery
            ramp = np.linspace(self.gain, target, HOP, dtype=np.float32)
        self.gain = target
        out = block * ramp
        # Last-resort soft clip, only touches what the limiter could not catch.
        magnitude = np.abs(out)
        knee = 0.95
        return np.where(
            magnitude > knee,
            np.sign(out) * (knee + 0.05 * np.tanh((magnitude - knee) / 0.05)),
            out,
        ).astype(np.float32)


class Renderer:
    def __init__(self, hrir_path, vocals=None, profile="game", neural=True):
        hrirs, self.positions = load_hrirs(hrir_path)
        self.direct = Convolver(hrirs)
        self.rooms = {name: room_response(values["rt60"], seed=7) for name, values in PROFILES.items()}
        self.room = Convolver(self.rooms[profile])
        self.vocals = vocals
        self.ambience_delay = Delay(2, int(AMBIENCE_DELAY * RATE))
        self.dynamics = Dynamics()
        lfe_b, lfe_a = dsp.butter(2, 120 / (RATE / 2))
        self.lfe_filter = (lfe_b, lfe_a, np.zeros(2))
        self.surround_seen = -1e9
        self.clock = 0.0
        self.bypass = False
        self.source = "stereo"
        self.model_active = False
        self.neural = neural  # StemgenRT for stereo; otherwise a passive matrix
        self.set_profile(profile)

    def set_profile(self, profile):
        if profile not in PROFILES:
            raise ValueError(profile)
        self.profile = profile
        self.settings = PROFILES[profile]
        self.room.set_responses(self.rooms[profile])

    # -- source analysis ------------------------------------------------

    def _detect(self, block):
        self.clock += HOP / RATE
        rms = np.sqrt(np.mean(block[[FC, LFE, RL, RR, SL, SR]] ** 2, axis=-1))
        if np.any(rms > SILENCE):
            self.surround_seen = self.clock
        return "surround" if self.clock - self.surround_seen < SURROUND_HOLD else "stereo"

    # -- speaker feeds ----------------------------------------------------

    def _surround_feeds(self, block):
        feeds = block[[FL, FR, FC, RL, RR, SL, SR]].copy()
        feeds[2] *= self.settings["dialogue"]
        sides = np.sqrt(np.mean(block[[SL, SR]] ** 2))
        if sides <= SILENCE:
            # 5.1: its surround pair sits at ~110 degrees, not behind.
            feeds[5:7], feeds[3:5] = feeds[3:5], 0.0
        return feeds, block[LFE]

    def _stereo_feeds(self, block):
        dry = block[[FL, FR]]
        if self.neural and self.vocals is not None:
            dry, rest = self.vocals.process(np.ascontiguousarray(dry))
            voice = dry - rest
        else:
            # Passive upmix: half of the shared (mid) signal becomes the centre.
            # Dialogue is usually centre-panned, so it still lands in front.
            mid = 0.5 * (dry[0] + dry[1])
            voice = np.stack((0.5 * mid, 0.5 * mid))
            rest = dry - voice
        # Ambience = what is not shared by both channels of the remaining mix.
        ambience = np.stack((rest[0] - 0.7 * rest[1], rest[1] - 0.7 * rest[0]))
        rear = self.ambience_delay.process(ambience)
        feeds = np.stack((
            rest[0],
            rest[1],
            voice.mean(axis=0) * self.settings["dialogue"],
            rear[0] * 0.40,
            rear[1] * 0.40,
            ambience[0] * 0.45,
            ambience[1] * 0.45,
        )).astype(np.float32)
        return feeds, None

    # -- rendering ------------------------------------------------------

    def _render(self, feeds, lfe):
        ears = self.direct.process(feeds)
        send = np.stack((
            feeds[0] + 0.7 * feeds[2] + feeds[5] + feeds[3],
            feeds[1] + 0.7 * feeds[2] + feeds[6] + feeds[4],
        )) * 0.5
        ears += self.room.process(send.astype(np.float32)) * self.settings["room"]
        if lfe is not None:
            b, a, state = self.lfe_filter
            low, state = dsp.lfilter(b, a, lfe, zi=state)
            self.lfe_filter = (b, a, state)
            ears += (low * 1.4).astype(np.float32)
        return ears * 0.85

    @staticmethod
    def _downmix(block):
        """Plain stereo fold-down for A/B comparison (what you'd hear without us)."""
        centre = 0.707 * block[FC]
        left = block[FL] + centre + 0.707 * (block[SL] + block[RL])
        right = block[FR] + centre + 0.707 * (block[SR] + block[RR])
        return np.stack((left, right)).astype(np.float32) * 0.6

    def process(self, block):
        """block: float32 (8, HOP). Returns float32 (2, HOP)."""
        self.source = self._detect(block)
        uses_model = not self.bypass and self.source == "stereo" and self.neural
        if uses_model and not self.model_active and hasattr(self.vocals, "reset"):
            self.vocals.reset()  # resume after a pause: drop the stale block
        self.model_active = uses_model
        if self.bypass:
            out = self._downmix(block)
        else:
            if self.source == "surround":
                feeds, lfe = self._surround_feeds(block)
            else:
                feeds, lfe = self._stereo_feeds(block)
            out = self._render(feeds, lfe)
        return self.dynamics.process(out, self.settings["compress"] and not self.bypass)
