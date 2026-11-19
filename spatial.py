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
    # Voices in 5.1/7.1 stay where the game put them (an enemy behind you).
    # The neural stereo upmix costs ~90% of a core, i.e. right at the 2.9 ms
    # block budget, so in a game it drops blocks and the centre channel pumps --
    # audible as crackle -- while stealing a core from the game. Games get the
    # passive matrix; the network stays for films, which are not interactive.
    "game": {"dialogue": 1.0, "room": 0.10, "rt60": 0.22, "compress": False,
             "dub_fix": False, "stereo_model": False},
    # Clear dialogue over effects, bigger room, tamed loud scenes. In 5.1/7.1
    # the network gathers voice from every channel into the centre, which
    # repairs dubs that were smeared over all speakers.
    "movie": {"dialogue": 10 ** (2.0 / 20), "room": 0.16, "rt60": 0.40, "compress": True,
              "dub_fix": True, "stereo_model": True},
}
ROOM_SECONDS = 0.45
SILENCE = 1e-5          # RMS below this counts as an unused channel (-100 dBFS)
SURROUND_HOLD = 1.0     # seconds a surround channel stays "seen" after going quiet
AMBIENCE_DELAY = 0.015  # rear ambience delay, decorrelates it from the front
# Five full-range speakers sum louder than the stereo fold-down; this trim
# keeps 5.1/7.1 loudness-matched to the A/B bypass (measured on Ally).
SURROUND_TRIM = 10 ** (-3.7 / 20)
# Stereo is matched to the untouched input (bypass passes it through as is);
# the render was measured 4.4 dB below that.
STEREO_TRIM = 10 ** (4.4 / 20)
# Output devices. Headphones get full binaural HRTF. Open speakers mix both
# channels in the air, so binaural cues are lost: "speakers" uses a widening
# virtualiser instead. Glasses (XREAL) have open speakers right at the ears,
# close enough for HRTF cues to survive, but light on bass.
DEVICES = ("headphones", "speakers", "glasses")
# Loudness trims for the speaker virtualiser per source, calibrated against
# bypass with tools/offline_render.py (replace the binaural trims).
SPEAKER_TRIM = {"stereo": 10 ** (-2.2 / 20), "surround": 10 ** (-8.0 / 20)}
GLASSES_BASS_DB = 2.0  # x20-speaker-dsp found bass boost buzzes on small drivers


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

    def __init__(self, responses, diagonal=False):
        # diagonal: responses are (S, 1, L) and input s only feeds output s.
        sources, ears, length = responses.shape
        self.diagonal = diagonal
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
        if self.diagonal:
            spectrum = (self.history * self.spectra[:, :, 0, :]).sum(axis=0)
        else:
            # Plain multiply-and-sum is ~5x faster than einsum for these shapes.
            spectrum = (self.history[:, :, None, :] * self.spectra).sum(axis=(0, 1))
        return np.fft.irfft(spectrum, n=2 * HOP, axis=-1)[:, HOP:].astype(np.float32)


def load_speaker_correction(path, max_taps=16384):
    """Stereo FIR speaker correction (e.g. CachyOS' ROG Ally game.wav) as (2, 1, L).

    Resampled to RATE, cut where 99.9% of the energy is reached, faded out,
    and normalised to unity average power gain over 150 Hz - 8 kHz (pink
    weighting), so it reshapes the tone without making things louder.
    """
    from scipy.io import wavfile

    rate, data = wavfile.read(path)
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        data = np.stack((data, data), axis=1)
    data = data[:, :2].T
    if rate != RATE:
        data = dsp.resample_poly(data, RATE, rate, axis=-1)
    energy = np.cumsum(np.sum(data ** 2, axis=0))
    if energy[-1] <= 0:
        raise ValueError(f"Empty speaker correction FIR: {path}")
    length = int(min(max_taps, np.searchsorted(energy, 0.999 * energy[-1]) + 1))
    length = max(length, 256)
    data = data[:, :length].copy()
    fade = min(256, length // 4)
    data[:, -fade:] *= np.linspace(1, 0, fade)
    spectrum = np.abs(np.fft.rfft(data, n=8192, axis=-1)) ** 2
    frequencies = np.fft.rfftfreq(8192, 1 / RATE)
    band = (frequencies >= 150) & (frequencies <= 8000)
    pink = 1 / frequencies[band]
    gain = np.sum(spectrum[:, band] * pink, axis=-1) / np.sum(pink)
    data /= np.sqrt(np.maximum(gain, 1e-12))[:, None]
    data *= 10 ** (-2.0 / 20)  # its 150-500 Hz lift still reads ~2 dB louder (measured)
    return data[:, None, :].astype(np.float32)


class Filter:
    """Stateful IIR filter over a (channels, HOP) block."""

    def __init__(self, b, a, channels):
        self.b, self.a = b, a
        self.state = np.zeros((channels, max(len(a), len(b)) - 1))

    def process(self, block):
        out, self.state = dsp.lfilter(self.b, self.a, block, axis=-1, zi=self.state)
        return out.astype(np.float32)


class Delay:
    def __init__(self, channels, samples):
        self.buffer = np.zeros((channels, samples), dtype=np.float32)

    def process(self, block):
        joined = np.concatenate((self.buffer, block), axis=-1)
        self.buffer = joined[:, block.shape[-1]:]
        return joined[:, :block.shape[-1]]


class Dynamics:
    """Leveler -> fast compressor -> look-ahead limiter.

    Leveler and compressor follow ismailkose/x20-speaker-dsp (MIT): a slow
    loudness leveler (-20 LUFS target, +-12 dB, 10 s up / 8 s down, holds
    below -50 LUFS) and a fast stereo-linked RMS compressor (-14 dB, 3:1,
    6 dB soft knee, 3 ms attack, 120 ms release). The limiter looks one
    block (2.9 ms) ahead, so peaks are met by a gain already ramped down.
    """

    TARGET_LUFS = -20.0
    RANGE_DB = 12.0
    SILENCE_LUFS = -50.0
    THRESHOLD_DB = -14.0
    RATIO = 3.0
    KNEE_DB = 6.0
    CEILING = 0.89

    def __init__(self):
        # Rough K-weighting: high-pass at 100 Hz plus a +4 dB shelf above 2 kHz.
        self.k_high = Filter(*dsp.butter(2, 100 / (RATE / 2), "high"), 2)
        self.k_shelf = Filter(*dsp.butter(1, 2000 / (RATE / 2), "high"), 2)
        self.mean_square = 10 ** (self.TARGET_LUFS / 10)
        self.momentary = np.exp(-HOP / (RATE * 0.4))
        self.grow = np.exp(-HOP / (RATE * 10.0))
        self.fall = np.exp(-HOP / (RATE * 8.0))
        self.relax = np.exp(-HOP / (RATE * 0.5))
        self.level_gain_db = 0.0
        self.envelope_db = -90.0
        self.attack = np.exp(-HOP / (RATE * 0.003))
        self.release = np.exp(-HOP / (RATE * 0.120))
        self.lookahead = Delay(2, HOP)
        self.pending = 1.0    # dynamics gain computed for the block in the delay line
        self.gain = 1.0       # gain applied at the end of the last output block

    def _leveler(self, block, enabled):
        if not enabled:
            self.level_gain_db *= self.relax
            return
        weighted = self.k_high.process(block)
        weighted = weighted + 0.585 * self.k_shelf.process(weighted)  # ~+4 dB highs
        # ITU-R BS.1770 sums per-channel mean-square power for stereo LUFS.
        power = float(np.sum(np.mean(weighted ** 2, axis=-1)))
        self.mean_square = self.momentary * self.mean_square + (1 - self.momentary) * power
        loudness = -0.691 + 10 * np.log10(self.mean_square + 1e-12)
        if loudness <= self.SILENCE_LUFS:
            return  # hold: never pump up silence or room tone
        wanted = np.clip(self.TARGET_LUFS - loudness, -self.RANGE_DB, self.RANGE_DB)
        coefficient = self.grow if wanted > self.level_gain_db else self.fall
        self.level_gain_db = coefficient * self.level_gain_db + (1 - coefficient) * wanted

    def _compressor(self, block, gain_db):
        level = 10 * np.log10(np.mean(block ** 2) + 1e-12) + gain_db
        coefficient = self.attack if level > self.envelope_db else self.release
        self.envelope_db = coefficient * self.envelope_db + (1 - coefficient) * level
        over = self.envelope_db - self.THRESHOLD_DB
        half = self.KNEE_DB / 2
        slope = 1 - 1 / self.RATIO
        if over <= -half:
            return 0.0
        if over < half:
            return slope * (over + half) ** 2 / (2 * self.KNEE_DB)
        return slope * over

    def process(self, block, leveler=False, compress=False):
        self._leveler(block, leveler)
        gain_db = self.level_gain_db
        if compress:
            gain_db -= self._compressor(block, gain_db)
        incoming = 10 ** (gain_db / 20)
        delayed = self.lookahead.process(block)
        # Gain for the delayed block: its own dynamics gain, safe for its own
        # peak, and already safe for the next block's peak by the block's end.
        own_peak = float(np.max(np.abs(delayed))) + 1e-12
        next_peak = float(np.max(np.abs(block))) + 1e-12
        end = min(self.pending, self.CEILING / own_peak, self.CEILING / next_peak)
        if end > self.gain:
            end = self.gain + (end - self.gain) * 0.05  # ~200 ms recovery
        ramp = np.linspace(self.gain, end, HOP, dtype=np.float32)
        self.gain = end
        self.pending = incoming
        out = delayed * ramp
        # Last-resort soft clip, only touches what the limiter could not catch.
        magnitude = np.abs(out)
        knee = 0.95
        return np.where(
            magnitude > knee,
            np.sign(out) * (knee + 0.05 * np.tanh((magnitude - knee) / 0.05)),
            out,
        ).astype(np.float32)


class CrosstalkWidener:
    """Speaker crosstalk-cancelling widener, after ismailkose/x20-speaker-dsp.

    Only the side signal (L-R)/2, band-limited to 250 Hz - 6 kHz, is fed
    back with one and two 75 us delays (gains 0.55 and 0.55^2): the first
    two terms of the recursive canceller for speakers ~10 degrees apart.
    The centre (dialogue) is untouched.
    """

    DELAY = 75e-6
    GAIN = 0.55

    def __init__(self):
        high_b, high_a = dsp.butter(2, 250 / (RATE / 2), "high")
        low_b, low_a = dsp.butter(2, 6000 / (RATE / 2))
        b, a = np.polymul(high_b, low_b), np.polymul(high_a, low_a)
        # Fractional delays as linear-interpolation taps, combined into one FIR.
        taps = np.zeros(16)
        for order, weight in ((1, self.GAIN), (2, self.GAIN ** 2)):
            position = order * self.DELAY * RATE
            whole = int(position)
            fraction = position - whole
            taps[whole] += weight * (1 - fraction)
            taps[whole + 1] += weight * fraction
        self.filter = Filter(np.convolve(b, taps), a, 1)

    def process(self, stereo):
        side = 0.5 * (stereo[0] - stereo[1])
        extra = self.filter.process(side[None])[0]
        return np.stack((stereo[0] + extra, stereo[1] - extra)).astype(np.float32)


class Renderer:
    def __init__(self, hrir_path, make_vocals=None, profile="game", neural=True, reverb=True):
        hrirs, self.positions = load_hrirs(hrir_path)
        self.direct = Convolver(hrirs)
        self.rooms = {name: room_response(values["rt60"], seed=7) for name, values in PROFILES.items()}
        self.room = Convolver(self.rooms[profile])
        self._make_vocals = make_vocals
        self.model = None
        self.model_latency = 1
        self.model_delay = Delay(8, self.model_latency * HOP)  # channels kept in step with the model
        self.model_silence_blocks = self.model_latency + 2
        self._last_dub_tick = -1
        self.voice_power = 0.0               # running statistics for dub-fix
        self.voice_xcorr = np.zeros(7)
        self.voice_decay = np.exp(-HOP / (RATE * 0.25))
        self.tick = 0
        self.ambience_delay = Delay(2, int(AMBIENCE_DELAY * RATE))
        self.dynamics = Dynamics()
        lfe_b, lfe_a = dsp.butter(2, 120 / (RATE / 2))
        self.lfe_filter = (lfe_b, lfe_a, np.zeros(2))
        self.surround_seen = -1e9
        self.clock = 0.0
        self.bypass = False
        self.source = "stereo"
        self.neural = neural  # StemgenRT for stereo; otherwise a passive matrix
        self.reverb = reverb  # synthetic room reverb (can be disabled for dry sound)
        # Speaker virtualiser state.
        self.device = "headphones"
        self.correction = None             # Convolver with the speaker FIR, or None
        self.correction_path = None
        self.correction_enabled = False
        self.widener = CrosstalkWidener()
        self.speaker_delay = Delay(1, int(0.012 * RATE))
        self.speaker_lfe = Filter(*dsp.butter(2, 120 / (RATE / 2)), 1)
        self.glasses_bass = Filter(*dsp.butter(2, 220 / (RATE / 2)), 2)
        self.cinema_air = Filter(*dsp.butter(2, 4500 / (RATE / 2), "high"), 2)
        self.set_profile(profile)

    def set_device(self, kind, correction_path=None):
        """kind: headphones | speakers | glasses. correction_path: speaker FIR."""
        if kind not in DEVICES:
            raise ValueError(kind)
        self.device = kind
        self.correction_enabled = False
        if kind == "speakers" and correction_path:
            if self.correction is None or self.correction_path != correction_path:
                self.correction = Convolver(load_speaker_correction(correction_path), diagonal=True)
                self.correction_path = correction_path
            self.correction_enabled = True

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

    def _ensure_model(self):
        if self.model is None and self._make_vocals is not None:
            self.model = self._make_vocals()
            self.model_latency = getattr(self.model, "latency", 1)
            self.model_delay = Delay(8, self.model_latency * HOP)
            self.model_silence_blocks = self.model_latency + 2
        return self.model is not None

    def _model(self, audio):
        """(dry, rest) for a stereo pair, both delayed by the model latency.

        Drains the pipeline tail for `model_latency + 2` blocks when audio goes
        silent, then pauses inference to save CPU and battery. On resume,
        PipelinedVocals sees the tick gap and resets cleanly from zero.
        """
        if np.sqrt(np.mean(audio ** 2)) <= SILENCE:
            self.model_silence_blocks += 1
            if self.model_silence_blocks > self.model_latency + 2:
                zeros = np.zeros((2, HOP), dtype=np.float32)
                return zeros, zeros
        else:
            self.model_silence_blocks = 0
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        return self.model.process(audio, self.tick)

    def _surround_feeds(self, block):
        if self.neural and self.settings["dub_fix"] and self._ensure_model():
            return self._dub_fix_feeds(block)
        feeds = block[[FL, FR, FC, RL, RR, SL, SR]].copy()
        feeds[2] *= self.settings["dialogue"]
        sides = np.sqrt(np.mean(block[[SL, SR]] ** 2))
        if sides <= SILENCE:
            # 5.1: its surround pair sits at ~110 degrees, not behind.
            feeds[5:7], feeds[3:5] = feeds[3:5], 0.0
        return feeds, block[LFE]

    def _dub_fix_feeds(self, block):
        """5.1/7.1 with voice gathered from every channel into the centre.

        A careless dub copies the same voice track into several channels. One
        StemgenRT pass over a fold-down of all channels gives a reference voice
        v (2 x HOP); each channel's share of it is tracked with a running
        least-squares fit (~0.25 s memory), subtracted there, and the voice is
        re-placed in the centre alone. One model pass instead of one per pair.
        """
        fold = np.stack((
            block[FL] + 0.707 * block[FC] + 0.5 * (block[SL] + block[RL]),
            block[FR] + 0.707 * block[FC] + 0.5 * (block[SR] + block[RR]),
        ))
        dry, rest = self._model(fold)
        # Dub voice is effectively mono: one reference keeps the fit well posed.
        voice = (dry - rest).mean(axis=0).astype(np.float64)
        if self._last_dub_tick != self.tick - 1:
            self.model_delay = Delay(8, self.model_latency * HOP)
        self._last_dub_tick = self.tick
        late = self.model_delay.process(block)  # aligned with the model output
        channels = late[[FL, FR, FC, RL, RR, SL, SR]].astype(np.float64)
        decay = self.voice_decay
        self.voice_power = decay * self.voice_power + voice @ voice
        self.voice_xcorr = decay * self.voice_xcorr + channels @ voice
        # Never an inverted voice; the small floor avoids dividing by silence.
        # The network's voice is slightly coloured, so a full-strength fit
        # over-subtracts (measured -0.13); 0.85 leaves a trace instead of an
        # inverted, phasey echo.
        weights = 0.85 * np.maximum(self.voice_xcorr, 0.0) / (self.voice_power + 1e-9)
        parts = np.outer(weights, voice)          # voice found in each channel
        clean = (channels - parts).astype(np.float32)
        # Energy-conserving gather: preserves the voice's original RMS power
        # across all channels instead of over-summing smeared dubs into FC.
        gathered = min(float(np.linalg.norm(weights)), 1.25) * voice
        clean[2] += (gathered * self.settings["dialogue"]).astype(np.float32)
        # Gently lift the clean (voice-free) surrounds for richer immersion.
        clean[3:7] *= 1.18
        if np.sqrt(np.mean(late[[SL, SR]] ** 2)) <= SILENCE:
            # 5.1: its surround pair sits at ~110 degrees, not behind.
            clean[5:7], clean[3:5] = clean[3:5], 0.0
        return clean, late[LFE]

    def _stereo_feeds(self, block):
        dry = block[[FL, FR]]
        lfe = None
        if self.neural and self.settings.get("stereo_model", True) and self._ensure_model():
            dry, sep_rest = self._model(dry)
            # Move 75% of the separated voice to the centre and keep 25% in
            # L/R so dialogue stays natural and blended instead of pushy.
            voice = (dry - sep_rest) * 0.75
            # Enrich only the voice-free stem (music & effects): subtle high-end
            # air (+2 dB > 4.5 kHz), wider surround spread, and gentle LFE sub.
            air = self.cinema_air.process(sep_rest) * 0.26
            rest = (dry - voice + air).astype(np.float32)
            amb_src = (sep_rest + air).astype(np.float32)
            side_gain, rear_gain = 0.58, 0.52
            lfe = (sep_rest.mean(axis=0) * 0.25).astype(np.float32)
        else:
            # Passive upmix: half of the shared (mid) signal becomes the centre.
            # Dialogue is usually centre-panned, so it still lands in front.
            mid = 0.5 * (dry[0] + dry[1])
            voice = np.stack((0.5 * mid, 0.5 * mid))
            rest = dry - voice
            amb_src = rest
            side_gain, rear_gain = 0.45, 0.40
        # Ambience = what is not shared by both channels of the remaining mix.
        ambience = np.stack((amb_src[0] - 0.7 * amb_src[1], amb_src[1] - 0.7 * amb_src[0]))
        rear = self.ambience_delay.process(ambience)
        feeds = np.stack((
            rest[0],
            rest[1],
            voice.mean(axis=0) * self.settings["dialogue"],
            rear[0] * rear_gain,
            rear[1] * rear_gain,
            ambience[0] * side_gain,
            ambience[1] * side_gain,
        )).astype(np.float32)
        return feeds, lfe

    # -- rendering ------------------------------------------------------

    def _render(self, feeds, lfe):
        ears = self.direct.process(feeds)
        if self.reverb:
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

    def _render_speakers(self, feeds, lfe):
        """Pseudo-surround for two open speakers.

        Centre (voice) is mono in both; surrounds get extra side signal and
        their common part arrives 12 ms late so it does not fuse with the
        front; a crosstalk-cancelling widener then pushes the side content
        beyond the chassis. A small room and the speaker correction finish.
        """
        fl, fr, centre, rl, rr, sl, sr = feeds
        voice = 0.707 * centre
        surround_left, surround_right = sl + 0.8 * rl, sr + 0.8 * rr
        surround_mid = 0.5 * (surround_left + surround_right)
        surround_side = 0.75 * (surround_left - surround_right)  # surrounds sit wider than the front
        late = self.speaker_delay.process(surround_mid[None])[0] * 0.6
        left = fl + voice + late + surround_side
        right = fr + voice + late - surround_side
        out = self.widener.process(np.stack((left, right)).astype(np.float32))
        if self.reverb:
            out += self.room.process(out * 0.5) * (self.settings["room"] * 0.5)
        if lfe is not None:
            out += self.speaker_lfe.process(lfe[None])[0] * 0.7
        if self.correction_enabled:
            out = self.correction.process(out)
        return out

    def _render_device(self, feeds, lfe):
        if self.device == "speakers":
            return self._render_speakers(feeds, lfe)
        ears = self._render(feeds, lfe)
        if self.device == "glasses":
            # Low shelf: add the low band back on top, open speakers lack bass.
            ears = ears + self.glasses_bass.process(ears) * (10 ** (GLASSES_BASS_DB / 20) - 1)
        return ears

    def _downmix(self, block):
        """What you would hear without us: stereo untouched, 5.1/7.1 folded down."""
        if self.source == "stereo":
            return block[[FL, FR]].astype(np.float32)
        centre = 0.707 * block[FC]
        left = block[FL] + centre + 0.707 * (block[SL] + block[RL])
        right = block[FR] + centre + 0.707 * (block[SR] + block[RR])
        return np.stack((left, right)).astype(np.float32) * 0.6

    def process(self, block):
        """block: float32 (8, HOP). Returns float32 (2, HOP)."""
        self.tick += 1
        self.source = self._detect(block)
        if self.bypass:
            # A/B compares the spatial part only: same dynamics on both sides.
            out = self._downmix(block)
        else:
            if self.source == "surround":
                feeds, lfe = self._surround_feeds(block)
            else:
                feeds, lfe = self._stereo_feeds(block)
            if self.device == "speakers":
                trim = SPEAKER_TRIM[self.source]
            else:
                trim = SURROUND_TRIM if self.source == "surround" else STEREO_TRIM
            out = self._render_device(feeds, lfe) * trim
        # Films everywhere, and small open speakers always, get the leveler
        # and compressor; games on headphones keep their full dynamics.
        dynamic = self.settings["compress"] or self.device != "headphones"
        return self.dynamics.process(out, leveler=dynamic, compress=dynamic)
