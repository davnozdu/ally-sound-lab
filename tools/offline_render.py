"""Render a file through the live chain block by block, for listening tests.

usage: offline_render.py INPUT OUTPUT.wav [--profile game|movie] [--bypass]
                         [--seconds N] [--no-model]

The input is decoded by ffmpeg into the plugin's 7.1 layout, so a stereo file
takes the stereo/StemgenRT path and a 5.1/7.1 file the discrete path.
"""

import argparse
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from audio_worker import MODELS, Vocals  # noqa: E402
from spatial import HOP, PROFILES, RATE, Renderer  # noqa: E402


def decode(path, seconds):
    command = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-vn"]
    if seconds:
        command += ["-t", str(seconds)]
    probe = subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=channels", "-of", "csv=p=0", str(path)], text=True,
    ).strip()
    channels = int(probe.splitlines()[0])
    if channels <= 2:
        # Keep stereo in FL/FR only, as PipeWire does (no upmix).
        command += ["-af", "pan=7.1|FL=c0|FR=c1" if channels == 2 else "pan=7.1|FL=c0|FR=c0"]
    else:
        command += ["-af", "aresample=matrix_encoding=none", "-ac", "8", "-channel_layout", "7.1"]
    command += ["-ar", str(RATE), "-f", "f32le", "pipe:1"]
    audio = np.frombuffer(subprocess.check_output(command), dtype="<f4")
    return audio.reshape(-1, 8).T.copy(), channels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--profile", choices=tuple(PROFILES), default="movie")
    parser.add_argument("--bypass", action="store_true")
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--no-model", action="store_true")
    args = parser.parse_args()

    audio, channels = decode(args.source, args.seconds)
    renderer = Renderer(MODELS / "ku100.sofa", make_vocals=Vocals, profile=args.profile, neural=not args.no_model)
    renderer.bypass = args.bypass
    blocks = audio.shape[1] // HOP
    out = np.zeros((2, blocks * HOP), dtype=np.float32)
    times = []
    sources = {}
    for index in range(blocks):
        block = np.ascontiguousarray(audio[:, index * HOP:(index + 1) * HOP])
        tick = perf_counter()
        out[:, index * HOP:(index + 1) * HOP] = renderer.process(block)
        times.append((perf_counter() - tick) * 1000)
        sources[renderer.source] = sources.get(renderer.source, 0) + 1
    wavfile.write(args.output, RATE, np.rint(out.T * 32767).astype(np.int16))

    times = np.array(times[50:])  # skip warm-up
    budget = HOP / RATE * 1000
    rms_db = 20 * np.log10(np.sqrt(np.mean(out ** 2)) + 1e-12)
    print(
        f"{args.source.name}: {channels}ch -> {args.output.name} profile={args.profile} "
        f"bypass={args.bypass} source={sources}\n"
        f"  block avg {times.mean():.3f} ms, p99 {np.percentile(times, 99):.3f} ms, "
        f"max {times.max():.2f} ms, budget {budget:.2f} ms, over {int((times > budget).sum())}/{len(times)}\n"
        f"  output RMS {rms_db:.1f} dBFS, peak {np.max(np.abs(out)):.3f}"
    )
    print("  speakers:", [tuple(map(float, p)) for p in renderer.positions])


if __name__ == "__main__":
    main()
