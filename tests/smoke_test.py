#!/usr/bin/env python3
"""smoke_test.py — End-to-end check on synthetic audio (no user songs needed).

    python3 tests/smoke_test.py

Cases:
  1. 4/4 drum loop, 120 BPM   → stabilizer passes, 4/4, ~120 BPM, 4-bar lead-in
  2. 3/4 drum loop, 150 BPM   → stabilizer passes, 3/4
  3. Pink noise (no beat)     → exit 2 with EARLY_STOP
  4. Full pipeline on case 1  → exit 0, stabilised m4a + 4 stem m4a files
Exits non-zero on the first failed assertion.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SR = 44100


def drum_loop(bpm: float, bpb: int, seconds: float) -> np.ndarray:
    rng = np.random.default_rng(0)
    n = int(seconds * SR)
    y = np.zeros(n)
    beat = 60.0 / bpm
    t_kick = np.arange(int(0.15 * SR)) / SR
    kick = np.sin(2 * np.pi * (50 + 100 * np.exp(-t_kick * 40)) * t_kick) * np.exp(-t_kick * 18)
    snare = rng.normal(0, 1, int(0.12 * SR)) * np.exp(-np.arange(int(0.12 * SR)) / SR * 30) * 0.5
    hat = rng.normal(0, 1, int(0.03 * SR)) * np.exp(-np.arange(int(0.03 * SR)) / SR * 150) * 0.2
    t_bass = np.arange(int(beat * SR)) / SR
    k = 0
    while k * beat < seconds - 1:
        i = int(k * beat * SR)
        pos = k % bpb
        if pos == 0:
            seg = kick * 1.0 + np.pad(np.sin(2 * np.pi * 55 * t_bass) * 0.3 * np.exp(-t_bass * 3),
                                      (0, max(0, len(kick) - len(t_bass))))[:len(kick)]
            y[i:i + len(seg)] += seg[:n - i]
            # chord stab on beat 1 so downbeats carry more energy
            stab = np.sin(2 * np.pi * 220 * t_bass) + np.sin(2 * np.pi * 277 * t_bass)
            stab = stab * 0.15 * np.exp(-t_bass * 4)
            y[i:i + len(stab)] += stab[:n - i]
        elif (bpb == 4 and pos in (1, 3)) or (bpb == 3 and pos in (1, 2)):
            y[i:i + len(snare)] += snare[:n - i]
        j = int((k + 0.5) * beat * SR)
        if j < n:
            y[j:j + len(hat)] += hat[:n - j]
        y[i:i + len(hat)] += hat[:n - i]
        k += 1
    y /= np.abs(y).max() * 1.1
    return np.stack([y, y], axis=1)


def pink_noise(seconds: float) -> np.ndarray:
    rng = np.random.default_rng(1)
    w = rng.normal(0, 1, int(seconds * SR))
    f = np.fft.rfft(w)
    f /= np.sqrt(np.arange(1, len(f) + 1))
    y = np.fft.irfft(f, len(w))
    y /= np.abs(y).max() * 2
    return np.stack([y, y], axis=1)


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)


def check(cond: bool, msg: str) -> None:
    print(("  ✓ " if cond else "  ✗ ") + msg)
    if not cond:
        sys.exit(1)


def stabilize(d: str, name: str, audio: np.ndarray) -> tuple[int, dict, str]:
    src = os.path.join(d, f"{name}.wav")
    sf.write(src, audio, SR)
    out_json = os.path.join(d, f"{name}.json")
    r = run([sys.executable, "beat_stabilizer.py", "-i", src, "-o", os.path.join(d, f"{name}_out.wav"),
             "--analysis-out", out_json])
    a = json.load(open(out_json)) if os.path.exists(out_json) else {}
    return r.returncode, a, r.stdout + r.stderr


def main() -> None:
    with tempfile.TemporaryDirectory() as d:
        print("1. 4/4 drum loop, 120 BPM")
        rc, a, log = stabilize(d, "four", drum_loop(120, 4, 40))
        check(rc == 0, f"exit 0 (got {rc})" + ("" if rc == 0 else "\n" + log[-1500:]))
        check(a.get("meter") == "4/4", f"meter 4/4 (got {a.get('meter')})")
        check(abs(float(a.get("bpm", 0)) - 120) <= 1, f"BPM ≈ 120 (got {a.get('bpm')})")
        check(a.get("lead_in_bars") == 4, f"4-bar lead-in (got {a.get('lead_in_bars')})")

        print("2. 3/4 drum loop, 150 BPM")
        rc, a, log = stabilize(d, "three", drum_loop(150, 3, 40))
        check(rc == 0, f"exit 0 (got {rc})" + ("" if rc == 0 else "\n" + log[-1500:]))
        check(a.get("meter") == "3/4", f"meter 3/4 (got {a.get('meter')})")

        print("3. Pink noise (no beat)")
        rc, a, log = stabilize(d, "noise", pink_noise(30))
        check(rc == 2, f"exit 2 (got {rc})")
        check("EARLY_STOP" in log, "EARLY_STOP line printed")
        check(a.get("ok") is False, f"analysis ok=false, failed_check={a.get('failed_check')}")

        print("4. Full pipeline on the 4/4 loop")
        out = os.path.join(d, "pipe")
        r = run([sys.executable, "pipeline.py", "-i", os.path.join(d, "four.wav"), "--output-dir", out])
        check(r.returncode == 0, f"exit 0 (got {r.returncode})" + ("" if r.returncode == 0 else "\n" + (r.stdout + r.stderr)[-1500:]))
        files = {os.path.relpath(os.path.join(p, f), out) for p, _, fs in os.walk(out) for f in fs}
        check("four/four_stabilised.m4a" in files, "stabilised m4a written")
        stems = {f for f in files if f.startswith("four/four_stems/") and f.endswith(".m4a")}
        check(len(stems) == 4, f"4 stem m4a files (got {sorted(stems)})")
        check(not any("_work" in f for f in files), "_work removed")
    print("All smoke tests passed.")


if __name__ == "__main__":
    main()
