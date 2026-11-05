#!/usr/bin/env python3
"""
quick_analyze.py — Fast metadata detection for the upload preview.

Goal: pre-fill the web UI's Song info panel within a few seconds of the user
dropping a file, so they can confirm or override before the slow full
pipeline runs. We deliberately keep this lightweight — no madmom, no
downbeat tracker. Just librosa for beat tracking + autocorrelation
for meter.

Output: one JSON line on stdout:

    {
      "bpm":              92,                 # int rounded; null on failure
      "time_sig":         "4/4",              # e.g. "4/4", "3/4", "6/8"; null
      "duration_seconds": 187.4,
    }

Runs in the system Python (the one with the beat-stabilizer deps), since
both detectors only need numpy / librosa / soundfile.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

# Suppress librosa's deprecation chatter so the JSON line stays the only
# thing on stdout.
warnings.filterwarnings("ignore")

import tempfile

import numpy as np


def load_audio_mono(path: str, sr: int = 44100) -> tuple[np.ndarray, int]:
    """Load audio as mono float32, converting formats via pydub if needed."""
    import soundfile as sf
    import librosa

    ext = os.path.splitext(path)[1].lower()
    if ext in {".mp3", ".m4a", ".aiff", ".aif"}:
        try:
            from pydub import AudioSegment
        except ImportError:
            sys.exit("pydub required for mp3/m4a/aiff: pip install pydub")
        seg = AudioSegment.from_file(path)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            seg.export(tmp_path, format="wav")
            y, sr_orig = sf.read(tmp_path, always_2d=False)
        finally:
            os.unlink(tmp_path)
    else:
        y, sr_orig = sf.read(path, always_2d=False)

    # Mix to mono
    if y.ndim == 2:
        y = y.mean(axis=1)

    if sr_orig != sr:
        # Use numpy linear interpolation to avoid scipy.fft / resampy deadlocks
        # on macOS (scipy 1.15+ hangs on import of scipy.fft in subprocesses).
        # Linear interpolation is sufficient for beat/key/timesig detection.
        n_out = int(round(len(y) * sr / sr_orig))
        y = np.interp(
            np.linspace(0, len(y) - 1, n_out),
            np.arange(len(y)),
            y,
        ).astype(np.float32)

    return y.astype(np.float32), sr


def detect_time_signature(
    y: np.ndarray,
    sr: int,
    beat_times: np.ndarray,
    window_factor: float = 0.15,
) -> int:
    """
    Estimate beats-per-bar (2, 3, or 4) by scoring how well the onset strength
    envelope repeats at each candidate bar length via autocorrelation.

    Returns the most likely beats-per-bar integer.
    """
    import librosa

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    beat_frames = librosa.time_to_frames(beat_times, sr=sr)
    avg_beat_frames = float(np.median(np.diff(beat_frames)))

    # Normalised autocorrelation
    ac = np.correlate(onset_env, onset_env, mode="full")
    ac = ac[len(ac) // 2:]  # keep non-negative lags only
    ac /= ac[0] + 1e-8

    scores: dict[int, float] = {}
    for bpb in (2, 3, 4):
        lag = int(round(avg_beat_frames * bpb))
        if lag < len(ac):
            # Average the autocorrelation over a small window around the lag
            window = max(1, int(avg_beat_frames * window_factor))
            scores[bpb] = float(ac[max(0, lag - window): lag + window + 1].mean())

    best = max(scores, key=scores.get)

    # Tiebreak: if 4/4 and 2/4 are very close, prefer 4/4 (more common in pop)
    if best == 2 and scores.get(4, 0) > scores[2] * 0.85:
        best = 4

    # Compound-duple check: 3/4 vs 6/8.
    # In 6/8 the dotted-quarter beat (= 1.5 quarter notes) creates a strong
    # periodic accent.  The autocorrelation at that lag should therefore be
    # almost as large as the bar-period lag.  In simple 3/4 the 1.5-beat
    # position lands between quarter-note beats and is acoustically weak.
    # We encode 6/8 as the return value 6 → "6/8".
    if best == 3:
        half_lag = int(round(avg_beat_frames * 1.5))
        if half_lag < len(ac):
            window = max(1, int(avg_beat_frames * window_factor))
            half_score = float(
                ac[max(0, half_lag - window): half_lag + window + 1].mean()
            )
            if half_score >= scores[3] * 0.80:
                best = 6  # compound duple → caller renders as 6/8

    return best


def detect_bpm(y: np.ndarray, sr: int) -> tuple[float | None, np.ndarray]:
    """Return (bpm, beat_times) using librosa's default beat tracker.

    Includes an octave correction: librosa commonly locks onto 8th notes for
    half-time grooves, returning double the quarter-note tempo.  We detect
    this by checking whether the detected beats alternate between strong and
    weak onsets — the hallmark of 8th-note tracking.  If a clear alternating
    pattern is found (one set of every-other beats is ≥ 25 % stronger on
    average), we halve the BPM and keep only the stronger beat set.
    """
    import librosa

    try:
        hop = 512
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)

        tempo, beat_frames = librosa.beat.beat_track(
            onset_envelope=onset_env, sr=sr, units="frames", hop_length=hop
        )
        beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop)
        bpm_val = float(np.atleast_1d(tempo)[0]) if tempo is not None else None

        # Octave correction — only worth checking when BPM is high enough that
        # double-tracking is plausible, and we have enough beats to compare.
        if bpm_val is not None and bpm_val > 120 and len(beat_times) >= 8:
            idxs = np.clip(
                librosa.time_to_frames(beat_times, sr=sr, hop_length=hop),
                0, len(onset_env) - 1,
            ).astype(int)
            strengths  = onset_env[idxs]
            even_mean  = float(np.mean(strengths[::2]))   # beats 0, 2, 4, …
            odd_mean   = float(np.mean(strengths[1::2]))  # beats 1, 3, 5, …
            stronger   = max(even_mean, odd_mean)
            weaker     = min(even_mean, odd_mean)

            # If one set is clearly stronger (≥ 25 % gap), the tracker found
            # 8th notes — the "strong" set is the true quarter-note pulse.
            if weaker > 0 and stronger / weaker >= 1.25:
                bpm_val    = bpm_val / 2.0
                beat_times = beat_times[::2] if even_mean >= odd_mean else beat_times[1::2]

        return bpm_val, np.asarray(beat_times, dtype=float)
    except Exception:
        return None, np.zeros(0, dtype=float)


def detect_meter(y: np.ndarray, sr: int, beat_times: np.ndarray) -> str | None:
    """Return a time-signature string ('4/4', '3/4', '6/8'), or None."""
    if beat_times.size < 4:
        return None
    try:
        bpb = detect_time_signature(y, sr, beat_times)
    except Exception:
        return None
    # detect_time_signature returns 6 as a sentinel for compound duple (6/8).
    if bpb == 6:
        return "6/8"
    if bpb in (2, 3, 4, 5):
        return f"{bpb}/4"
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fast metadata detection for the upload preview.")
    p.add_argument("-i", "--input", required=True)
    p.add_argument("--sample-rate", type=int, default=22050,
                   help="Internal sample rate. Lower = faster but slightly less accurate.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.isfile(args.input):
        sys.exit(f"File not found: {args.input}")

    # load_audio_mono downsamples + mixes to mono in one pass.
    y, sr = load_audio_mono(args.input, sr=args.sample_rate)
    duration = float(len(y) / sr) if len(y) else 0.0

    bpm, beat_times = detect_bpm(y, sr)
    time_sig        = detect_meter(y, sr, beat_times)

    out = {
        "bpm":              round(bpm) if bpm and bpm > 0 else None,
        "time_sig":         time_sig,
        "duration_seconds": round(duration, 1),
    }
    sys.stdout.write(json.dumps(out) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
