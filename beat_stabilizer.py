#!/usr/bin/env python3
"""
beat_stabilizer.py - Stabilize audio to a consistent BPM grid.

Detects beats in an audio file (or accepts a manual BPM), then warps
the audio so every beat lands on an even musical grid — like Ableton's
"warp to grid" feature but from the command line.

Beat + downbeat detection uses Beat This! (CPJKU; MIT code and weights) via
beat_detect.py in venv_demucs. A detector failure is a hard error (EARLY_STOP
detector_failed); librosa only runs with --detector-backend auto/librosa.
Warping uses Rubber Band's CLI (--timemap) through pyrubberband.

After saving, writes a <output>.bpm sidecar file so downstream tools
(pipeline.py) can pick up the exact target BPM
without re-detecting it.

Half-time detection
-------------------
When --bpm is supplied, the detected beat count is compared against the
expected count (target_bpm / 60 × duration).  If the ratio is ≈ 2, the
tracker locked onto 8th notes instead of quarter notes.  All detected
beats are kept as warp anchors (2× correction density) but mapped to
the 8th-note grid, so the output plays at the correct quarter-note tempo
without stretching the file to double length.

Lead-in (on by default)
-----------------------
The output starts exactly --lead-in-bars (default 4) bars before the first
detected downbeat.  Drop the file into a DAW at bar 1: the song's first
downbeat lands on bar 5.  Audio before the first downbeat (pickups, intros) is
never trimmed: a longer intro extends the lead-in to the next whole bar, and
the rest is silence-padded.  Disable with
--no-trim-intro.

Usage:
    python3 beat_stabilizer.py -i input.mp3 -o output.wav
    python3 beat_stabilizer.py -i input.wav -o output.wav --bpm 120
    python3 beat_stabilizer.py -i input.aiff -o output.wav --strength 0.75
    python3 beat_stabilizer.py -i song.m4a --detect-only
    python3 beat_stabilizer.py -i song.wav -o out.wav --no-trim-intro
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np
import soundfile as sf
import librosa
import pyrubberband as pyrb

import quality_checks as qc


# ---------------------------------------------------------------------------
# Progress reporting (enabled by --progress-json; no-op otherwise)
# ---------------------------------------------------------------------------

_PROGRESS_JSON = False

# Side-channel for extra detector outputs (beat activations, meter fit)
# used by the quality checks. Filled by detect_beats_beatthis().
_DETECTOR_EXTRA: dict = {}


def _emit(sub: str, pct: float, msg: str | None = None) -> None:
    """Emit a single PROGRESS JSON line on stdout.

    `pct` is a local 0.0–1.0 value. pipeline.py remaps to the global range.
    """
    if not _PROGRESS_JSON:
        return
    payload = {"sub": sub, "pct": float(pct)}
    if msg:
        payload["msg"] = msg
    sys.stdout.write(f"PROGRESS {json.dumps(payload)}\n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

SUPPORTED_INPUT = {".mp3", ".wav", ".aiff", ".aif", ".m4a", ".flac", ".ogg"}


def load_audio(path: str, sr: int = 44100) -> tuple[np.ndarray, int]:
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_INPUT:
        sys.exit(f"Unsupported format: {ext}. Supported: {', '.join(SUPPORTED_INPUT)}")

    if ext in {".mp3", ".m4a", ".aiff", ".aif"}:
        try:
            from pydub import AudioSegment
        except ImportError:
            sys.exit("pydub is required for mp3/m4a/aiff: pip install pydub")
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

    if sr_orig != sr:
        print(f"  Resampling {sr_orig} Hz → {sr} Hz …")
        if y.ndim == 2:
            y = librosa.resample(y.T, orig_sr=sr_orig, target_sr=sr).T
        else:
            y = librosa.resample(y, orig_sr=sr_orig, target_sr=sr)

    return y, sr


def save_audio(path: str, y: np.ndarray, sr: int) -> str:
    """Save audio and return the actual path written."""
    ext = os.path.splitext(path)[1].lower()
    if ext in {".mp3", ".m4a"}:
        print(f"  Note: saving as WAV (lossless).")
        path = os.path.splitext(path)[0] + ".wav"
    sf.write(path, y, sr)
    print(f"  Wrote: {path}  ({len(y)/sr:.2f}s)")
    return path


def write_bpm_sidecar(audio_path: str, bpm: float) -> None:
    """Write <audio_path>.bpm so downstream tools know the exact target BPM."""
    sidecar = audio_path + ".bpm"
    with open(sidecar, "w") as f:
        f.write(f"{bpm}\n")
    print(f"  BPM sidecar: {sidecar}")


# ---------------------------------------------------------------------------
# Beat detection
# ---------------------------------------------------------------------------

def detect_beats_beatthis(
    y_mono: np.ndarray,
    sr: int,
    timeout_s: int = 300,
) -> tuple[np.ndarray, np.ndarray | None, int | None]:
    """
    Beat + downbeat detection with Beat This! (MIT code + weights), run in
    venv_demucs via beat_detect.py. Meter is one constant beats-per-bar fitted
    from downbeat probabilities (see beat_detect.py).

    Returns (beat_times, downbeat_indices, meter). Extra outputs (beat
    activations, meter fit) go to _DETECTOR_EXTRA for the quality checks.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    py = os.path.join(script_dir, "venv_demucs", "bin", "python")
    if not os.path.isfile(py):
        raise RuntimeError(f"venv_demucs not found at {py} — run bash setup.sh")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        sf.write(tmp_path, y_mono.astype(np.float32), sr)
        r = subprocess.run([py, os.path.join(script_dir, "beat_detect.py"), tmp_path],
                           capture_output=True, text=True, timeout=timeout_s)
        if r.returncode != 0:
            raise RuntimeError(r.stderr[-400:])
        payload = json.loads(r.stdout.strip().splitlines()[-1])
    finally:
        os.unlink(tmp_path)
    _DETECTOR_EXTRA.clear()
    _DETECTOR_EXTRA.update({k: payload.get(k) for k in
                            ("beat_act", "downbeat_act", "meter_fit", "meter_margin",
                             "raw_bar_counts", "phase", "device")})
    beats = np.array(payload["beats"], dtype=float)
    dbi = payload.get("downbeat_indices")
    dbi = np.array(dbi, dtype=int) if dbi else None
    return beats, dbi, payload.get("meter")


def detect_beats_librosa(
    y_mono: np.ndarray,
    sr: int,
    start_bpm: float = 120.0,
    tightness: float = 100.0,
    hop_length: int = 512,
) -> tuple[np.ndarray, float]:
    tempo, beat_frames = librosa.beat.beat_track(
        y=y_mono, sr=sr, units="frames",
        start_bpm=start_bpm, tightness=tightness, hop_length=hop_length,
    )
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)

    if len(beat_times) < 2:
        print("  [librosa] beat_track found <2 beats, switching to onset detection …")
        beat_times = librosa.onset.onset_detect(y=y_mono, sr=sr, units="time", hop_length=hop_length)
        tempo = _bpm_from_times(beat_times) if len(beat_times) >= 2 else 120.0

    return np.asarray(beat_times, dtype=float), float(np.atleast_1d(tempo)[0])


def _detector_heartbeat(stop: threading.Event, duration_s: float) -> None:
    """Emit synthetic progress ticks while the detector subprocess is blocking.

    Interpolates pct from 0.16 → 0.48 over an estimated runtime so the
    frontend bar visibly moves instead of freezing for 30–90 s.
    """
    expected_s = max(30.0, duration_s * 0.6)
    start = time.monotonic()
    while not stop.is_set():
        time.sleep(4.0)
        if stop.is_set():
            break
        elapsed = time.monotonic() - start
        frac = min(elapsed / expected_s, 0.90)
        _emit("detect_beats", 0.16 + frac * (0.48 - 0.16))


def detect_beats(
    y: np.ndarray,
    sr: int,
    backend: str = "auto",
    timeout_s: int = 300,
    librosa_start_bpm: float = 120.0,
    librosa_tightness: float = 100.0,
    librosa_hop_length: int = 512,
) -> tuple[np.ndarray, float, np.ndarray | None, int | None]:
    """
    Beat (and downbeat, when available) detection.

    `backend` is one of:
      - "auto"     : try Beat This!, fall back to librosa
      - "beatthis" : Beat This! only (errors propagate)
      - "librosa" : librosa only (no downbeat info)

    Returns (beat_times, bpm, downbeat_indices, meter).
    `downbeat_indices` and `meter` are None when the downbeat tracker
    couldn't run or the librosa backend was used.
    """
    y_mono = y.mean(axis=1) if y.ndim == 2 else y
    duration_s = (len(y) if y.ndim == 1 else y.shape[0]) / sr

    if backend == "librosa":
        beat_times, bpm = detect_beats_librosa(
            y_mono, sr,
            start_bpm=librosa_start_bpm,
            tightness=librosa_tightness,
            hop_length=librosa_hop_length,
        )
        print(f"  [librosa] {len(beat_times)} beats  |  {bpm:.2f} BPM")
        return beat_times, bpm, None, None

    stop_event = threading.Event()
    if _PROGRESS_JSON:
        t = threading.Thread(target=_detector_heartbeat, args=(stop_event, duration_s), daemon=True)
        t.start()
    else:
        t = None

    try:
        beat_times, downbeat_indices, meter = detect_beats_beatthis(
            y_mono, sr, timeout_s=timeout_s,
        )
        bpm = _bpm_from_times(beat_times)
        if downbeat_indices is not None and meter is not None:
            print(f"  [beat_this] {len(beat_times)} beats, {len(downbeat_indices)} downbeats  "
                  f"|  {bpm:.2f} BPM  |  meter ≈ {meter}/4")
        else:
            print(f"  [beat_this] {len(beat_times)} beats  |  {bpm:.2f} BPM  (downbeats unavailable)")
        return beat_times, bpm, downbeat_indices, meter
    except Exception as e:
        if backend == "beatthis":
            raise
        print(f"  [beat_this] failed ({e}), falling back to librosa …")
    finally:
        stop_event.set()
        if t is not None:
            t.join(timeout=1)

    beat_times, bpm = detect_beats_librosa(
        y_mono, sr,
        start_bpm=librosa_start_bpm,
        tightness=librosa_tightness,
        hop_length=librosa_hop_length,
    )
    print(f"  [librosa] {len(beat_times)} beats  |  {bpm:.2f} BPM")
    return beat_times, bpm, None, None


def _bpm_from_times(beat_times: np.ndarray) -> float:
    """BPM from a linear fit of beat time vs beat index.

    The median inter-beat interval is quantised by the detector's frame rate
    (Beat This! = 50 fps → 20 ms steps, e.g. 0.70 s → 85.71 BPM), so fit a
    line through all beats instead. Beats whose residual exceeds 1/4 beat
    (missed / extra beats) are dropped and the fit repeated.
    """
    if len(beat_times) < 2:
        return 120.0
    if len(beat_times) < 8:
        return float(60.0 / np.median(np.diff(beat_times)))
    t = np.asarray(beat_times, dtype=float)
    k = np.arange(len(t), dtype=float)
    keep = np.ones(len(t), dtype=bool)
    slope = float(np.median(np.diff(t)))
    for _ in range(3):
        slope, icpt = np.polyfit(k[keep], t[keep], 1)
        resid = np.abs(t - (slope * k + icpt))
        new_keep = resid < 0.25 * slope
        if new_keep.sum() < len(t) * 0.5 or (new_keep == keep).all():
            break
        keep = new_keep
    return float(60.0 / slope)


# ---------------------------------------------------------------------------
# Tempo-change detection (arrangement-level, not jitter)
# ---------------------------------------------------------------------------

def normalize_beat_octaves(
    beat_times: np.ndarray,
    fold_tol: float = 0.18,
    ema_alpha: float = 0.3,
) -> tuple[np.ndarray, dict]:
    """
    Collapse mid-song beat-tracker octave flips into a single quarter-note grid.

    Beat trackers commonly relock between quarter-note and eighth-note
    spacing within a single song (especially around section boundaries with
    different rhythmic density).  Once the lock flips, the downstream warper
    stretches the doubled-density section to 2× its original duration —
    correct for what the tracker LABELLED but musically catastrophic.

    Algorithm: walk the beat list while maintaining a running estimate `T` of
    the true quarter-note interval (initialised from the first 4 intervals).
    For each next beat:
      - gap ≈ T / 2  → 8th-note lock; drop the beat (don't add to output).
      - gap ≈ 2 T    → half-note lock; insert one midpoint, add the beat.
      - gap ≈ T      → normal; add the beat and EMA-update T.
      - anything else → add the beat without touching T (avoids letting noise
                        drift the estimate).

    Returns (normalised_beat_times, info_dict) where info_dict reports the
    number of skip / insert events and the final T estimate so the caller
    can log a structured summary.
    """
    info = {"dropped": 0, "inserted": 0, "final_T": 0.0}
    if len(beat_times) < 6:
        info["final_T"] = float(np.median(np.diff(beat_times))) if len(beat_times) >= 2 else 0.0
        return beat_times, info

    intervals = np.diff(beat_times)
    T = float(np.median(intervals[:min(4, len(intervals))]))
    if T <= 0:
        info["final_T"] = T
        return beat_times, info

    result: list[float] = [float(beat_times[0])]
    i = 1
    while i < len(beat_times):
        gap = float(beat_times[i]) - result[-1]
        ratio = gap / T

        if abs(ratio - 0.5) < 0.5 * fold_tol:
            # 8th-note lock — skip the doubled beat
            info["dropped"] += 1
            i += 1
        elif abs(ratio - 2.0) < 2.0 * fold_tol:
            # Half-note lock — insert one midpoint to restore quarter-note grid
            mid = result[-1] + T
            result.append(mid)
            result.append(float(beat_times[i]))
            info["inserted"] += 1
            # Update T from the inserted half (mid → beat[i]) so we adapt to
            # any subtle drift inside the half-locked region
            T = (1 - ema_alpha) * T + ema_alpha * (float(beat_times[i]) - mid)
            i += 1
        elif abs(ratio - 1.0) < fold_tol:
            # Normal beat — keep and EMA-update T
            result.append(float(beat_times[i]))
            T = (1 - ema_alpha) * T + ema_alpha * gap
            i += 1
        else:
            # Ambiguous (rubato, dropped/added beat, etc.) — keep, don't update T
            result.append(float(beat_times[i]))
            i += 1

    info["final_T"] = T
    return np.array(result, dtype=float), info


def _octave_fold_bpm(bpm: float, reference: float, fold_tol: float = 0.15) -> float:
    """Fold `bpm` by ×2 or ×0.5 when doing so brings it close to `reference`.

    Beat trackers regularly produce octave errors: a 100-BPM song reads as
    either 100 or 200 depending on whether the tracker locked onto quarter
    notes or eighth notes.  When the lock flips mid-song we get a "phantom
    tempo change" that's really just one tempo viewed at two metrical levels.

    Returns the folded value only when it lands within `fold_tol` (relative)
    of the reference; otherwise returns `bpm` unchanged.  This preserves true
    1.5× / 1.3× tempo changes (which should still trip the guard) while
    silently collapsing clean 2.0× / 0.5× octave flips.
    """
    if bpm <= 0 or reference <= 0:
        return bpm
    candidates = (bpm, bpm * 0.5, bpm * 2.0)
    best = min(candidates, key=lambda x: abs(x - reference))
    if best is bpm:
        return bpm
    if abs(best - reference) / reference <= fold_tol:
        return best
    return bpm


def detect_tempo_change(
    beat_times: np.ndarray,
    downbeat_indices: np.ndarray | None,
    window_bars: int = 8,
    persist_bars: int = 4,
    threshold_pct: float = 0.06,
    threshold_floor_bpm: float = 6.0,
    octave_fold_tol: float = 0.15,
) -> dict | None:
    """
    Detect a sustained arrangement-level tempo change.

    The current pipeline assumes a single global BPM and warps everything to
    it. That's fine for AI jitter or human rubato (high-frequency wobble
    around a stable mean) but wrong for songs with real section-level tempo
    shifts (e.g. ballad section at 70 BPM, chorus at 90 BPM). For the latter
    the warp would produce audibly broken output, so we'd rather stop and
    tell the user.

    Algorithm:
      1. Compute per-bar median beat interval → per-bar BPM.
         If downbeats are unavailable, treat groups of 4 beats as a bar
         (acceptable fallback — the smoothing window absorbs the error).
      2. Rolling-median over `window_bars` to flatten jitter/rubato.
      3. Scan for sustained step changes: |Δbpm| ≥ max(6, 6%) of the base
         BPM, that persists for at least `persist_bars` more windows
         without returning to the prior level.

    Returns
    -------
    None if no change is detected, else
    {"from_bpm": float, "to_bpm": float, "at_bar": int, "at_time_s": float}.
    """
    if len(beat_times) < (window_bars + persist_bars) * 4:
        # Too short to reason about — let it through.
        return None

    intervals = np.diff(beat_times)
    if intervals.size == 0:
        return None

    # Group beats into bars.
    if downbeat_indices is not None and len(downbeat_indices) >= 4:
        # Use real bar boundaries.
        bar_bpms: list[float] = []
        bar_starts_t: list[float] = []
        for i in range(len(downbeat_indices) - 1):
            lo = int(downbeat_indices[i])
            hi = int(downbeat_indices[i + 1])
            if hi <= lo + 1:
                continue
            bar_intervals = intervals[lo:hi - 1] if hi - 1 <= len(intervals) else intervals[lo:]
            if len(bar_intervals) == 0:
                continue
            median_interval = float(np.median(bar_intervals))
            bar_bpms.append(60.0 / median_interval)
            bar_starts_t.append(float(beat_times[lo]))
    else:
        # No downbeats → assume 4 beats per bar.
        beats_per_bar = 4
        n_bars = len(intervals) // beats_per_bar
        if n_bars < window_bars + persist_bars:
            return None
        bar_bpms = []
        bar_starts_t = []
        for b in range(n_bars):
            lo = b * beats_per_bar
            hi = lo + beats_per_bar
            median_interval = float(np.median(intervals[lo:hi]))
            bar_bpms.append(60.0 / median_interval)
            bar_starts_t.append(float(beat_times[lo]))

    bpms = np.array(bar_bpms, dtype=float)
    if len(bpms) < window_bars + persist_bars + 1:
        return None

    # Rolling median over window_bars (centred).
    half = window_bars // 2
    smoothed = np.empty_like(bpms)
    for i in range(len(bpms)):
        lo = max(0, i - half)
        hi = min(len(bpms), i + half + 1)
        smoothed[i] = float(np.median(bpms[lo:hi]))

    # Scan for step changes.
    # We look for: smoothed[i] differs from smoothed[i-1] by ≥ threshold,
    # AND smoothed[i+1..i+persist_bars] also stays on the new level (within
    # half the threshold of smoothed[i]).
    #
    # Before each comparison, octave-fold the candidate BPM against the
    # reference: a 105 → 207 flip becomes 105 → 103.5 (the second half is
    # the same tempo, just heard in eighth notes by the tracker).  See
    # _octave_fold_bpm for the rationale and tolerance.
    for i in range(window_bars, len(smoothed) - persist_bars):
        prev_bpm = float(smoothed[i - 1])
        curr_bpm = _octave_fold_bpm(float(smoothed[i]), prev_bpm, octave_fold_tol)
        delta = abs(curr_bpm - prev_bpm)
        threshold = max(threshold_floor_bpm, threshold_pct * prev_bpm)
        if delta < threshold:
            continue
        # Check persistence: every smoothed value in [i, i+persist_bars]
        # (also octave-folded against prev_bpm) must stay within threshold/2
        # of curr_bpm.
        persistence_window = np.array([
            _octave_fold_bpm(float(b), prev_bpm, octave_fold_tol)
            for b in smoothed[i: i + persist_bars + 1]
        ])
        if np.all(np.abs(persistence_window - curr_bpm) < threshold / 2):
            return {
                "from_bpm": round(prev_bpm, 2),
                "to_bpm":   round(curr_bpm, 2),
                "at_bar":   i + 1,  # 1-indexed bar number
                "at_time_s": round(bar_starts_t[i], 2),
            }
    return None


# ---------------------------------------------------------------------------
# Beat stabilisation (warping)
# ---------------------------------------------------------------------------

def build_timemap(beat_samples: np.ndarray, target_samples: np.ndarray, n_total: int) -> np.ndarray:
    pairs: list[tuple[int, int]] = []

    if beat_samples[0] > 0:
        pairs.append((0, 0))

    for s, t in zip(beat_samples.tolist(), target_samples.tolist()):
        pairs.append((int(s), int(t)))

    last_src, last_tgt = pairs[-1]
    pairs.append((int(n_total), int(last_tgt + (n_total - last_src))))

    clean: list[tuple[int, int]] = [pairs[0]]
    for s, t in pairs[1:]:
        if s > clean[-1][0] and t > clean[-1][1]:
            clean.append((s, t))

    return np.array(clean, dtype=np.int32)


def stabilize(
    y: np.ndarray,
    sr: int,
    beat_times: np.ndarray,
    target_bpm: float,
    strength: float = 1.0,
    crispness: int | None = None,
) -> np.ndarray:
    beat_interval_samples = sr * 60.0 / target_bpm
    beat_samples = np.round(beat_times * sr).astype(np.int64)

    first = int(beat_samples[0])
    ideal_samples = np.array(
        [first + round(i * beat_interval_samples) for i in range(len(beat_samples))],
        dtype=np.int64,
    )
    target_samples = np.round(
        (1.0 - strength) * beat_samples + strength * ideal_samples
    ).astype(np.int64)

    n_total = len(y) if y.ndim == 1 else y.shape[0]
    timemap = build_timemap(beat_samples, target_samples, n_total)

    print(f"  Warping {len(timemap)-2} beat anchors …")
    # pyrubberband exposes rubberband's --crispness flag via rbargs. 0 = smooth,
    # 6 = sharp transients. None = library default.
    rbargs: dict | None = None
    if crispness is not None:
        rbargs = {"--crispness": str(int(crispness))}
    return pyrb.timemap_stretch(y, sr, timemap, rbargs=rbargs) if rbargs else pyrb.timemap_stretch(y, sr, timemap)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stabilise audio to a consistent BPM grid.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 beat_stabilizer.py -i live_drums.wav -o stable_drums.wav
  python3 beat_stabilizer.py -i guitar.mp3 -o guitar_stable.wav --bpm 98 --strength 0.8
  python3 beat_stabilizer.py -i song.m4a --detect-only
        """,
    )
    p.add_argument("-i", "--input",       required=True)
    p.add_argument("-o", "--output",      default=None,  help="Output WAV file")
    p.add_argument("--bpm",               type=float,    default=None, help="Target BPM (auto-detected if omitted)")
    p.add_argument("--strength",          type=float,    default=1.0,  metavar="0-1", help="Quantisation strength (default 1.0)")
    p.add_argument("--sample-rate",       type=int,      default=44100)
    p.add_argument("--detect-only",       action="store_true",
                   help="Print detected BPM and exit without writing")
    # Intro trim is ON by default: the output starts one bar before the first
    # detected beat so the file drops straight into a DAW without manual offsetting.
    # Use --no-trim-intro to get the raw stabilised audio with no padding changes.
    p.add_argument("--no-trim-intro",     action="store_false", dest="trim_intro",
                   help="Disable the default intro trim (output starts at sample 0)")
    p.add_argument("--trim-intro",        action="store_true",  dest="trim_intro",
                   help="Trim output to start one bar before the first detected beat "
                        "(on by default; silence-padded when the beat is near the file start)")
    p.set_defaults(trim_intro=True)
    p.add_argument("--beats-per-bar",     type=int,      default=None, dest="beats_per_bar",
                   help="Beats per bar for the intro trim length "
                        "(default: auto-detected from downbeats, falling back to 4)")
    p.add_argument("--allow-tempo-change", action="store_true", dest="allow_tempo_change",
                   help="Proceed even if a sustained tempo change is detected. "
                        "Default: stop and emit EARLY_STOP so the caller can warn the user. "
                        "Enable only if you accept that the warp will be musically wrong "
                        "across the tempo boundary.")
    p.add_argument("--lead-in-bars", "--intro-trim-bars", type=int, default=4, dest="intro_trim_bars",
                   help="Bars of lead-in before the first downbeat when --trim-intro is on. "
                        "Pre-roll audio older than this is trimmed; missing time is "
                        "silence-padded (default: 4)")
    p.add_argument("--skip-checks",       action="store_true", dest="skip_checks",
                   help="Record quality checks but never abort on them")
    p.add_argument("--analysis-out",      default=None, dest="analysis_out",
                   help="Write detected BPM / meter / lead-in info as JSON to this path")
    # Detector backend & library-level knobs.
    det = p.add_argument_group("Beat detector (library knobs)")
    det.add_argument("--detector-backend", default="beatthis",
                     choices=("beatthis", "auto", "librosa"), dest="detector_backend",
                     help="Beat detector (default: beatthis — fail if Beat This! fails). "
                          "'auto' falls back to librosa (no downbeats); local debugging only")
    det.add_argument("--detector-timeout-s", type=int, default=300, dest="detector_timeout_s",
                     help="Seconds before the Beat This! subprocess is aborted (default: 300)")
    det.add_argument("--librosa-start-bpm", type=float, default=120.0, dest="librosa_start_bpm",
                     help="Initial tempo guess for librosa.beat.beat_track (default: 120)")
    det.add_argument("--librosa-tightness", type=float, default=100.0, dest="librosa_tightness",
                     help="librosa beat-tracker onset-strength weighting (default: 100)")
    det.add_argument("--librosa-hop-length", type=int, default=512, dest="librosa_hop_length",
                     help="librosa STFT hop length in samples (default: 512)")
    # Tempo-change guard knobs.
    tc = p.add_argument_group("Tempo-change guard")
    tc.add_argument("--tempo-change-window-bars",  type=int, default=8, dest="tempo_change_window_bars",
                    help="Rolling-median window for the tempo-change scanner (default: 8 bars)")
    tc.add_argument("--tempo-change-persist-bars", type=int, default=4, dest="tempo_change_persist_bars",
                    help="Bars the new tempo must persist before the guard fires (default: 4)")
    tc.add_argument("--tempo-change-threshold-pct", type=float, default=0.10, dest="tempo_change_threshold_pct",
                    help="Percentage tempo step that counts as a change (default: 0.10 = 10%%)")
    tc.add_argument("--tempo-change-threshold-floor", type=float, default=6.0, dest="tempo_change_threshold_floor",
                    help="Minimum absolute BPM step that counts as a change (default: 6)")
    # Beat-octave normalisation (runs after detection, before warping).
    p.add_argument("--beat-octave-normalize", action="store_true",
                   dest="beat_octave_normalize",
                   help="Legacy (off by default): thin doubled beats / insert midpoints. "
                        "Seeds its tempo from the first 4 intervals, so an odd intro "
                        "re-grids the whole song, and it discards the detector's "
                        "downbeats. Beat This! does not flip octaves mid-song.")
    p.add_argument("--no-beat-octave-normalize", action="store_false",
                   dest="beat_octave_normalize", help=argparse.SUPPRESS)
    p.set_defaults(beat_octave_normalize=False)
    # Warp engine knobs.
    p.add_argument("--pyrb-crispness", type=int, default=None, dest="pyrb_crispness",
                   help="rubberband --crispness (0=smoothest, 6=sharpest transients). "
                        "Default: library default.")
    p.add_argument("--progress-json",     action="store_true", dest="progress_json",
                   help="Emit machine-readable PROGRESS JSON lines on stdout")
    return p.parse_args()


def _write_analysis(path: str | None, data: dict) -> None:
    if not path:
        return
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Analysis JSON: {path}")


def _emit_early_stop(reason: str, **details: object) -> None:
    """Emit a structured EARLY_STOP line on stdout. Always emitted (not gated
    by --progress-json) so the caller / web layer can parse it even when
    PROGRESS lines are off."""
    payload = {"reason": reason, **details}
    sys.stdout.write(f"EARLY_STOP {json.dumps(payload)}\n")
    sys.stdout.flush()


def main() -> None:
    global _PROGRESS_JSON
    args = parse_args()
    _PROGRESS_JSON = args.progress_json

    if not os.path.isfile(args.input):
        sys.exit(f"File not found: {args.input}")
    if not args.detect_only and args.output is None:
        sys.exit("Specify -o / --output (or use --detect-only).")
    if not 0.0 <= args.strength <= 1.0:
        sys.exit("--strength must be between 0.0 and 1.0")

    _emit("load", 0.02, "loading audio")
    print(f"\n[1/4] Loading  {args.input} …")
    y, sr = load_audio(args.input, args.sample_rate)
    duration = (len(y) if y.ndim == 1 else y.shape[0]) / sr
    channels = 1 if y.ndim == 1 else y.shape[1]
    print(f"  {duration:.2f}s  |  {sr} Hz  |  {channels}ch")

    _emit("detect_beats", 0.15, "detecting beats")
    print("\n[2/4] Detecting beats …")
    try:
        beat_times, detected_bpm, downbeat_indices, detected_meter = detect_beats(
            y, sr,
            backend=args.detector_backend,
            timeout_s=args.detector_timeout_s,
            librosa_start_bpm=args.librosa_start_bpm,
            librosa_tightness=args.librosa_tightness,
            librosa_hop_length=args.librosa_hop_length,
        )
    except Exception as e:  # noqa: BLE001 — any detector crash is a hard stop
        detail = f"{type(e).__name__}: {e}"[:500]
        print(f"\n  ✗  Beat detector failed: {detail}")
        if args.analysis_out:
            _write_analysis(args.analysis_out, {"ok": False, "failed_check": "detector_failed",
                                                "detail": detail})
        _emit_early_stop("detector_failed", detail=detail)
        sys.exit(2)
    _emit("detect_beats", 0.55, f"{len(beat_times)} beats")

    # ── Quality gates (see quality_checks.py) ─────────────────────────────
    checks: dict[str, dict] = {}
    y_mono_for_checks = y.mean(axis=1) if y.ndim == 2 else y

    def _gate(name: str, result: dict) -> None:
        checks[name] = result
        mark = "✓" if result["ok"] else "✗"
        print(f"  [check] {mark} {name:<13} {result['reason'] or 'ok'}")
        if result.get("warning"):
            print(f"          ⚠ {result['warning']}")
        if not result["ok"] and not args.skip_checks and not args.detect_only:
            _write_analysis(args.analysis_out, {"ok": False, "failed_check": name,
                                                "checks": checks})
            _emit_early_stop(name, **{("detail" if k == "reason" else k): v
                                      for k, v in result.items() if k != "ok"})
            sys.exit(2)

    _gate("beat_clarity", qc.check_beat_clarity(beat_times, _DETECTOR_EXTRA.get("beat_act")
                                                if args.detector_backend != "librosa" else None))
    _gate("meter", qc.check_meter(
        detected_meter, _DETECTOR_EXTRA.get("meter_fit"), _DETECTOR_EXTRA.get("phase"),
        qc.librosa_meter(y_mono_for_checks, sr, beat_times)))

    # Beat-octave normalisation — collapses mid-song quarter↔eighth-note
    # tracker flips into a single grid so (a) the tempo-change guard sees a
    # stable BPM and (b) the warper doesn't stretch the doubled-density
    # section to 2× its true length.  Disable with --no-beat-octave-normalize.
    if args.beat_octave_normalize and len(beat_times) >= 6:
        beat_times_before = beat_times
        beat_times, norm_info = normalize_beat_octaves(beat_times)
        if norm_info["dropped"] or norm_info["inserted"]:
            print(f"  [octave-normalize] dropped {norm_info['dropped']} 8th-note beats, "
                  f"inserted {norm_info['inserted']} half-note midpoints "
                  f"({len(beat_times_before)} → {len(beat_times)} beats)")
            # If we modified the sequence, downbeat indices from the detector
            # no longer match — drop them so detect_tempo_change falls back to
            # the 4-beat-bar assumption rather than dereferencing stale indices.
            if downbeat_indices is not None:
                downbeat_indices = None

    _gate("tempo_jumps", qc.check_tempo_jumps(beat_times))

    if args.detect_only:
        print(f"\nDetected BPM   : {detected_bpm:.3f}")
        print(f"Beat count     : {len(beat_times)}")
        if detected_meter:
            print(f"Detected meter : {detected_meter}/4")
        if downbeat_indices is not None:
            print(f"Downbeats      : {len(downbeat_indices)}")
        _write_analysis(args.analysis_out, {
            "ok": all(c["ok"] for c in checks.values()),
            "detected_bpm": round(float(detected_bpm), 2),
            "meter": f"{detected_meter}/4" if detected_meter else None,
            "checks": checks,
        })
        return

    # Tempo-change guard.
    # Catches songs with arrangement-level tempo shifts (ballad → chorus
    # bump, etc.) that the single-BPM warp would corrupt. AI-floating-BPM
    # jitter and human rubato are explicitly *not* flagged — that's what
    # the warp is for. See detect_tempo_change() docstring.
    tempo_event = detect_tempo_change(
        beat_times, downbeat_indices,
        window_bars=args.tempo_change_window_bars,
        persist_bars=args.tempo_change_persist_bars,
        threshold_pct=args.tempo_change_threshold_pct,
        threshold_floor_bpm=args.tempo_change_threshold_floor,
    )
    if tempo_event is not None:
        if args.allow_tempo_change:
            print(f"\n  ⚠  Tempo change detected at bar {tempo_event['at_bar']} "
                  f"({tempo_event['from_bpm']} → {tempo_event['to_bpm']} BPM); "
                  f"continuing anyway because --allow-tempo-change was set.")
        else:
            print(f"\n  ✗  Tempo change detected at bar {tempo_event['at_bar']} "
                  f"({tempo_event['from_bpm']} → {tempo_event['to_bpm']} BPM, "
                  f"≈{tempo_event['at_time_s']}s).")
            print(f"     Beat stabilization assumes a single tempo and would warp this incorrectly.")
            print(f"     Split the song at the section boundary, or pass --allow-tempo-change.")
            _emit_early_stop("tempo_change", **tempo_event)
            sys.exit(2)

    # If the user didn't pass --beats-per-bar, use the detected meter (when
    # available) as the default for the intro trim length. Fall back to 4.
    if args.beats_per_bar is None:
        args.beats_per_bar = detected_meter if detected_meter else 4

    if args.bpm is not None:
        target_bpm = args.bpm
        print(f"\n[3/4] Using manual BPM: {target_bpm:.2f}")
        # ── Duration-fit sanity check ──────────────────────────────────────
        # Expected beats at target BPM vs actually detected.  A ratio near 2
        # means the tracker locked onto 8th notes in a half-time groove —
        # thin to every other beat so we don't stretch the file to 2× length.
        expected_beats = target_bpm / 60.0 * duration
        ratio = len(beat_times) / expected_beats
        if 1.7 < ratio < 2.3:
            # Half-time groove: the tracker locked onto 8th notes.
            # Keep ALL detected beats as warp anchors (2× correction density)
            # but map them to the 8th-note grid at target_bpm so the output
            # audio plays at the correct quarter-note tempo.
            warp_bpm = target_bpm * 2          # 8th-note grid spacing
            print(f"  ⚠  Beat count ({len(beat_times)}) ≈ 2× expected "
                  f"({expected_beats:.0f}) at {target_bpm} BPM  →  half-time groove.")
            print(f"     Using all {len(beat_times)} sub-beat anchors at "
                  f"{warp_bpm:.0f} BPM grid (2× correction density).")
        elif 0.43 < ratio < 0.57:
            warp_bpm = target_bpm
            print(f"  ⚠  Beat count ({len(beat_times)}) ≈ ½ expected "
                  f"({expected_beats:.0f}) at {target_bpm} BPM  →  double-time "
                  f"detected; consider a higher --bpm value.")
        else:
            warp_bpm = target_bpm
    else:
        if detected_bpm <= 0 or len(beat_times) < 2:
            sys.exit("Could not detect a valid BPM. Try --bpm <value>.")
        target_bpm = round(detected_bpm)
        warp_bpm   = target_bpm
        print(f"\n[3/4] Auto BPM: {detected_bpm:.2f} → rounded to {target_bpm}")

    _emit("warp", 0.60, "warping")
    print(f"\n[4/4] Stabilising (strength={args.strength}) …")
    y_out = stabilize(
        y, sr, beat_times, warp_bpm,
        strength=args.strength,
        crispness=args.pyrb_crispness,
    )
    _emit("warp", 0.90, "warped")

    n_in  = len(y) if y.ndim == 1 else y.shape[0]
    n_out_warped = len(y_out) if y_out.ndim == 1 else y_out.shape[0]
    _gate("length", qc.check_length(n_in, n_out_warped))

    # ── Lead-in ───────────────────────────────────────────────────────────────
    # Place the first downbeat exactly --lead-in-bars bars (default 4) after the
    # start of the file, padding with silence. Audio before the first downbeat
    # is never trimmed; a longer intro extends the lead-in to whole bars.
    if args.trim_intro:
        bar_duration   = args.beats_per_bar * 60.0 / target_bpm
        intro_duration = max(0, int(args.intro_trim_bars)) * bar_duration
        # Anchor on the first downbeat (beat 1 of bar 1) so the lead-in
        # lands on whole bars. Its position in the *output* is its warp target
        # (same formula as stabilize()). Falls back to the first beat when the
        # downbeat tracker is unavailable.
        anchor_idx = (int(downbeat_indices[0])
                      if downbeat_indices is not None and len(downbeat_indices) else 0)
        beat_iv    = 60.0 / warp_bpm
        ideal_t    = float(beat_times[0]) + anchor_idx * beat_iv
        first_beat_t = (1.0 - args.strength) * float(beat_times[anchor_idx]) + args.strength * ideal_t
        print(f"\n  Lead-in:")
        print(f"    First downbeat : {first_beat_t:.3f}s  (beat #{anchor_idx})")
        print(f"    Bar duration   : {bar_duration:.3f}s  "
              f"({args.beats_per_bar} beats @ {target_bpm} BPM)")
        # Never cut music: if the audio before the first downbeat is longer than
        # the requested lead-in (e.g. a beatless intro), extend the lead-in to
        # the next whole bar so the intro is kept and bar lines still line up.
        pre_roll = max(0.0, first_beat_t)
        lead_in_bars = max(int(args.intro_trim_bars),
                           int(np.ceil(pre_roll / bar_duration - 1e-6)))
        intro_duration = lead_in_bars * bar_duration
        pad_t = intro_duration - pre_roll           # >= 0 by construction
        print(f"    Pre-roll audio : {pre_roll:.3f}s")
        print(f"    Lead-in        : {intro_duration:.3f}s  ({lead_in_bars} bars"
              f"{', extended to keep the intro' if lead_in_bars > args.intro_trim_bars else ''})")
        pad_frames = round(pad_t * sr)
        if pad_frames > 0:
            shape = (pad_frames,) if y_out.ndim == 1 else (pad_frames, y_out.shape[1])
            y_out = np.concatenate([np.zeros(shape, dtype=y_out.dtype), y_out], axis=0)
        print(f"    →  prepended {pad_t:.3f}s of silence (no audio trimmed)")
        args.intro_trim_bars = lead_in_bars

    print(f"\nSaving …")
    saved_path = save_audio(args.output, y_out, sr)
    write_bpm_sidecar(saved_path, target_bpm)
    lead_in = int(args.intro_trim_bars) if args.trim_intro else 0
    n_out = len(y_out) if y_out.ndim == 1 else y_out.shape[0]
    _write_analysis(args.analysis_out, {
        "ok":               True,
        "bpm":              target_bpm,
        "detected_bpm":     round(float(detected_bpm), 2),
        "meter":            f"{detected_meter}/4" if detected_meter else None,
        "beats_per_bar":    int(args.beats_per_bar),
        "lead_in_bars":     lead_in,
        "first_downbeat_s": round(lead_in * args.beats_per_bar * 60.0 / target_bpm, 4)
                            if args.trim_intro else None,
        "duration_s":       round(n_out / sr, 3),
        "input_duration_s": round(n_in / sr, 3),
        "checks":           checks,
    })
    _emit("save", 1.0, "done")
    print("\nDone.")


if __name__ == "__main__":
    main()
