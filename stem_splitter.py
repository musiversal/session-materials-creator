#!/usr/bin/env python3
"""
stem_splitter.py  —  Split an audio file into stems using Demucs.

6-stem model:  htdemucs_6s  →  6 stems: vocals, drums, bass, guitar, piano, other
Default model: htdemucs     →  4 stems: vocals, drums, bass, other

Run with the demucs venv:
    ./venv_demucs/bin/python stem_splitter.py -i song.wav
    ./venv_demucs/bin/python stem_splitter.py -i song.wav --model htdemucs
    ./venv_demucs/bin/python stem_splitter.py -i song.wav --stems vocals,drums

On first run, Demucs downloads the model weights (~80–320 MB, cached afterwards).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Progress reporting (enabled by --progress-json; no-op otherwise)
# ---------------------------------------------------------------------------

_PROGRESS_JSON = False


def _emit(sub: str, pct: float, msg: str | None = None) -> None:
    """Emit a single PROGRESS JSON line on stdout (local 0.0–1.0)."""
    if not _PROGRESS_JSON:
        return
    payload = {"sub": sub, "pct": float(pct)}
    if msg:
        payload["msg"] = msg
    sys.stdout.write(f"PROGRESS {json.dumps(payload)}\n")
    sys.stdout.flush()


# Demucs prints tqdm progress on stderr like:
#   " 23%|██▎       | 23/100 [00:..]"
# Capture the percentage at the start of each line.
_TQDM_PCT = re.compile(r"(\d+)%\|")


MODELS = {
    "htdemucs_6s": "6 stems (vocals, drums, bass, guitar, piano, other) — best coverage",
    "htdemucs":    "4 stems (vocals, drums, bass, other) — faster",
    "htdemucs_ft": "4 stems, fine-tuned — best quality on some material",
    "mdx_extra":   "4 stems, alternative architecture",
}


# ---------------------------------------------------------------------------
# Stem presence detection
# ---------------------------------------------------------------------------
#
# Demucs always returns every stem its model can output, including ones the
# song doesn't actually contain (e.g. "piano" on a track with no piano). The
# unwanted ones come back near-silent or as bleed. We detect this so the UI
# can label them honestly instead of offering empty downloads as if they were
# real takes.
#
# A stem is "present" iff there's at least one 2-second window anywhere in
# the file with sustained energy above PRESENCE_DB. Short bursts of bleed
# at the start or end don't qualify; an actual played part will easily clear
# the threshold over multiple consecutive 1-second windows.

PRESENCE_DB = -30.0   # dBFS threshold for a "loud" 1-second window
PRESENCE_WIN_S = 1.0  # window length in seconds (RMS measured per window)
PRESENCE_RUN_S = 2.0  # minimum consecutive seconds above threshold


def detect_stem_presence(
    wav_path: str,
    presence_db: float = PRESENCE_DB,
    window_s: float = PRESENCE_WIN_S,
    run_s: float = PRESENCE_RUN_S,
) -> dict:
    """
    Return {"present": bool, "rms_dbfs_peak": float, "loud_seconds": float}
    for one stem WAV. `present` is True if at least `run_s` seconds of
    consecutive `window_s`-length windows clear `presence_db` dBFS.
    """
    import numpy as np

    # Try torchaudio first (always present in venv_demucs), fall back to
    # soundfile if someone runs this in a different env. Audio is normalised
    # to float32 in [-1, 1] either way.
    y = None
    sr = 0
    try:
        import torchaudio
        wav, sr_torch = torchaudio.load(wav_path)
        y = wav.mean(dim=0).numpy().astype(np.float32)  # mono mix
        sr = int(sr_torch)
    except Exception:
        try:
            import soundfile as sf
            data, sr_sf = sf.read(wav_path, always_2d=False)
            if data.ndim == 2:
                data = data.mean(axis=1)
            y = data.astype(np.float32)
            sr = int(sr_sf)
        except Exception as e:
            return {"present": True, "rms_dbfs_peak": 0.0, "loud_seconds": 0.0,
                    "error": f"could not load wav: {e}"}

    win = max(1, int(round(window_s * sr)))
    n_full = (len(y) // win) * win
    if n_full == 0:
        return {"present": False, "rms_dbfs_peak": -120.0, "loud_seconds": 0.0}

    frames = y[:n_full].reshape(-1, win)
    # RMS per window, in dBFS (full scale = ±1.0)
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    rms_db = 20.0 * np.log10(rms)
    peak_db = float(rms_db.max())

    above = rms_db >= presence_db
    # Longest run of consecutive "above" windows
    longest = 0
    current = 0
    for flag in above.tolist():
        if flag:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    loud_seconds = float(longest * window_s)
    present = loud_seconds >= run_s
    return {
        "present":      bool(present),
        "rms_dbfs_peak": round(peak_db, 2),
        "loud_seconds": round(loud_seconds, 2),
    }


def split(
    input_path: str,
    output_dir: str,
    model: str,
    wanted_stems: set[str] | None,
    demucs_shifts: int = 0,
    demucs_overlap: float = 0.0,
    demucs_jobs: int = 0,
    demucs_segment: int = 0,
    demucs_device: str = "auto",
    demucs_int24: bool = False,
    demucs_mp3: bool = False,
    presence_db: float = PRESENCE_DB,
    presence_window_s: float = PRESENCE_WIN_S,
    presence_run_s: float = PRESENCE_RUN_S,
) -> list[str]:
    """
    Run Demucs separation and return list of output file paths.
    Files are moved out of Demucs's nested folder structure into output_dir directly.
    """
    os.makedirs(output_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        cmd = [
            sys.executable, "-m", "demucs",
            "-n", model,
            "-o", tmp,
        ]
        # Demucs CLI pass-through. Only push non-default values so the command
        # line stays readable in the logs.
        if demucs_shifts is not None and demucs_shifts != 1:
            cmd += ["--shifts", str(int(demucs_shifts))]
        if demucs_overlap is not None and demucs_overlap != 0.25:
            cmd += ["--overlap", str(float(demucs_overlap))]
        if demucs_jobs and demucs_jobs > 0:
            cmd += ["--jobs", str(int(demucs_jobs))]
        if demucs_segment and demucs_segment > 0:
            cmd += ["--segment", str(int(demucs_segment))]
        # Demucs's own "auto" only picks CUDA, else CPU — it never uses Apple's
        # GPU. Resolve auto here: cuda > mps > cpu (MPS is ~3.6x faster than
        # CPU on an M-series Mac: 13.5 s vs 48.7 s for a 2:07 song).
        if not demucs_device or demucs_device == "auto":
            import torch
            demucs_device = ("cuda" if torch.cuda.is_available()
                             else "mps" if torch.backends.mps.is_available() else "cpu")
        cmd += ["-d", demucs_device]
        if demucs_int24:
            cmd += ["--int24"]
        if demucs_mp3:
            cmd += ["--mp3"]
        cmd += [input_path]
        print(f"  Model : {model}")
        print(f"  Loading model … (first run downloads weights, subsequent runs are instant)")
        _emit("demucs", 0.02, "starting demucs")

        # Stream demucs' tqdm output on stderr, parse percentages, re-emit
        # PROGRESS events. Pass stdout through unchanged.
        if _PROGRESS_JSON:
            proc = subprocess.Popen(
                cmd,
                stdout=None,            # inherit
                stderr=subprocess.PIPE,
                bufsize=0,
            )
            last_pct = 0.0
            buf = bytearray()
            assert proc.stderr is not None
            while True:
                ch = proc.stderr.read(1)
                if not ch:
                    break
                if ch in (b"\n", b"\r"):
                    line = buf.decode("utf-8", errors="replace")
                    sys.stderr.write(line + ch.decode("utf-8", errors="replace"))
                    sys.stderr.flush()
                    m = _TQDM_PCT.search(line)
                    if m:
                        pct = int(m.group(1)) / 100.0
                        # Map demucs 0..100 → 0.05..0.98 so we don't claim done before files are moved
                        scaled = 0.05 + pct * 0.93
                        if scaled > last_pct + 0.01:
                            last_pct = scaled
                            _emit("demucs", scaled)
                    buf.clear()
                else:
                    buf.append(ch[0])
            ret = proc.wait()
            if ret != 0:
                sys.exit(f"Demucs failed (exit {ret}).")
        else:
            result = subprocess.run(cmd)
            if result.returncode != 0:
                sys.exit(f"Demucs failed (exit {result.returncode}).")

        # Demucs outputs to: {tmp}/{model}/{track_name}/{stem}.wav
        track_name = Path(input_path).stem
        stems_dir = Path(tmp) / model / track_name
        if not stems_dir.exists():
            # Some models use a slightly different name — find it
            candidates = list(Path(tmp).rglob("*.wav"))
            if not candidates:
                sys.exit("Demucs produced no output files.")
            stems_dir = candidates[0].parent

        written = []
        presence: dict[str, dict] = {}
        stem_files = sorted(stems_dir.glob("*.wav")) or sorted(stems_dir.glob("*.mp3"))
        for stem_file in stem_files:
            stem_name = stem_file.stem
            if wanted_stems and stem_name.lower() not in wanted_stems:
                continue
            dest = os.path.join(output_dir, f"{stem_name}.wav")
            shutil.move(str(stem_file), dest)
            size_mb = os.path.getsize(dest) / 1_000_000
            info = detect_stem_presence(
                dest,
                presence_db=presence_db,
                window_s=presence_window_s,
                run_s=presence_run_s,
            )
            flag = "" if info["present"] else "  ⚠ low energy"
            print(f"  ✓  {stem_name:<10}  →  {dest}  ({size_mb:.1f} MB, "
                  f"peak {info['rms_dbfs_peak']:>5.1f} dBFS, "
                  f"{info['loud_seconds']:>4.1f}s loud){flag}")
            written.append(dest)
            presence[stem_name] = info

        # Write a sidecar JSON next to the stems so the pipeline / web layer
        # can read presence info without re-analysing the audio.
        info_path = os.path.join(output_dir, "stems_info.json")
        with open(info_path, "w") as f:
            json.dump({"stems": presence}, f, indent=2)

    return written


def mix_backing_track(
    stems_dir: str,
    exclude_stem: str,
    output_path: str,
    peak_dbfs: float = -1.0,
    bit_depth: int = 24,
) -> str | None:
    """
    Mix all WAV stems in stems_dir (except exclude_stem) into a single WAV.
    Reads + writes via torchaudio (always present in venv_demucs).
    Returns output_path on success, None if no mixable stems found.
    """
    import numpy as np

    mixed: np.ndarray | None = None
    sr: int = 0
    included: list[str] = []
    exclude = exclude_stem.strip().lower()

    for fname in sorted(os.listdir(stems_dir)):
        if not (fname.endswith(".wav") or fname.endswith(".mp3")):
            continue
        stem_name = os.path.splitext(fname)[0].lower()
        if stem_name == exclude:
            continue

        wav_path = os.path.join(stems_dir, fname)
        data: np.ndarray | None = None
        stem_sr: int = 0

        try:
            import torchaudio  # type: ignore
            wav_t, sr_t = torchaudio.load(wav_path)
            data = wav_t.numpy().T.astype(np.float64)   # (samples, channels)
            stem_sr = int(sr_t)
        except Exception:
            pass

        if data is None:
            try:
                import soundfile as sf  # type: ignore
                data, stem_sr = sf.read(wav_path, always_2d=True)
                data = data.astype(np.float64)
            except Exception as e:
                print(f"  ⚠  backing track: skipping {fname}: {e}", file=sys.stderr)
                continue

        if mixed is None:
            mixed = data.copy()
            sr = stem_sr
        else:
            # Align shapes: both should be (samples, 2) from Demucs stereo output
            min_ch = min(mixed.shape[1], data.shape[1])
            mixed = mixed[:, :min_ch]
            data  = data[:, :min_ch]
            # Align length
            if data.shape[0] > mixed.shape[0]:
                mixed = np.pad(mixed, ((0, data.shape[0] - mixed.shape[0]), (0, 0)))
            elif mixed.shape[0] > data.shape[0]:
                data = np.pad(data, ((0, mixed.shape[0] - data.shape[0]), (0, 0)))
            mixed = mixed + data

        included.append(stem_name)

    if mixed is None or not included:
        return None

    # Peak-normalise to the requested ceiling (default −1 dBFS) to prevent clipping.
    peak_linear = 10.0 ** (peak_dbfs / 20.0)
    peak = float(np.abs(mixed).max())
    if peak > peak_linear:
        mixed = mixed * (peak_linear / peak)

    # Save via torchaudio (always available in venv_demucs).
    # Demucs itself writes its stems this way; the TorchCodec deprecation
    # warnings are noisy but harmless.
    import torch       # type: ignore
    import torchaudio  # type: ignore
    out_tensor = torch.from_numpy(mixed.T.astype(np.float32))  # (channels, samples)
    try:
        torchaudio.save(
            output_path, out_tensor, sample_rate=sr,
            bits_per_sample=int(bit_depth), encoding="PCM_S",
        )
    except Exception:
        # Fallback: let torchaudio pick defaults (16-bit) if the codec rejects this depth
        torchaudio.save(output_path, out_tensor, sample_rate=sr)
    print(f"  ✓  Backing track  →  {output_path}")
    print(f"         Mixed      : {', '.join(included)}")
    print(f"         Excluded   : {exclude_stem}")
    _emit("backing_track", 1.0, "backing track done")
    return output_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Split an audio file into stems using Demucs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Models:
{"".join(f"  {k:<16} {v}{chr(10)}" for k, v in MODELS.items())}
Examples:
  ./venv_demucs/bin/python stem_splitter.py -i song.wav
  ./venv_demucs/bin/python stem_splitter.py -i song.wav --model htdemucs
  ./venv_demucs/bin/python stem_splitter.py -i song.wav --stems vocals,bass
        """,
    )
    p.add_argument("-i", "--input",      required=True, help="Input audio file")
    p.add_argument("-o", "--output-dir", default=None,  dest="output_dir",
                   help="Output folder (default: <input>_stems/)")
    p.add_argument("--model",  default="htdemucs", choices=list(MODELS),
                   help="Demucs model to use (default: htdemucs)")
    p.add_argument("--stems",  default=None,
                   help="Comma-separated stems to keep e.g. 'vocals,drums' (default: all)")
    p.add_argument("--progress-json", action="store_true", dest="progress_json",
                   help="Emit machine-readable PROGRESS JSON lines on stdout")
    p.add_argument("--session-type", default=None, dest="session_type",
                   help="Session instrument to exclude from the backing track (e.g. 'bass')")
    p.add_argument("--backing-track-out", default=None, dest="backing_track_out",
                   help="Output path for the backing track WAV")
    # Demucs CLI pass-through
    dem = p.add_argument_group("Demucs (library knobs)")
    dem.add_argument("--demucs-shifts",  type=int,   default=0,    dest="demucs_shifts",
                     help="Number of random shifts for equivariant stabilisation. Higher = "
                          "better separation but proportionally slower (default: 0 = fastest; demucs default 1)")
    dem.add_argument("--demucs-overlap", type=float, default=0.0, dest="demucs_overlap",
                     help="Overlap between processing chunks, 0.0–0.99 (default: 0 = fastest; demucs default 0.25)")
    dem.add_argument("--demucs-jobs",    type=int,   default=0,    dest="demucs_jobs",
                     help="Parallel worker jobs. 0 = let demucs decide (default: 0)")
    dem.add_argument("--demucs-segment", type=int,   default=0,    dest="demucs_segment",
                     help="Segment length in seconds. 0 = full file (default: 0)")
    dem.add_argument("--demucs-device",  default="auto", dest="demucs_device",
                     choices=("auto", "cpu", "cuda", "mps"),
                     help="Inference device (default: auto)")
    dem.add_argument("--demucs-int24",   action="store_true", dest="demucs_int24",
                     help="Save stems as 24-bit WAV instead of 16-bit")
    dem.add_argument("--demucs-mp3",     action="store_true", dest="demucs_mp3",
                     help="Save stems as MP3 instead of WAV")
    # Stem presence detector knobs
    pres = p.add_argument_group("Stem presence detector")
    pres.add_argument("--presence-db",        type=float, default=PRESENCE_DB,     dest="presence_db",
                      help=f"dBFS threshold for a 'loud' window (default: {PRESENCE_DB})")
    pres.add_argument("--presence-window-s",  type=float, default=PRESENCE_WIN_S,  dest="presence_window_s",
                      help=f"RMS window length in seconds (default: {PRESENCE_WIN_S})")
    pres.add_argument("--presence-run-s",     type=float, default=PRESENCE_RUN_S,  dest="presence_run_s",
                      help=f"Consecutive loud seconds needed to mark a stem 'present' "
                           f"(default: {PRESENCE_RUN_S})")
    # Backing-track mixer knobs
    bt = p.add_argument_group("Backing track")
    bt.add_argument("--backing-peak-dbfs",  type=float, default=-1.0, dest="backing_peak_dbfs",
                    help="Peak ceiling for the mixed backing track in dBFS (default: -1)")
    bt.add_argument("--backing-bit-depth",  type=int,   default=24,   dest="backing_bit_depth",
                    choices=(16, 24, 32),
                    help="Backing track WAV bit depth (default: 24)")
    return p.parse_args()


def main() -> None:
    global _PROGRESS_JSON
    args = parse_args()
    _PROGRESS_JSON = args.progress_json

    if not os.path.isfile(args.input):
        sys.exit(f"File not found: {args.input}")

    stem_root = os.path.splitext(args.input)[0]
    output_dir = args.output_dir or f"{stem_root}_stems"

    wanted = (
        {s.strip().lower() for s in args.stems.split(",")}
        if args.stems else None
    )

    print(f"\nStem splitter")
    print(f"  Input  : {args.input}")
    print(f"  Output : {output_dir}/")
    if wanted:
        print(f"  Stems  : {', '.join(sorted(wanted))}")
    print()

    files = split(
        args.input, output_dir, args.model, wanted,
        demucs_shifts=args.demucs_shifts,
        demucs_overlap=args.demucs_overlap,
        demucs_jobs=args.demucs_jobs,
        demucs_segment=args.demucs_segment,
        demucs_device=args.demucs_device,
        demucs_int24=args.demucs_int24,
        demucs_mp3=args.demucs_mp3,
        presence_db=args.presence_db,
        presence_window_s=args.presence_window_s,
        presence_run_s=args.presence_run_s,
    )
    _emit("demucs", 1.0, "done")

    if args.session_type and args.backing_track_out:
        _emit("backing_track", 0.99, "mixing backing track")
        mix_backing_track(
            output_dir, args.session_type, args.backing_track_out,
            peak_dbfs=args.backing_peak_dbfs,
            bit_depth=args.backing_bit_depth,
        )

    print(f"\nDone — {len(files)} stem(s) written to {output_dir}/")


if __name__ == "__main__":
    main()
