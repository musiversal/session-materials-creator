#!/usr/bin/env python3
"""
pipeline.py  —  Session materials: detect BPM + meter → beat-stabilize → stem split.

Run with system Python (python3):
    python3 pipeline.py -i song.wav
    python3 pipeline.py -i song.m4a --bpm 84
    python3 pipeline.py -i song.wav --skip-stabilize
    python3 pipeline.py -i song.wav --skip-stems
    python3 pipeline.py -i song.wav --stems vocals,drums
    python3 pipeline.py -i song.wav --session-type bass

Outputs (under <output-dir>/<input_base>/):
    <input_base>_stabilised.wav        beat-locked audio with a 4-bar lead-in
    <input_base>_stabilised.wav.bpm    BPM sidecar
    <input_base>_analysis.json         BPM, meter, lead-in, first-downbeat time
    <input_base>_stems/<stem>.wav      one WAV per Demucs stem (+ stems_info.json)
    <input_base>_backing_track.wav     only with --session-type

Lead-in is on by default: the stabilised WAV starts exactly --lead-in-bars
(default 4) bars before the first detected downbeat, so it drops into a DAW at
bar 1 with the song starting on bar 5. Disable with --no-trim-intro.

Half-time songs (e.g. 78 BPM where the detector locks onto 156 BPM 8th notes)
are handled automatically when --bpm is supplied.
"""

from __future__ import annotations

import argparse
import json
import shutil
import os
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Detect BPM/meter → beat-stabilize (with lead-in) → stem split.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 pipeline.py -i song.wav
  python3 pipeline.py -i song.m4a --bpm 84 --lead-in-bars 2
  python3 pipeline.py -i song.wav --stems vocals,bass
  python3 pipeline.py -i song.wav --stem-model htdemucs_ft
  python3 pipeline.py -i already_stable.wav --skip-stabilize
        """,
    )
    # ── Input ───────────────────────────────────────────────
    p.add_argument("-i", "--input",      required=True, help="Input audio file")
    p.add_argument("--title",            default=None,  help="Song title (informational)")
    p.add_argument("--output-dir",       default=None,  dest="output_dir",
                   help="Base directory for outputs (default: ~/Desktop/audio-tools-tests)")

    # ── Beat stabilizer ─────────────────────────────────────
    stab = p.add_argument_group("Beat stabilizer")
    stab.add_argument("--bpm",            type=float, default=None,
                      help="Target BPM (auto-detected if omitted)")
    stab.add_argument("--strength",       type=float, default=1.0, metavar="0-1",
                      help="Quantisation strength (default: 1.0)")
    stab.add_argument("--skip-stabilize", action="store_true",
                      help="Skip stabilization; split stems from the input as-is")
    stab.add_argument("--no-trim-intro",  action="store_false", dest="trim_intro",
                      help="Disable the lead-in trim/pad (output starts at sample 0)")
    stab.add_argument("--trim-intro",     action="store_true",  dest="trim_intro",
                      help="Apply the lead-in trim/pad (default)")
    p.set_defaults(trim_intro=True)
    stab.add_argument("--lead-in-bars", "--intro-trim-bars", type=int, default=4,
                      dest="lead_in_bars",
                      help="Bars of lead-in before the first downbeat (default: 4)")
    stab.add_argument("--beats-per-bar",  type=int, default=None, dest="beats_per_bar",
                      help="Override meter (beats per bar). Default: auto-detected.")
    stab.add_argument("--allow-tempo-change", action="store_true", dest="allow_tempo_change",
                      help="Proceed even if a sustained tempo change is detected")

    det = p.add_argument_group("Beat detector (library knobs)")
    det.add_argument("--detector-backend", default="beatthis",
                     choices=("beatthis", "auto", "librosa"), dest="detector_backend",
                     help="default beatthis: fail if Beat This! fails; auto/librosa for debugging only")
    det.add_argument("--detector-timeout-s", type=int,   default=300,   dest="detector_timeout_s")
    det.add_argument("--librosa-start-bpm",  type=float, default=120.0, dest="librosa_start_bpm")
    det.add_argument("--librosa-tightness",  type=float, default=100.0, dest="librosa_tightness")
    det.add_argument("--librosa-hop-length", type=int,   default=512,   dest="librosa_hop_length")

    bs = p.add_argument_group("Beat stabilizer (library knobs)")
    bs.add_argument("--tempo-change-window-bars",     type=int,   default=8,    dest="tempo_change_window_bars")
    bs.add_argument("--tempo-change-persist-bars",    type=int,   default=4,    dest="tempo_change_persist_bars")
    bs.add_argument("--tempo-change-threshold-pct",   type=float, default=0.10, dest="tempo_change_threshold_pct")
    bs.add_argument("--tempo-change-threshold-floor", type=float, default=6.0,  dest="tempo_change_threshold_floor")
    bs.add_argument("--pyrb-crispness",               type=int,   default=None, dest="pyrb_crispness",
                    help="rubberband --crispness 0–6 (default: library default)")

    # ── Stem splitter ───────────────────────────────────────
    stems = p.add_argument_group("Stem splitter")
    stems.add_argument("--skip-stems",  action="store_true", help="Skip stem splitting")
    stems.add_argument("--stems",       default=None,
                       help="Comma-separated stems to keep e.g. 'vocals,drums' (default: all)")
    stems.add_argument("--stem-model",  default="htdemucs",
                       choices=["htdemucs_6s", "htdemucs", "htdemucs_ft", "mdx_extra"],
                       dest="stem_model",
                       help="Demucs model (default: htdemucs = 4 stems, fastest)")
    stems.add_argument("--session-type", default=None, dest="session_type",
                       help="Session instrument; mix all other stems into <input_base>_backing_track.wav")
    dem = p.add_argument_group("Demucs (library knobs)")
    dem.add_argument("--demucs-shifts",  type=int,   default=0,    dest="demucs_shifts")
    dem.add_argument("--demucs-overlap", type=float, default=0.0,  dest="demucs_overlap",
                     help="Chunk overlap (default: 0 = fastest; demucs default 0.25)")
    dem.add_argument("--demucs-jobs",    type=int,   default=0,    dest="demucs_jobs")
    dem.add_argument("--demucs-segment", type=int,   default=0,    dest="demucs_segment")
    dem.add_argument("--demucs-device",  default="auto", choices=("auto", "cpu", "cuda", "mps"),
                     dest="demucs_device")
    dem.add_argument("--demucs-int24",   action="store_true", dest="demucs_int24")
    dem.add_argument("--demucs-mp3",     action="store_true", dest="demucs_mp3")
    pres = p.add_argument_group("Stem presence detector")
    pres.add_argument("--presence-db",        type=float, default=-30.0, dest="presence_db")
    pres.add_argument("--presence-window-s",  type=float, default=1.0,   dest="presence_window_s")
    pres.add_argument("--presence-run-s",     type=float, default=2.0,   dest="presence_run_s")
    bt = p.add_argument_group("Backing track")
    bt.add_argument("--backing-peak-dbfs",  type=float, default=-1.0, dest="backing_peak_dbfs")
    bt.add_argument("--backing-bit-depth",  type=int,   default=24,   choices=(16, 24, 32),
                    dest="backing_bit_depth")

    p.add_argument("--format", default="m4a", choices=("m4a", "wav"), dest="out_format",
                   help="Final audio format for stabilised track, stems and backing track "
                        "(default: m4a = AAC 192 kb/s; wav keeps the lossless intermediates)")
    p.add_argument("--skip-checks", action="store_true", dest="skip_checks",
                   help="Record quality checks in the analysis JSON but never abort")
    p.add_argument("--progress-json", action="store_true", dest="progress_json",
                   help="Emit machine-readable PROGRESS JSON lines on stdout, "
                        "remapping each child stage onto a global 0..100 scale.")

    return p.parse_args()


# ── Progress plumbing ───────────────────────────────────────────────────────

_PROGRESS_JSON = False


_WORK_DIR: str | None = None


def _stage_failed(label: str, ret: int) -> None:
    """Exit with the stage's code (2 = quality gate / EARLY_STOP) and drop _work."""
    if _WORK_DIR:
        shutil.rmtree(_WORK_DIR, ignore_errors=True)
    print(f"\n✗  {label} failed (exit {ret}).", file=sys.stderr)
    sys.exit(2 if ret == 2 else 1)


def _emit_global(stage: str, pct: float, msg: str | None = None) -> None:
    """Emit a PROGRESS line on stdout in the global 0..100 scale."""
    if not _PROGRESS_JSON:
        return
    payload = {"stage": stage, "pct": round(float(pct), 2)}
    if msg:
        payload["msg"] = msg
    sys.stdout.write(f"PROGRESS {json.dumps(payload)}\n")
    sys.stdout.flush()


def run_with_progress(
    cmd: list[str],
    label: str,
    stage: str,
    pct_start: float,
    pct_end: float,
) -> None:
    """Run a child process, remapping its PROGRESS lines onto the global
    [pct_start, pct_end] window. Other stdout passes through unchanged."""
    print(f"\n{'='*54}")
    print(f"  {label}")
    print(f"{'='*54}")
    sys.stdout.flush()

    if not _PROGRESS_JSON:
        result = subprocess.run(cmd)
        if result.returncode != 0:
            _stage_failed(label, result.returncode)
        return

    _emit_global(stage, pct_start, label)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=None,           # inherit — forwards demucs tqdm etc.
        bufsize=1,
        text=True,
    )
    assert proc.stdout is not None
    span = pct_end - pct_start
    for raw in proc.stdout:
        line = raw.rstrip("\n")
        if line.startswith("PROGRESS "):
            try:
                payload = json.loads(line[len("PROGRESS "):])
                local_pct = float(payload.get("pct", 0.0))
                global_pct = pct_start + max(0.0, min(1.0, local_pct)) * span
                _emit_global(stage, global_pct, payload.get("msg"))
                continue
            except (ValueError, json.JSONDecodeError):
                pass
        sys.stdout.write(line + "\n")
        sys.stdout.flush()
    ret = proc.wait()
    if ret != 0:
        _stage_failed(label, ret)
    _emit_global(stage, pct_end, f"{label} done")


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    global _PROGRESS_JSON, _WORK_DIR
    args = parse_args()
    _PROGRESS_JSON = args.progress_json

    if not os.path.isfile(args.input):
        sys.exit(f"File not found: {args.input}")

    script_dir    = os.path.dirname(os.path.abspath(__file__))
    demucs_python = os.path.join(script_dir, "venv_demucs", "bin", "python")
    stabilizer    = os.path.join(script_dir, "beat_stabilizer.py")
    stem_splitter = os.path.join(script_dir, "stem_splitter.py")

    if not args.skip_stems and not os.path.isfile(demucs_python):
        sys.exit(f"venv_demucs not found at {demucs_python}\nRun  bash setup.sh  first.")

    input_base = os.path.splitext(os.path.basename(args.input))[0]
    base_dir   = (os.path.abspath(args.output_dir) if args.output_dir
                  else os.path.expanduser("~/Desktop/audio-tools-tests"))
    out_dir    = os.path.join(base_dir, input_base)
    os.makedirs(out_dir, exist_ok=True)

    # Stage weights → global 0..100 progress ranges.
    weights = {
        "stabilize": 0.0 if args.skip_stabilize else 15.0,
        "stems":     0.0 if args.skip_stems     else 85.0,
    }
    scale  = 100.0 / (sum(weights.values()) or 1.0)
    cursor = 0.0
    ranges: dict[str, tuple[float, float]] = {}
    for stage in ("stabilize", "stems"):
        w = weights[stage] * scale
        ranges[stage] = (cursor, cursor + w)
        cursor += w

    _emit_global("start", 0.0, "pipeline starting")
    analysis_json = os.path.join(out_dir, input_base + "_analysis.json")

    # ── Step 0: decode any input format → 44.1 kHz stereo WAV ────────────
    # One code path for every container/codec ffmpeg can read (mp3, m4a/aac,
    # flac, ogg, aiff, wav of any bit depth / sample rate / channel count).
    work_dir = _WORK_DIR = os.path.join(out_dir, "_work")
    os.makedirs(work_dir, exist_ok=True)
    source_wav = os.path.join(work_dir, input_base + "_source.wav")
    r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", args.input,
                        "-vn", "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le", source_wav])
    if r.returncode != 0 or not os.path.isfile(source_wav):
        sys.exit(f"✗  Could not decode {args.input} with ffmpeg (unsupported or corrupt file).")

    # ── Step 1: detect BPM + meter, stabilize, add lead-in ───────────────
    if args.skip_stabilize:
        print("\n[pipeline] Skipping beat stabilization.")
        stabilised = source_wav
    else:
        stabilised = os.path.join(work_dir, input_base + "_stabilised.wav")
        cmd = [sys.executable, stabilizer, "-i", source_wav, "-o", stabilised,
               "--analysis-out", analysis_json,
               "--lead-in-bars", str(args.lead_in_bars)]
        if args.bpm:                         cmd += ["--bpm",           str(args.bpm)]
        if args.strength != 1.0:             cmd += ["--strength",      str(args.strength)]
        if not args.trim_intro:              cmd += ["--no-trim-intro"]
        if args.beats_per_bar:               cmd += ["--beats-per-bar", str(args.beats_per_bar)]
        if args.allow_tempo_change:          cmd += ["--allow-tempo-change"]
        if args.skip_checks:                 cmd += ["--skip-checks"]
        cmd += ["--detector-backend", args.detector_backend]
        if args.detector_timeout_s != 300:   cmd += ["--detector-timeout-s", str(args.detector_timeout_s)]
        if args.librosa_start_bpm != 120.0:  cmd += ["--librosa-start-bpm", str(args.librosa_start_bpm)]
        if args.librosa_tightness != 100.0:  cmd += ["--librosa-tightness", str(args.librosa_tightness)]
        if args.librosa_hop_length != 512:   cmd += ["--librosa-hop-length", str(args.librosa_hop_length)]
        if args.tempo_change_window_bars != 8:
            cmd += ["--tempo-change-window-bars", str(args.tempo_change_window_bars)]
        if args.tempo_change_persist_bars != 4:
            cmd += ["--tempo-change-persist-bars", str(args.tempo_change_persist_bars)]
        if args.tempo_change_threshold_pct != 0.10:
            cmd += ["--tempo-change-threshold-pct", str(args.tempo_change_threshold_pct)]
        if args.tempo_change_threshold_floor != 6.0:
            cmd += ["--tempo-change-threshold-floor", str(args.tempo_change_threshold_floor)]
        if args.pyrb_crispness is not None:  cmd += ["--pyrb-crispness", str(args.pyrb_crispness)]
        if _PROGRESS_JSON:                   cmd += ["--progress-json"]
        run_with_progress(cmd, "STEP 1 / 2  —  BPM / Meter + Beat Stabilization",
                          "stabilize", *ranges["stabilize"])
        if not os.path.isfile(stabilised):
            sys.exit(f"\n✗  Beat stabilizer reported success but did not write:\n   {stabilised}")

    # ── Step 2: stems ────────────────────────────────────────────────────
    stems_out          = os.path.join(work_dir, input_base + "_stems")
    backing_track_path = os.path.join(work_dir, input_base + "_backing_track.wav")

    if args.skip_stems:
        print("\n[pipeline] Skipping stem splitting.")
    else:
        cmd = [demucs_python, stem_splitter, "-i", stabilised, "-o", stems_out,
               "--model", args.stem_model]
        if args.stems: cmd += ["--stems", args.stems]
        if args.session_type:
            cmd += ["--session-type", args.session_type,
                    "--backing-track-out", backing_track_path]
        cmd += ["--demucs-shifts", str(args.demucs_shifts)]
        cmd += ["--demucs-overlap", str(args.demucs_overlap)]
        if args.demucs_jobs > 0:             cmd += ["--demucs-jobs",    str(args.demucs_jobs)]
        if args.demucs_segment > 0:          cmd += ["--demucs-segment", str(args.demucs_segment)]
        if args.demucs_device != "auto":     cmd += ["--demucs-device",  args.demucs_device]
        if args.demucs_int24:                cmd += ["--demucs-int24"]
        if args.demucs_mp3:                  cmd += ["--demucs-mp3"]
        if args.presence_db != -30.0:        cmd += ["--presence-db",       str(args.presence_db)]
        if args.presence_window_s != 1.0:    cmd += ["--presence-window-s", str(args.presence_window_s)]
        if args.presence_run_s != 2.0:       cmd += ["--presence-run-s",    str(args.presence_run_s)]
        if args.backing_peak_dbfs != -1.0:   cmd += ["--backing-peak-dbfs", str(args.backing_peak_dbfs)]
        if args.backing_bit_depth != 24:     cmd += ["--backing-bit-depth", str(args.backing_bit_depth)]
        if _PROGRESS_JSON:                   cmd += ["--progress-json"]
        run_with_progress(cmd, "STEP 2 / 2  —  Stem Splitting",
                          "stems", *ranges["stems"])

    # ── Step 3: export final files ───────────────────────────────────────
    import shutil

    def _export(src: str, dst_noext: str) -> str:
        if args.out_format == "wav":
            dst = dst_noext + ".wav"
            shutil.move(src, dst)
        else:
            dst = dst_noext + ".m4a"
            r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src,
                                "-c:a", "aac", "-b:a", "192k", dst])
            if r.returncode != 0:
                sys.exit(f"✗  ffmpeg failed encoding {dst}")
        return dst

    finals: list[str] = []
    if not args.skip_stabilize:
        finals.append(_export(stabilised, os.path.join(out_dir, input_base + "_stabilised")))
        if os.path.isfile(stabilised + ".bpm"):
            shutil.move(stabilised + ".bpm", os.path.join(out_dir, input_base + ".bpm"))
    if not args.skip_stems and os.path.isdir(stems_out):
        final_stems = os.path.join(out_dir, input_base + "_stems")
        os.makedirs(final_stems, exist_ok=True)
        for name in sorted(os.listdir(stems_out)):
            src = os.path.join(stems_out, name)
            stem, ext = os.path.splitext(name)
            if ext in (".wav", ".mp3"):
                finals.append(_export(src, os.path.join(final_stems, stem)))
            elif ext == ".json":
                shutil.move(src, os.path.join(final_stems, name))
        if args.session_type and os.path.isfile(backing_track_path):
            finals.append(_export(backing_track_path,
                                  os.path.join(out_dir, input_base + "_backing_track")))
    shutil.rmtree(work_dir, ignore_errors=True)

    print(f"\n{'='*54}")
    print("  ✓  Pipeline complete!")
    print(f"     Output folder : {out_dir}")
    for f in finals:
        print(f"       {os.path.relpath(f, out_dir)}")
    if not args.skip_stabilize:
        print(f"     Analysis JSON : {os.path.basename(analysis_json)}")
    print(f"{'='*54}\n")
    _emit_global("done", 100.0, "pipeline complete")


if __name__ == "__main__":
    main()
