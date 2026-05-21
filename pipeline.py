#!/usr/bin/env python3
"""
pipeline.py  —  Full pipeline: beat stabilize → chord chart PDF → stem split.

Run with system Python (python3):
    python3 pipeline.py -i song.wav
    python3 pipeline.py -i song.m4a --bpm 84 --title "My Song" --open
    python3 pipeline.py -i song.wav --skip-stabilize
    python3 pipeline.py -i song.wav --skip-stems
    python3 pipeline.py -i song.wav --stems vocals,drums

The stabiliser writes a BPM sidecar that the chord chart picks up
automatically, so you never have to pass --bpm twice.

Intro trim is on by default: the stabilised WAV starts exactly one bar
before the first detected beat, ready to drop into a DAW at bar 1.
Disable with --no-trim-intro.

Half-time songs (e.g. 78 BPM where the detector locks onto 156 BPM 8th
notes) are handled automatically when --bpm is supplied.  The stabiliser
compares beat count vs expected count and uses 2× anchor density when a
half-time groove is detected.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Beat-stabilize → chord chart → stem split.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 pipeline.py -i song.wav
  python3 pipeline.py -i song.m4a --bpm 84 --title "My Song" --open
  python3 pipeline.py -i song.wav --strength 0.8 --key "bes:major"
  python3 pipeline.py -i song.wav --stems vocals,bass
  python3 pipeline.py -i already_stable.wav --skip-stabilize --open
        """,
    )
    # ── Input ───────────────────────────────────────────────
    p.add_argument("-i", "--input",      required=True, help="Input audio file")
    p.add_argument("--title",            default=None,  help="Chart title (default: filename)")
    p.add_argument("--output-dir",       default=None,  dest="output_dir",
                   help="Directory for all output files (default: same as input)")
    p.add_argument("--open",             action="store_true", help="Open PDF when done")

    # ── Beat stabilizer ─────────────────────────────────────
    stab = p.add_argument_group("Beat stabilizer")
    stab.add_argument("--bpm",            type=float, default=None,
                      help="Target BPM (auto-detected if omitted)")
    stab.add_argument("--strength",       type=float, default=1.0, metavar="0-1",
                      help="Quantisation strength (default: 1.0)")
    stab.add_argument("--skip-stabilize", action="store_true",
                      help="Skip beat stabilization and use input directly")
    # Trim-intro is ON by default so the output drops into a DAW without manual
    # offsetting.  Pass --no-trim-intro to get the raw stabilised audio.
    stab.add_argument("--no-trim-intro",  action="store_false", dest="trim_intro",
                      help="Disable the default intro trim")
    stab.add_argument("--trim-intro",     action="store_true",  dest="trim_intro",
                      help="Start the output one bar before beat 1 (on by default; "
                           "silence-padded when the first beat is near the file start)")
    p.set_defaults(trim_intro=True)
    stab.add_argument("--beats-per-bar",  type=int, default=4, dest="beats_per_bar",
                      help="Beats per bar for the intro trim length (default: 4)")
    stab.add_argument("--allow-tempo-change", action="store_true", dest="allow_tempo_change",
                      help="Continue even if a sustained tempo change is detected. "
                           "Default behaviour is to stop with EARLY_STOP so the caller can warn.")

    # ── Chord chart ─────────────────────────────────────────
    chart = p.add_argument_group("Chord chart")
    chart.add_argument("--key",           default="auto",
                       help="Key signature e.g. 'f:minor', 'bes:major' (default: auto)")
    chart.add_argument("--time-sig",      type=int, default=None, dest="time_sig",
                       help="Beats per bar (default: auto)")
    chart.add_argument("--bars-per-line", type=int, default=4, dest="bars_per_line")
    chart.add_argument("--skip-sections", action="store_true", dest="skip_sections",
                       help="Skip allin1 structural segmentation (no section rehearsal marks).")
    chart.add_argument("--section-threshold", type=float, default=0.0, dest="section_threshold",
                       help=(
                           "Boundary-strength threshold for allin1 section detection "
                           "(default: 0.0 = accept every local peak). Higher values produce "
                           "fewer but more confident sections. Typical useful range: 0.0–0.5."
                       ))
    chart.add_argument("--no-bpm",          action="store_true", help="Omit BPM from chart subtitle")
    chart.add_argument("--no-key",          action="store_true", help="Omit key from chart subtitle")
    chart.add_argument("--no-meter",        action="store_true", help="Omit meter from chart subtitle")
    chart.add_argument("--subtitle",        default=None, help="Override entire subtitle text")
    chart.add_argument("--add-7th",         action="store_true", dest="add_7th",
                       help="Keep maj7, m7, dominant 7 chords (default: simplify to major/minor)")
    chart.add_argument("--mid-bar-threshold", type=float, default=0.80, dest="mid_bar_threshold",
                       help="Confidence threshold for mid-bar chord changes (default: 0.80)")
    chart.add_argument("--no-madmom-fallback",  action="store_false", dest="madmom_fallback",
                       help="Disable the default madmom fallback for low-confidence bars")
    chart.add_argument("--madmom-fallback",     action="store_true",  dest="madmom_fallback",
                       help="Re-evaluate low-confidence bars with madmom (on by default)")
    p.set_defaults(madmom_fallback=True)
    chart.add_argument("--madmom-threshold",   type=float, default=0.70, dest="madmom_threshold",
                       help="Bar mean-confidence below which madmom fallback triggers (default: 0.70)")
    chart.add_argument("--key-tiebreak",       action="store_true", dest="key_tiebreak",
                       help="Refine chromagram key by chord-root frequency (resolves major/minor ambiguity)")
    chart.add_argument("--key-snap",           action="store_true", dest="key_snap",
                       help="Snap non-diatonic low-confidence chords to the nearest diatonic equivalent")
    chart.add_argument("--key-snap-threshold", type=float, default=0.65, dest="key_snap_threshold",
                       help="Bars below this confidence are eligible for key snapping (default: 0.65)")
    chart.add_argument("--half-time",          action="store_true", dest="half_time",
                       help="Keep every other beat (fixes half-time grooves where the tracker "
                            "locks onto 8th notes). Auto-triggered when --bpm is ~half the detected rate.")
    chart.add_argument("--compound",           action="store_true", dest="compound",
                       help="Force 6/8 notation when beats-per-bar is 3 (auto-detected for most 6/8 songs).")

    # ── Beat detector library knobs (relayed to beat_stabilizer + chord_chart_render) ──
    det = p.add_argument_group("Beat detector (library knobs)")
    det.add_argument("--detector-backend", default="auto",
                     choices=("auto", "madmom", "librosa"), dest="detector_backend",
                     help="Beat detector backend (default: auto)")
    det.add_argument("--madmom-bpb-options", default="3,4", dest="madmom_bpb_options",
                     help="Candidate beats-per-bar for madmom downbeats (default: 3,4)")
    det.add_argument("--madmom-fps",         type=int,   default=100, dest="madmom_fps",
                     help="madmom RNN/DBN frame rate Hz (default: 100)")
    det.add_argument("--madmom-timeout-s",   type=int,   default=240, dest="madmom_timeout_s",
                     help="Madmom subprocess timeout in seconds (default: 240)")
    det.add_argument("--librosa-start-bpm",  type=float, default=120.0, dest="librosa_start_bpm",
                     help="librosa beat-tracker initial tempo guess (default: 120)")
    det.add_argument("--librosa-tightness",  type=float, default=100.0, dest="librosa_tightness",
                     help="librosa beat-tracker tightness (default: 100)")
    det.add_argument("--librosa-hop-length", type=int,   default=512,   dest="librosa_hop_length",
                     help="librosa STFT hop length in samples (default: 512)")
    det.add_argument("--ts-window-factor",   type=float, default=0.15,  dest="ts_window_factor",
                     help="Time-signature autocorrelation window factor (default: 0.15)")
    # ── Beat-stabilizer extra knobs ──
    bs = p.add_argument_group("Beat stabilizer (library knobs)")
    bs.add_argument("--intro-trim-bars",            type=int,   default=1,    dest="intro_trim_bars",
                    help="Bars before the first beat retained when trimming (default: 1)")
    bs.add_argument("--tempo-change-window-bars",   type=int,   default=8,    dest="tempo_change_window_bars",
                    help="Tempo-change scanner rolling-median window (default: 8 bars)")
    bs.add_argument("--tempo-change-persist-bars",  type=int,   default=4,    dest="tempo_change_persist_bars",
                    help="Bars new tempo must persist before firing the guard (default: 4)")
    bs.add_argument("--tempo-change-threshold-pct", type=float, default=0.06, dest="tempo_change_threshold_pct",
                    help="Percentage tempo step counting as a change (default: 0.06)")
    bs.add_argument("--tempo-change-threshold-floor", type=float, default=6.0, dest="tempo_change_threshold_floor",
                    help="Minimum absolute BPM step counting as a change (default: 6)")
    bs.add_argument("--pyrb-crispness",             type=int,   default=None, dest="pyrb_crispness",
                    help="rubberband --crispness 0–6 (default: library default)")
    # ── Chord-detection extra knobs ──
    cd = p.add_argument_group("Chord detection (library knobs)")
    cd.add_argument("--no-bar-phase",        action="store_false", dest="bar_phase",
                    help="Disable chord-grid phase alignment to bar downbeats")
    p.set_defaults(bar_phase=True)
    cd.add_argument("--confidence-warn",     type=float, default=0.45, dest="confidence_warn",
                    help="Confidence below which a chord is flagged '?' (default: 0.45)")
    cd.add_argument("--hpss-mode",           default="hpss",
                    choices=("off", "hpss", "hpss-no-drums"), dest="hpss_mode",
                    help="HPSS preprocessing of the time-aligned audio before crema. "
                         "'off' = full mix; 'hpss' = strip drum transients (default); "
                         "'hpss-no-drums' = also subtract the drums stem (requires stems).")
    cd.add_argument("--hpss-margin",         type=float, default=3.0, dest="hpss_margin",
                    help="HPSS margin (higher = more aggressive separation; default: 3.0)")
    cd.add_argument("--bass-anchor",         action="store_true", dest="bass_anchor",
                    help="Use the bass stem's dominant pitch to correct chord roots "
                         "(requires stems). Most useful when crema confuses a major chord "
                         "with its relative minor or labels an inversion as the bass note's "
                         "chord. Adds Demucs time when stems aren't already enabled-first.")
    cd.add_argument("--section-consistency", action="store_true", dest="section_consistency",
                    help="Force same-named sections (Chorus 1, Chorus 2, …) to share their "
                         "chord progression by voting per bar position. Requires section "
                         "detection to be on. Pure post-processing — costs nothing.")
    cd.add_argument("--slash-chords",        action="store_true", dest="slash_chords",
                    help="Detect chord inversions like C/E or G/B by reading the bass stem. "
                         "Requires stems (same constraint as --bass-anchor). Costs almost "
                         "nothing once stems are available.")
    cd.add_argument("--viterbi-smoothing",   action="store_true", dest="viterbi_smoothing",
                    help="Key-conditioned Viterbi smoothing of the chord sequence. Replaces "
                         "one-off chord misdetections that violate basic music-theory priors "
                         "(e.g. a stray Bdim between two Cs in C major). Bars with mid-bar "
                         "splits are left untouched.")
    cd.add_argument("--viterbi-stay-prob",   type=float, default=0.35, dest="viterbi_stay_prob",
                    help="Viterbi self-transition prior (default: 0.35; higher = stickier).")
    cd.add_argument("--viterbi-cadence-boost", type=float, default=4.0, dest="viterbi_cadence_boost",
                    help="Viterbi cadence-transition multiplier (default: 4.0).")

    # ── Stem splitter ────────────────────────────────────────
    stems = p.add_argument_group("Stem splitter")
    stems.add_argument("--skip-stems",  action="store_true", help="Skip stem splitting")
    stems.add_argument("--stems",       default=None,
                       help="Comma-separated stems to keep e.g. 'vocals,drums' (default: all)")
    stems.add_argument("--stem-model",  default="htdemucs_6s",
                       choices=["htdemucs_6s", "htdemucs", "htdemucs_ft", "mdx_extra"],
                       dest="stem_model",
                       help="Demucs model (default: htdemucs_6s = 6 stems)")
    stems.add_argument("--session-type", default=None, dest="session_type",
                       help="Session instrument; mix all other stems into <input_base>_backing_track.wav")
    # ── Demucs library knobs ──
    dem = p.add_argument_group("Demucs (library knobs)")
    dem.add_argument("--demucs-shifts",  type=int,   default=1,    dest="demucs_shifts")
    dem.add_argument("--demucs-overlap", type=float, default=0.25, dest="demucs_overlap")
    dem.add_argument("--demucs-jobs",    type=int,   default=0,    dest="demucs_jobs")
    dem.add_argument("--demucs-segment", type=int,   default=0,    dest="demucs_segment")
    dem.add_argument("--demucs-device",  default="auto", choices=("auto","cpu","cuda","mps"),
                     dest="demucs_device")
    dem.add_argument("--demucs-int24",   action="store_true", dest="demucs_int24")
    dem.add_argument("--demucs-mp3",     action="store_true", dest="demucs_mp3")
    # ── Stem presence detector knobs ──
    pres = p.add_argument_group("Stem presence detector")
    pres.add_argument("--presence-db",        type=float, default=-30.0, dest="presence_db")
    pres.add_argument("--presence-window-s",  type=float, default=1.0,   dest="presence_window_s")
    pres.add_argument("--presence-run-s",     type=float, default=2.0,   dest="presence_run_s")
    # ── Backing-track mixer knobs ──
    bt = p.add_argument_group("Backing track")
    bt.add_argument("--backing-peak-dbfs",  type=float, default=-1.0, dest="backing_peak_dbfs")
    bt.add_argument("--backing-bit-depth",  type=int,   default=24,   choices=(16,24,32),
                    dest="backing_bit_depth")

    p.add_argument("--progress-json", action="store_true", dest="progress_json",
                   help="Emit machine-readable PROGRESS JSON lines on stdout, "
                        "remapping each child stage onto a global 0..100 scale.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Progress remapping
# ---------------------------------------------------------------------------

_PROGRESS_JSON = False


def _emit_global(stage: str, pct: float, msg: str | None = None) -> None:
    """Emit a PROGRESS line on stdout in the global 0..100 scale."""
    if not _PROGRESS_JSON:
        return
    payload = {"stage": stage, "pct": round(float(pct), 2)}
    if msg:
        payload["msg"] = msg
    sys.stdout.write(f"PROGRESS {json.dumps(payload)}\n")
    sys.stdout.flush()


def run(cmd: list[str], label: str) -> None:
    print(f"\n{'='*54}")
    print(f"  {label}")
    print(f"{'='*54}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"\n✗  {label} failed (exit {result.returncode}).")


def run_with_progress(
    cmd: list[str],
    label: str,
    stage: str,
    pct_start: float,
    pct_end: float,
) -> None:
    """Run a child process, parsing PROGRESS JSON lines from its stdout and
    remapping them onto the global [pct_start, pct_end] window.

    Non-PROGRESS stdout is passed through unchanged so existing debug
    output is preserved.
    """
    print(f"\n{'='*54}")
    print(f"  {label}")
    print(f"{'='*54}")

    if not _PROGRESS_JSON:
        # Fast path: no progress parsing needed.
        result = subprocess.run(cmd)
        if result.returncode != 0:
            sys.exit(f"\n✗  {label} failed (exit {result.returncode}).")
        return

    _emit_global(stage, pct_start, label)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=None,           # inherit — pipeline.py forwards demucs tqdm etc.
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
        # Pass through any non-PROGRESS line unchanged
        sys.stdout.write(line + "\n")
        sys.stdout.flush()
    ret = proc.wait()
    if ret != 0:
        sys.exit(f"\n✗  {label} failed (exit {ret}).")
    _emit_global(stage, pct_end, f"{label} done")


def main() -> None:
    global _PROGRESS_JSON
    args = parse_args()
    _PROGRESS_JSON = args.progress_json

    if not os.path.isfile(args.input):
        sys.exit(f"File not found: {args.input}")

    script_dir      = os.path.dirname(os.path.abspath(__file__))
    crema_python    = os.path.join(script_dir, "venv_crema",   "bin", "python3.11")
    demucs_python   = os.path.join(script_dir, "venv_demucs",  "bin", "python3.11")
    allin1_python   = os.path.join(script_dir, "venv_allin1",  "bin", "python3.11")
    stabilizer      = os.path.join(script_dir, "beat_stabilizer.py")
    chart_render    = os.path.join(script_dir, "chord_chart_render.py")
    allin1_runner   = os.path.join(script_dir, "run_allin1.py")
    stem_splitter   = os.path.join(script_dir, "stem_splitter.py")

    required_venvs = [(crema_python, "venv_crema"), (demucs_python, "venv_demucs")]
    if not args.skip_sections:
        required_venvs.append((allin1_python, "venv_allin1"))
    for venv, name in required_venvs:
        if not os.path.isfile(venv):
            sys.exit(f"{name} not found at {venv}\nRun  bash setup.sh  first.")

    input_base = os.path.splitext(os.path.basename(args.input))[0]
    base_dir   = os.path.abspath(args.output_dir) if args.output_dir else os.path.expanduser("~/Desktop/audio-tools-tests")
    out_dir    = os.path.join(base_dir, input_base)
    os.makedirs(out_dir, exist_ok=True)
    title = args.title or input_base

    # ── Validation for features that need stems before chord ───────────────
    # `hpss-no-drums` (needs drums.wav), `bass-anchor` and `slash-chords`
    # (both need bass.wav) require stems to be produced before the chord
    # step.  Skip-stems makes this impossible — surface a clear error
    # instead of silently degrading.
    _stems_dependent = []
    if args.hpss_mode == "hpss-no-drums": _stems_dependent.append("--hpss-mode=hpss-no-drums")
    if args.bass_anchor:                  _stems_dependent.append("--bass-anchor")
    if args.slash_chords:                 _stems_dependent.append("--slash-chords")
    if _stems_dependent and args.skip_stems:
        sys.exit(
            f"✗  {', '.join(_stems_dependent)} requires stem splitting to be enabled "
            "(needs the bass/drums stems before chord detection)."
        )

    # ── Global progress allocation ─────────────────────────
    # Default ranges: stabilize 0–10, chord 10–40, stems 40–100.
    # When stems run BEFORE chord (hpss-no-drums OR bass-anchor OR slash-chords),
    # swap windows so the global bar stays monotonic: stabilize 0–10,
    # stems 10–70, chord 70–100.
    stems_first = (
        (args.hpss_mode == "hpss-no-drums" or args.bass_anchor or args.slash_chords)
        and not args.skip_stems
    )
    weights = {
        "stabilize": 0.0 if args.skip_stabilize else 10.0,
        "chord":     30.0,
        "stems":     0.0 if args.skip_stems else 60.0,
    }
    total_w = sum(weights.values()) or 1.0
    scale   = 100.0 / total_w
    cursor  = 0.0
    ranges: dict[str, tuple[float, float]] = {}
    stage_order = (
        ("stabilize", "stems", "chord") if stems_first
        else ("stabilize", "chord", "stems")
    )
    for stage in stage_order:
        w = weights[stage] * scale
        ranges[stage] = (cursor, cursor + w)
        cursor += w

    _emit_global("start", 0.0, "pipeline starting")

    # ── Step 1 / 3  —  Beat stabilization ───────────────────
    if args.skip_stabilize:
        print("\n[pipeline] Skipping beat stabilization.")
        stabilised = args.input
    else:
        stabilised = os.path.join(out_dir, input_base + "_stabilised.wav")
        cmd = [sys.executable, stabilizer, "-i", args.input, "-o", stabilised]
        if args.bpm:                cmd += ["--bpm",           str(args.bpm)]
        if args.strength != 1.0:    cmd += ["--strength",      str(args.strength)]
        if not args.trim_intro:     cmd += ["--no-trim-intro"]
        if args.beats_per_bar != 4: cmd += ["--beats-per-bar", str(args.beats_per_bar)]
        if args.allow_tempo_change: cmd += ["--allow-tempo-change"]
        if args.intro_trim_bars != 1: cmd += ["--intro-trim-bars", str(args.intro_trim_bars)]
        # Beat detector knobs
        if args.detector_backend != "auto": cmd += ["--detector-backend", args.detector_backend]
        if args.madmom_bpb_options != "3,4": cmd += ["--madmom-bpb-options", args.madmom_bpb_options]
        if args.madmom_fps != 100: cmd += ["--madmom-fps", str(args.madmom_fps)]
        if args.madmom_timeout_s != 240: cmd += ["--madmom-timeout-s", str(args.madmom_timeout_s)]
        if args.librosa_start_bpm != 120.0: cmd += ["--librosa-start-bpm", str(args.librosa_start_bpm)]
        if args.librosa_tightness != 100.0: cmd += ["--librosa-tightness", str(args.librosa_tightness)]
        if args.librosa_hop_length != 512: cmd += ["--librosa-hop-length", str(args.librosa_hop_length)]
        # Tempo-change guard knobs
        if args.tempo_change_window_bars != 8: cmd += ["--tempo-change-window-bars", str(args.tempo_change_window_bars)]
        if args.tempo_change_persist_bars != 4: cmd += ["--tempo-change-persist-bars", str(args.tempo_change_persist_bars)]
        if args.tempo_change_threshold_pct != 0.06: cmd += ["--tempo-change-threshold-pct", str(args.tempo_change_threshold_pct)]
        if args.tempo_change_threshold_floor != 6.0: cmd += ["--tempo-change-threshold-floor", str(args.tempo_change_threshold_floor)]
        if args.pyrb_crispness is not None: cmd += ["--pyrb-crispness", str(args.pyrb_crispness)]
        if _PROGRESS_JSON:          cmd += ["--progress-json"]
        run_with_progress(
            cmd, "STEP 1 / 3  —  Beat Stabilization",
            "stabilize", *ranges["stabilize"],
        )
        if not os.path.isfile(stabilised):
            sys.exit(
                f"\n✗  Beat stabilizer reported success but did not write the output file:\n"
                f"   {stabilised}\n"
                f"   This is unexpected. Please try again."
            )

    # Paths we may need before stems actually run (when stems_first, the chord
    # step needs --drums-wav pointed at the as-yet-unproduced file).
    stems_out          = os.path.join(out_dir, input_base + "_stems")
    backing_track_path = os.path.join(out_dir, input_base + "_backing_track.wav")
    drums_wav_path     = os.path.join(stems_out, "drums.wav")
    bass_wav_path      = os.path.join(stems_out, "bass.wav")
    stems_done = False

    def _build_stems_cmd(stems_filter: str | None) -> list[str]:
        cmd = [
            demucs_python, stem_splitter,
            "-i", stabilised,
            "-o", stems_out,
            "--model", args.stem_model,
        ]
        if stems_filter: cmd += ["--stems", stems_filter]
        if args.session_type:
            cmd += ["--session-type", args.session_type,
                    "--backing-track-out", backing_track_path]
        # Demucs library knobs
        if args.demucs_shifts != 1:     cmd += ["--demucs-shifts",  str(args.demucs_shifts)]
        if args.demucs_overlap != 0.25: cmd += ["--demucs-overlap", str(args.demucs_overlap)]
        if args.demucs_jobs > 0:        cmd += ["--demucs-jobs",    str(args.demucs_jobs)]
        if args.demucs_segment > 0:     cmd += ["--demucs-segment", str(args.demucs_segment)]
        if args.demucs_device != "auto": cmd += ["--demucs-device", args.demucs_device]
        if args.demucs_int24:           cmd += ["--demucs-int24"]
        if args.demucs_mp3:             cmd += ["--demucs-mp3"]
        # Presence detector
        if args.presence_db != -30.0:        cmd += ["--presence-db",        str(args.presence_db)]
        if args.presence_window_s != 1.0:    cmd += ["--presence-window-s",  str(args.presence_window_s)]
        if args.presence_run_s != 2.0:       cmd += ["--presence-run-s",     str(args.presence_run_s)]
        # Backing track
        if args.backing_peak_dbfs != -1.0:   cmd += ["--backing-peak-dbfs",  str(args.backing_peak_dbfs)]
        if args.backing_bit_depth != 24:     cmd += ["--backing-bit-depth",  str(args.backing_bit_depth)]
        if _PROGRESS_JSON: cmd += ["--progress-json"]
        return cmd

    # ── Step 2 / 3 (or 3 / 3) — early stems pass when chord step needs stems ─
    # Triggered when --hpss-mode=hpss-no-drums (needs drums.wav) or
    # --bass-anchor (needs bass.wav).  Both stems are silently added to the
    # user's stem filter if absent so the required WAVs always exist.
    if stems_first:
        stems_filter = args.stems
        needed = []
        if args.hpss_mode == "hpss-no-drums":         needed.append("drums")
        if args.bass_anchor or args.slash_chords:     needed.append("bass")
        if stems_filter is not None:
            current = stems_filter.split(",")
            added = [s for s in needed if s not in current]
            if added:
                stems_filter = ",".join(current + added)
                print(f"[pipeline] chord features need {added}: stem filter → {stems_filter}")
        why = []
        if args.hpss_mode == "hpss-no-drums": why.append("HPSS drum removal")
        if args.bass_anchor:                  why.append("bass-anchor")
        if args.slash_chords:                 why.append("slash chords")
        run_with_progress(
            _build_stems_cmd(stems_filter),
            f"STEP 2 / 3  —  Stem Splitting (early, for {' + '.join(why)})",
            "stems", *ranges["stems"],
        )
        stems_done = True
        if args.hpss_mode == "hpss-no-drums" and not os.path.isfile(drums_wav_path):
            print(f"[pipeline] WARNING: expected drums stem not produced at {drums_wav_path} — "
                  "chord step will fall back to plain HPSS.")
        if (args.bass_anchor or args.slash_chords) and not os.path.isfile(bass_wav_path):
            print(f"[pipeline] WARNING: expected bass stem not produced at {bass_wav_path} — "
                  "chord step will skip bass-anchor / slash-chords.")

    # ── allin1 section detection — runs in parallel with chord chart ────────
    # allin1 internally runs Demucs, which takes ~3-5 min on CPU.  Starting it
    # as a background process alongside chord_chart_render (which takes ~5-8 min
    # for crema + madmom) hides the allin1 cost entirely on typical songs.
    # chord_chart_render polls for the JSON file via --sections-json-wait-s.
    # stdout=None inherits the parent pipe so the web UI log receives allin1 lines.
    sections_json = os.path.join(out_dir, input_base + "_sections.json")
    allin1_proc: subprocess.Popen | None = None
    if not args.skip_sections:
        print("\n[pipeline] Starting section detection (allin1) in background …")
        print("[pipeline] Note: first run downloads ~1.5 GB model weights — may take extra time.")
        sys.stdout.flush()
        _emit_global("sections", ranges["chord"][0], "section detection running in background")
        allin1_cmd = [allin1_python, allin1_runner, "-i", stabilised, "-o", sections_json]
        if args.section_threshold:
            allin1_cmd += ["--section-threshold", str(args.section_threshold)]
        allin1_proc = subprocess.Popen(
            allin1_cmd,
            stdout=None,   # inherit — allin1 log lines go to the same pipe
            stderr=None,
        )

    # ── Step 2 / 3  —  Chord chart ──────────────────────────
    chart_out = os.path.join(out_dir, input_base + "_chord_chart")
    cmd = [
        crema_python, chart_render,
        "-i", stabilised, "--title", title, "--output", chart_out,
        "--bars-per-line", str(args.bars_per_line),
    ]
    if args.key != "auto":              cmd += ["--key",              args.key]
    if args.bpm and args.skip_stabilize: cmd += ["--bpm",             str(args.bpm)]
    if args.time_sig:                   cmd += ["--time-sig",         str(args.time_sig)]
    if args.no_bpm:                     cmd += ["--no-bpm"]
    if args.no_key:                     cmd += ["--no-key"]
    if args.no_meter:                   cmd += ["--no-meter"]
    if args.subtitle is not None:       cmd += ["--subtitle",         args.subtitle]
    if args.add_7th:                    cmd += ["--add-7th"]
    if args.mid_bar_threshold != 0.80:  cmd += ["--mid-bar-threshold", str(args.mid_bar_threshold)]
    if not args.madmom_fallback:              cmd += ["--no-madmom-fallback"]
    if args.madmom_threshold != 0.70:         cmd += ["--madmom-threshold",   str(args.madmom_threshold)]
    if args.key_tiebreak:                     cmd += ["--key-tiebreak"]
    if args.key_snap:                         cmd += ["--key-snap"]
    if args.key_snap_threshold != 0.65:       cmd += ["--key-snap-threshold", str(args.key_snap_threshold)]
    if args.half_time:                        cmd += ["--half-time"]
    if args.compound:                         cmd += ["--compound"]
    if not args.skip_sections:
        cmd += ["--sections-json", sections_json, "--sections-json-wait-s", "600"]
    if args.open:                             cmd += ["--open"]
    # Chord-detection library knobs
    if not args.bar_phase:                    cmd += ["--no-bar-phase"]
    if args.confidence_warn != 0.45:          cmd += ["--threshold", str(args.confidence_warn)]
    # HPSS preprocessing
    if args.hpss_mode != "hpss":              cmd += ["--hpss-mode", args.hpss_mode]
    if args.hpss_margin != 3.0:               cmd += ["--hpss-margin", str(args.hpss_margin)]
    if args.hpss_mode == "hpss-no-drums":     cmd += ["--drums-wav", drums_wav_path]
    # Bass-anchored root correction (needs bass.wav from the early stems pass)
    if args.bass_anchor:                      cmd += ["--bass-anchor", "--bass-wav", bass_wav_path]
    # Slash chord (inversion) labelling — same bass.wav dependency.
    if args.slash_chords:
        cmd += ["--slash-chords"]
        if "--bass-wav" not in cmd:           cmd += ["--bass-wav", bass_wav_path]
    # Section-aware chord consistency (post-processing; needs sections-json)
    if args.section_consistency:              cmd += ["--section-consistency"]
    # Key-conditioned Viterbi smoothing (uses crema posteriors only — no stems).
    if args.viterbi_smoothing:
        cmd += ["--viterbi-smoothing"]
        if args.viterbi_stay_prob != 0.35:    cmd += ["--viterbi-stay-prob", str(args.viterbi_stay_prob)]
        if args.viterbi_cadence_boost != 4.0: cmd += ["--viterbi-cadence-boost", str(args.viterbi_cadence_boost)]
    # Beat-detector knobs (also used by chord_chart_render's own beat detection)
    if args.ts_window_factor != 0.15:         cmd += ["--ts-window-factor", str(args.ts_window_factor)]
    if args.librosa_start_bpm != 120.0:       cmd += ["--librosa-start-bpm", str(args.librosa_start_bpm)]
    if args.librosa_tightness != 100.0:       cmd += ["--librosa-tightness", str(args.librosa_tightness)]
    if args.librosa_hop_length != 512:        cmd += ["--librosa-hop-length", str(args.librosa_hop_length)]
    if _PROGRESS_JSON:                        cmd += ["--progress-json"]
    run_with_progress(
        cmd, "STEP 2 / 3  —  Chord Chart",
        "chord", *ranges["chord"],
    )

    # allin1 should already be done (chord chart takes longer), but clean up
    # just in case the song was very short or sections-json-wait-s expired.
    if allin1_proc is not None and allin1_proc.poll() is None:
        print("[pipeline] Waiting for allin1 to finish …")
        sys.stdout.flush()
        allin1_proc.wait()
    if allin1_proc is not None:
        _emit_global("sections", ranges["chord"][1], "section detection complete")

    # ── Step 3 / 3  —  Stem splitting (skipped if already ran early) ───────
    if args.skip_stems:
        print("\n[pipeline] Skipping stem splitting.")
    elif stems_done:
        print("\n[pipeline] Stem splitting already completed in early pass.")
    else:
        run_with_progress(
            _build_stems_cmd(args.stems),
            "STEP 3 / 3  —  Stem Splitting",
            "stems", *ranges["stems"],
        )

    # ── Summary ──────────────────────────────────────────────
    print(f"\n{'='*54}")
    print(f"  ✓  Pipeline complete!")
    if not args.skip_stabilize:
        print(f"     Stabilised audio : {stabilised}")
    print(f"     Chord chart PDF  : {chart_out}.pdf")
    print(f"     Analysis JSON    : {chart_out}.json")
    if not args.skip_stems:
        print(f"     Stems            : {stems_out}/")
    if not args.skip_stems and args.session_type:
        print(f"     Backing track    : {backing_track_path}")
    print(f"{'='*54}\n")
    _emit_global("done", 100.0, "pipeline complete")


if __name__ == "__main__":
    main()
