# Session Materials Creator

Drop in a song, get back:

1. **BPM + meter** — [Beat This!](https://github.com/CPJKU/beat_this) beat + downbeat
   detection (MIT code and weights; no fallback: a detector crash fails the song), one constant meter fitted per song.
2. **Stabilized track** — audio warped onto a fixed BPM grid (Rubber Band CLI
   `--timemap`, GPL — fine server-side, not linked or distributed), with a
   **4-bar lead-in** before the first downbeat. Place it at bar 1 in a DAW set to
   the detected BPM and the song's first downbeat lands on bar 5.
3. **Stems** — Demucs separation, plus an optional backing track (all stems
   minus the session instrument).

CLI only: it either produces the files or exits with code 2 and an
`EARLY_STOP {...}` JSON line naming the failed check.

---

## Requirements

- macOS (Apple Silicon or Intel) or Linux
- Homebrew (macOS) — installs `python@3.11`, `rubberband`, `ffmpeg`
- System `python3` (beat stabilizer + quick analysis)

## Setup

```bash
bash setup.sh                 # brew deps, system-python deps, venv_demucs (+ Beat This!)
```

`venv_demucs` holds PyTorch for both Demucs and Beat This!. The stabilizer itself
runs on system Python and calls Beat This! as a subprocess.

---

## CLI

```bash
python3 pipeline.py -i song.wav
python3 pipeline.py -i song.wav --bpm 84                 # force BPM
python3 pipeline.py -i song.wav --beats-per-bar 3        # force meter
python3 pipeline.py -i song.wav --lead-in-bars 2         # default 4
python3 pipeline.py -i song.wav --stem-model htdemucs_6s   # 6 stems
python3 pipeline.py -i song.wav --format wav               # lossless outputs (default m4a)
python3 pipeline.py -i song.wav --stems vocals,drums
python3 pipeline.py -i song.wav --session-type bass      # + backing track minus bass
python3 pipeline.py -i song.wav --skip-stems
python3 pipeline.py -i song.wav --allow-tempo-change     # warp multi-tempo songs anyway
python3 pipeline.py -i song.wav --skip-checks            # record checks, never abort
```

Outputs in `<output-dir>/<song>/` (default `~/Desktop/audio-tools-tests/`):

| File | Content |
|---|---|
| `<song>_stabilised.m4a` | Beat-locked audio with lead-in |
| `<song>.bpm` | BPM sidecar |
| `<song>_analysis.json` | `ok`, `bpm`, `detected_bpm`, `meter`, `beats_per_bar`, `lead_in_bars`, `first_downbeat_s`, `duration_s`, `checks` |
| `<song>_stems/<stem>.m4a` | One file per stem |
| `<song>_stems/stems_info.json` | Per-stem `{present, rms_dbfs_peak, loud_seconds}` |
| `<song>_backing_track.m4a` | Only with `--session-type` |

Any input ffmpeg can decode (wav, mp3, m4a/aac, flac, ogg, aiff, any sample
rate / bit depth / channel count) is first normalised to 44.1 kHz stereo WAV.
Outputs are AAC 192 kb/s m4a unless `--format wav`.

### Quality checks

Defined in `quality_checks.py` (thresholds in `DEFAULTS`), tuned on 33 user AI tracks labelled by ear. The first failure
aborts with exit 2; all results are written to `checks` in the analysis JSON.

| Check | Fails when |
|---|---|
| `detector_failed` | Beat This! crashed, timed out (`--detector-timeout-s`, 300 s) or is not installed. No librosa fallback unless `--detector-backend auto` |
| `beat_clarity` | median Beat This! beat probability < 0.90, or < 70 % of beats above 0.9. Raw beat-spacing CV is recorded only |
| `meter` | winning meter's downbeat contrast < 0.15, or duple-vs-triple scores within 0.10. An unclear beat 1 is only a `warning` (best guess used). librosa's meter is recorded as `librosa_meter` for comparison only |
| `tempo_jumps` | never fails; jump count and size are recorded only (neither matched listening verdicts) |
| `tempo_change` | sustained BPM step (≥ 6 BPM and ≥ 10 %, 4+ bars) — splits the song into two tempos |
| `length` | warped audio is more than ±5 % longer/shorter than the original → wrong BPM |

### Lead-in

Anchored on the first detected **downbeat** (not the first beat), so bar lines
line up. Audio before the first downbeat is never trimmed: if the intro is
longer than 4 bars, the lead-in grows to the next whole bar (reported as
`lead_in_bars` / `first_downbeat_s`). `--no-trim-intro` disables it.

### Meter

Only duple vs triple is decided: any duple meter is reported as **4/4** (never
2/4), triple as 3/4. Odd meters fail the `meter` check. Beat 1 is placed by
Beat This!'s downbeat probability; when beat 1 vs beat 3 is close, the beat
that collects the biggest energy rises (parts entering) wins; if both are
unclear, the best guess is kept and `checks.meter.warning` says so.

### Tempo-change guard

The stabilizer assumes one tempo. A sustained BPM step (≥ 6 BPM and ≥ 10 %,
persisting 4+ bars) aborts with `EARLY_STOP {"reason":"tempo_change", ...}`.
Override with `--allow-tempo-change`.

### Stem models

| `--stem-model` | Stems |
|---|---|
| `htdemucs` (default) | vocals, drums, bass, other — fastest |
| `htdemucs_6s` | vocals, drums, bass, guitar, piano, other |
| `htdemucs_ft` | 4 stems, fine-tuned — best quality, ~4× slower |
| `mdx_extra` | 4 stems, alternative architecture |

### Individual tools

```bash
python3 beat_stabilizer.py -i song.wav --detect-only                       # BPM + meter only
python3 beat_stabilizer.py -i song.wav -o out.wav --analysis-out out.json
./venv_demucs/bin/python stem_splitter.py -i out.wav --model htdemucs
```

---

## Separation benchmark

```bash
bench/run_models.sh ~/ai-songs/*.mp3     # htdemucs (or DEMUCS_MODELS="..."), timed
python3 bench/make_review.py && open bench/review.html
```

`review.html` lists each model's stems (playing one pauses the others at the
same position, for A/B), with usable / borderline / unusable ratings and a
JSON export.

`bench/make_label_page.py <dir>` builds a page to label test songs by ear
(is the scenario's feature there? 3/4 or 4/4?). The checks were tuned on 33
user AI tracks labelled this way plus the 10 Song Clinic songs.

## Tests

```bash
python3 tests/smoke_test.py      # ~1 min: synthetic 4/4, 3/4 and noise through the stabilizer + pipeline
```
