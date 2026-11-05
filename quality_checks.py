"""
quality_checks.py — Pass/fail gates for the beat stabilizer.

Each check returns a dict: {"ok": bool, "reason": str, ...metrics}. The
stabilizer collects them into the analysis JSON and aborts (EARLY_STOP,
exit 2) on the first failure unless --skip-checks is set.

Checks
------
beat_clarity  Is there a trackable pulse at all?
              Beat This! beat probability at each tracked beat (0..1) + the
              regularity of the raw inter-beat intervals.
meter         Is the beats-per-bar estimate trustworthy?
              Downbeat-probability contrast of the fitted meter, and the
              separation between duple (2/4) and triple (3) fits, and an
              unambiguous beat-1 position (downbeat prob, energy tie-break). librosa's
              autocorrelation meter is recorded for comparison only.
tempo_jumps   Short glitches in the beat grid: inter-beat intervals that
              deviate >jump_pct from the local (rolling) median. Sustained
              tempo changes are handled separately by detect_tempo_change().
length        Warped body length vs original. A wrong BPM (e.g. user-forced,
              or a half/double-time mistake) stretches the song.
"""

from __future__ import annotations

import numpy as np

# Defaults — tune on a labelled set of good / bad songs.
DEFAULTS = {
    "min_beat_act":        0.90,   # median Beat This! beat probability at beats
    "min_strong_frac":     0.70,   # share of beats with probability >= 0.9
    "max_ibi_cv":          None,   # recorded only (was 0.12): the warp corrects raw spacing
    "min_meter_score":     0.15,   # downbeat-prob contrast of the winning meter
    "min_meter_sep":       0.10,   # |duple score - triple score|
    "jump_pct":            0.15,   # IBI deviation from local median that counts as a jump
    "max_jump_frac":       None,   # recorded only (was 0.03): did not match listening verdicts
    "max_length_dev":      0.05,   # |warped/original - 1|
}


def _ok(ok: bool, reason: str = "", **metrics) -> dict:
    return {"ok": bool(ok), "reason": reason, **{k: _r(v) for k, v in metrics.items()}}


def _r(v):
    if isinstance(v, (float, np.floating)):
        return round(float(v), 4)
    if isinstance(v, np.integer):
        return int(v)
    return v


def check_beat_clarity(beat_times: np.ndarray, beat_act: list[float] | None,
                       t: dict = DEFAULTS) -> dict:
    if len(beat_times) < 16:
        return _ok(False, f"only {len(beat_times)} beats detected", n_beats=len(beat_times))
    ibi = np.diff(beat_times)
    med = float(np.median(ibi))
    # Fold half/double intervals onto the median so octave flips don't count as irregularity.
    folded = ibi.copy()
    folded[ibi < med * 0.7] *= 2
    folded[ibi > med * 1.4] /= 2
    cv = float(np.std(folded) / np.mean(folded))
    m = {"n_beats": len(beat_times), "ibi_cv": cv}
    if beat_act:
        act = np.asarray(beat_act, dtype=float)
        m["median_beat_act"] = float(np.median(act))
        m["strong_frac"] = float(np.mean(act >= 0.9))
        if m["median_beat_act"] < t["min_beat_act"]:
            return _ok(False, f"weak pulse: median beat activation {m['median_beat_act']:.2f} "
                              f"< {t['min_beat_act']}", **m)
        if m["strong_frac"] < t["min_strong_frac"]:
            return _ok(False, f"pulse drops out: only {m['strong_frac']:.0%} of beats are "
                              f"clearly detected (< {t['min_strong_frac']:.0%})", **m)
    # CV is recorded only: raw spacing is what the warp corrects (20a: CV 0.41, usable).
    return _ok(True, **m)


def check_meter(meter: int | None, meter_fit: dict | None, phase: dict | None,
                librosa_meter: int | None, t: dict = DEFAULTS) -> dict:
    """meter_fit = {"2"|"3"|"4": {"score", "phase", "hit_rate"}} from beat_detect.py.

    score = mean downbeat probability on the chosen bar-1 beats minus the
    mean on all other beats. Two things must hold:
      - the winning score is high enough (downbeats are audible at all);
      - duple (2 or 4) vs triple (3) is clearly separated. 2 vs 4 is NOT
        tested: for a lead-in they only differ in bar length, and the
        detector prefers 4 unless 2 clearly wins.
    """
    if not meter or not meter_fit:
        return _ok(False, "downbeats unavailable — meter unknown",
                   meter=meter, librosa_meter=librosa_meter)
    sc = {int(k): v["score"] for k, v in meter_fit.items()}
    duple = max(sc.get(2, -1), sc.get(4, -1))
    triple = sc.get(3, -1)
    best = max(duple, triple)
    sep = abs(duple - triple)
    m = {
        "meter": int(meter),
        "scores": {k: round(v, 4) for k, v in sorted(sc.items())},
        "best_score": best,
        "duple_triple_sep": sep,
        "hit_rate": meter_fit[str(meter)]["hit_rate"],
        "phase": phase,
        "librosa_meter": librosa_meter,
        "agree_with_librosa": (None if not librosa_meter else
                               (librosa_meter in (2, 4)) == (meter in (2, 4))),
    }
    if best < t["min_meter_score"]:
        return _ok(False, f"downbeats unclear: best meter score {best:.2f} "
                          f"< {t['min_meter_score']}", **m)
    if sep < t["min_meter_sep"]:
        return _ok(False, f"ambiguous meter: duple vs triple scores differ by {sep:.2f} "
                          f"(< {t['min_meter_sep']})", **m)
    if phase and phase.get("method") == "ambiguous":
        # Not a failure: listening tests (08a) showed the best-scoring guess is
        # usable. Recorded so it can be reviewed.
        m["warning"] = (f"beat 1 unclear: picked beat {phase['candidates'][0] + 1}, runner-up "
                        f"{phase['candidates'][1] + 1} (downbeat margin {phase['prob_margin']:.2f}, "
                        f"energy margin {phase['energy_margin']:.2f})")
    return _ok(True, **m)


def check_tempo_jumps(beat_times: np.ndarray, t: dict = DEFAULTS, window: int = 9) -> dict:
    ibi = np.diff(beat_times)
    if len(ibi) < window:
        return _ok(True, n_jumps=0)
    half = window // 2
    pad = np.pad(ibi, half, mode="edge")
    local_med = np.array([np.median(pad[i:i + window]) for i in range(len(ibi))])
    dev = np.abs(ibi / local_med - 1.0)
    jumps = np.nonzero(dev > t["jump_pct"])[0]
    frac = len(jumps) / len(ibi)
    m = {"n_jumps": len(jumps), "jump_frac": frac,
         "worst_dev": float(dev.max()),
         "jump_times_s": [round(float(beat_times[i + 1]), 2) for i in jumps[:20]]}
    # Recorded only: neither the count nor the size of jumps matched listening
    # verdicts (strongest jump was a usable song with a break).
    return _ok(True, **m)


def check_length(n_in: int, n_warped: int, t: dict = DEFAULTS) -> dict:
    ratio = n_warped / max(1, n_in)
    m = {"length_ratio": ratio}
    if abs(ratio - 1.0) > t["max_length_dev"]:
        return _ok(False, f"stabilised audio is {ratio:.2f}× the original length "
                          f"(> ±{t['max_length_dev']:.0%}) — BPM is likely wrong "
                          f"(half/double time?)", **m)
    return _ok(True, **m)


def librosa_meter(y_mono: np.ndarray, sr: int, beat_times: np.ndarray) -> int | None:
    """Autocorrelation meter estimate (2/3/4, 6 = 6/8) for comparison with Beat This!."""
    try:
        from quick_analyze import detect_time_signature
        return int(detect_time_signature(y_mono, sr, beat_times))
    except Exception:
        return None
