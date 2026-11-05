#!/usr/bin/env python3
"""
beat_detect.py — Beat + downbeat detection with Beat This! (CPJKU, MIT code + weights).

Runs in venv_demucs (needs torch + beat_this). Called by beat_stabilizer.py as a
subprocess; prints one JSON object on stdout:

    {
      "beats":            [t0, t1, ...],         # seconds
      "downbeat_indices": [i0, i1, ...],         # indices into beats
      "meter":            4,                     # fitted beats per bar: 4 (any duple) or 3, or null
      "beat_act":         [p0, p1, ...],         # beat probability (0..1) at each beat
      "downbeat_act":     [q0, q1, ...],         # downbeat probability at each beat
      "meter_fit":        {"2": {score, phase, hit_rate}, "3": ..., "4": ...},
      "meter_margin":     0.31,                  # best minus runner-up meter score
      "raw_bar_counts":   {...},                 # beats between Beat This!'s raw downbeats
      "phase":            {method, prob_margin, energy_share, ...},  # how beat 1 was chosen
      "octave_fix":       {inserted, dropped, period_s, raw_beats},  # half/double-speed sections regridded
    }

Beats come from Beat This!'s "minimal" peak picker (no madmom DBN, so no
madmom code or models). Meter is fitted here: one constant beats-per-bar +
phase for the whole song, chosen by downbeat-probability contrast.

    ./venv_demucs/bin/python beat_detect.py song.wav [--device auto|cpu|mps|cuda]
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import soundfile as sf
import torch


def pick_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def fix_octaves(beats: np.ndarray, tol: float = 0.2) -> tuple[np.ndarray, dict]:
    """Put every section on the song's main beat grid.

    Beat This! can lock to half speed in sparse parts (a drumless piano intro
    at 73 BPM in a 143 BPM song) or double speed in busy ones. Warping those
    beats onto one grid would play that part 2x fast (or slow). The main
    period P is the median inter-beat interval (the majority grid). Gaps of
    ~k*P (k = 2..16) get k-1 evenly spaced beats; beats closer than 0.7*P to
    the previous kept beat are dropped. Anything else is left untouched.
    """
    info = {"inserted": 0, "dropped": 0, "period_s": None}
    if len(beats) < 8:
        return beats, info
    P = float(np.median(np.diff(beats)))
    info["period_s"] = round(P, 4)
    kept = [float(beats[0])]
    for t in beats[1:]:
        if t - kept[-1] < 0.7 * P:
            info["dropped"] += 1
            continue
        kept.append(float(t))
    out = [kept[0]]
    for t in kept[1:]:
        gap = t - out[-1]
        k = int(round(gap / P))
        if 2 <= k <= 16 and abs(gap / P - k) < tol * k ** 0.5:
            base = out[-1]
            out.extend([base + gap * j / k for j in range(1, k)])
            info["inserted"] += k - 1
        out.append(t)
    return np.asarray(out), info


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("audio")
    p.add_argument("--device", default="auto")
    p.add_argument("--checkpoint", default="final0")
    a = p.parse_args()

    from beat_this.inference import Audio2Frames
    from beat_this.model.postprocessor import Postprocessor

    signal, sr = sf.read(a.audio, dtype="float32", always_2d=False)
    device = pick_device(a.device)
    try:
        a2f = Audio2Frames(checkpoint_path=a.checkpoint, device=device)
        beat_logits, down_logits = a2f(signal, sr)
    except Exception:
        if device == "cpu":
            raise
        a2f = Audio2Frames(checkpoint_path=a.checkpoint, device="cpu")
        beat_logits, down_logits = a2f(signal, sr)

    beats, downbeats = Postprocessor(type="minimal")(beat_logits, down_logits)
    beats = np.asarray(beats, dtype=float)
    downbeats = np.asarray(downbeats, dtype=float)
    raw_n = len(beats)
    beats, octave_info = fix_octaves(beats)
    octave_info["raw_beats"] = raw_n

    fps = 50.0  # Beat This! frame rate
    bp = torch.sigmoid(beat_logits.float()).cpu().numpy()
    dp = torch.sigmoid(down_logits.float()).cpu().numpy()

    def act_at(prob: np.ndarray, times: np.ndarray) -> list[float]:
        out = []
        for t in times:
            i = int(round(t * fps))
            out.append(float(prob[max(0, i - 1): i + 2].max()) if len(prob) else 0.0)
        return out

    # ── Meter + bar phase: fit ONE constant meter for the whole song ──────
    # Beat This!'s minimal peak picker has no meter model, so its raw
    # downbeats wander (e.g. every 2 beats in a 4/4 song). Instead, read the
    # downbeat probability at every beat and pick the (beats_per_bar, phase)
    # whose bar-1 positions stand out most from the other beats.
    db_at_beat = np.asarray(act_at(dp, beats))

    def phase_scores(bpb: int) -> list[float]:
        out = []
        for ph in range(bpb):
            on = db_at_beat[ph::bpb]
            off = np.delete(db_at_beat, np.arange(ph, len(db_at_beat), bpb))
            out.append(float(on.mean() - off.mean()) if len(on) >= 4 and len(off) >= 4 else -1.0)
        return out

    fits = {}
    for bpb in (2, 3, 4):
        sc = phase_scores(bpb)
        ph = int(np.argmax(sc))
        if sc[ph] > -1.0:
            fits[bpb] = {"score": sc[ph], "phase": ph,
                         "hit_rate": float(np.mean(db_at_beat[ph::bpb] > 0.5)),
                         "phase_scores": [round(x, 4) for x in sc]}

    # Meter: duple vs triple only. Duple is always reported as 4/4 — real 2/4
    # is rare in AI pop, a DAW defaults to 4/4, and calling it 2/4 halves the
    # lead-in. (bpb=2 is still scored so the duple/triple test sees it.)
    meter, phase, phase_info = None, 0, {}
    if 4 in fits or 3 in fits:
        duple = max(fits.get(2, {}).get("score", -1), fits.get(4, {}).get("score", -1))
        meter = 3 if fits.get(3, {}).get("score", -1) > duple else 4

        # Beat-1 position within the bar. Primary: downbeat probability per
        # phase. If the two best phases are close (typically beat 1 vs beat 3
        # in 4/4), break the tie with energy: AI songs bring in new parts on a
        # downbeat, so the phase that collects the biggest beat-to-beat energy
        # rises is beat 1.
        sc = phase_scores(meter)
        order = np.argsort(sc)[::-1]
        p1, p2 = int(order[0]), int(order[1])
        prob_margin = sc[p1] - sc[p2]

        mono = signal.mean(1) if signal.ndim == 2 else signal
        idx = np.clip((beats * sr).astype(int), 0, len(mono))
        rms = np.array([np.sqrt(np.mean(mono[i:j] ** 2)) if j > i else 0.0
                        for i, j in zip(idx[:-1], idx[1:])] + [0.0])
        lr = np.log(rms + 1e-6)
        rise = np.maximum(0.0, np.diff(lr, prepend=lr[0]))
        # Only the strongest rises (top 10 %) count: entrances, not groove.
        thr = np.quantile(rise, 0.9) if len(rise) else 0.0
        big = np.where(rise >= thr, rise, 0.0)
        e_share = [float(big[ph::meter].sum()) for ph in range(meter)]
        tot = sum(e_share) or 1.0
        e_share = [x / tot for x in e_share]
        e_margin = e_share[p1] - e_share[p2]

        if prob_margin >= 0.15:
            phase, method = p1, "downbeat_prob"
        elif abs(e_margin) >= 0.15:
            phase, method = (p1 if e_margin > 0 else p2), "energy_tiebreak"
        else:
            phase, method = p1, "ambiguous"
        phase_info = {"method": method, "prob_margin": round(prob_margin, 4),
                      "energy_share": [round(x, 3) for x in e_share],
                      "energy_margin": round(e_margin, 4), "candidates": [p1, p2],
                      "chosen": phase}
    dbi = list(range(phase, len(beats), meter)) if meter else []
    ranked = sorted((f["score"] for f in fits.values()), reverse=True)
    raw_counts = {}
    if len(downbeats) >= 2 and len(beats):
        idx = [int(np.argmin(np.abs(beats - t))) for t in downbeats]
        vals, freq = np.unique(np.diff(idx), return_counts=True)
        raw_counts = {int(v): int(f) for v, f in zip(vals, freq)}

    json.dump({
        "beats": beats.tolist(),
        "downbeat_indices": dbi,
        "meter": meter,
        "beat_act": act_at(bp, beats),
        "downbeat_act": [float(x) for x in db_at_beat],
        "meter_fit": {str(k): {kk: round(vv, 4) if isinstance(vv, float) else vv
                               for kk, vv in v.items()} for k, v in fits.items()},
        "meter_margin": round(ranked[0] - ranked[1], 4) if len(ranked) > 1 else None,
        "phase": phase_info,
        "raw_bar_counts": raw_counts,
        "octave_fix": octave_info,
        "device": device,
    }, sys.stdout)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
