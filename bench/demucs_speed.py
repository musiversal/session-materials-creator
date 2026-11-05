#!/usr/bin/env python3
"""
demucs_speed.py — Time htdemucs inference under different settings (in-process,
model loaded once, so only separation time is measured).

    ./venv_demucs/bin/python bench/demucs_speed.py song.wav [--out bench/out/<song>/speed]

Writes each config's stems as m4a under --out/<config>/ for listening.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from demucs.apply import apply_model
from demucs.pretrained import get_model

CONFIGS = [
    # name,                  device, overlap, segment, shifts, fp16
    ("cpu_default",          "cpu",  0.25,    None,    1,      False),
    ("mps_default",          "mps",  0.25,    None,    1,      False),
    ("mps_overlap0.1",       "mps",  0.10,    None,    1,      False),
    ("mps_overlap0",         "mps",  0.0,     None,    1,      False),
    ("mps_shifts0",          "mps",  0.25,    None,    0,      False),
    ("mps_ov0.1_shifts0",    "mps",  0.10,    None,    0,      False),
    ("mps_ov0.1_fp16",       "mps",  0.10,    None,    0,      True),
]


def sync(dev: str) -> None:
    if dev == "mps":
        torch.mps.synchronize()
    elif dev == "cuda":
        torch.cuda.synchronize()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("audio")
    p.add_argument("--out", default=None)
    p.add_argument("--only", default=None, help="comma-separated config names")
    a = p.parse_args()

    wav, sr = sf.read(a.audio, dtype="float32", always_2d=True)
    assert sr == 44100, "resample to 44.1 kHz first"
    dur = len(wav) / sr
    x = torch.from_numpy(wav.T.copy())
    ref = x.mean(0)
    x = (x - ref.mean()) / ref.std()

    base = get_model("htdemucs")
    out_root = Path(a.out) if a.out else None
    results = []
    only = set(a.only.split(",")) if a.only else None
    for name, dev, ov, seg, shifts, fp16 in CONFIGS:
        if only and name not in only:
            continue
        if dev == "mps" and not torch.backends.mps.is_available():
            continue
        model = get_model("htdemucs").to(dev)
        if fp16:
            model = model.half()
        model.eval()
        kw = dict(device=dev, shifts=shifts, overlap=ov, split=True, progress=False)
        if seg:
            kw["segment"] = seg
        inp = x[None].to(dev)
        if fp16:
            inp = inp.half()
        # warm-up on 10 s
        with torch.no_grad():
            apply_model(model, inp[..., : sr * 10], **kw)
        sync(dev)
        t0 = time.perf_counter()
        try:
            with torch.no_grad():
                out = apply_model(model, inp, **kw)
            sync(dev)
            err = None
        except Exception as e:  # noqa: BLE001
            out, err = None, repr(e)[:200]
        dt = time.perf_counter() - t0
        row = {"config": name, "seconds": round(dt, 1), "x_realtime": round(dur / dt, 2),
               "error": err}
        print(json.dumps(row), flush=True)
        results.append(row)
        if out is not None and out_root:
            d = out_root / name
            d.mkdir(parents=True, exist_ok=True)
            out = out[0].float().cpu() * ref.std() + ref.mean()
            for i, stem in enumerate(base.sources):
                tmp = d / f"{stem}.wav"
                sf.write(tmp, out[i].numpy().T, sr)
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(tmp), "-c:a", "aac",
                                "-b:a", "160k", str(d / f"{stem}.m4a")], check=True)
                tmp.unlink()
        del model
        if dev == "mps":
            torch.mps.empty_cache()
    if out_root:
        (out_root / "times.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
