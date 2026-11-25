#!/usr/bin/env python3
"""
make_review.py — Build a static HTML page to listen to and rate separated stems.

Scans bench/out/<song>/ as written by bench/run_models.sh:
    <song>/times.txt                              "<model> rc=0  12.3s" per line
    <song>/demucs_<model>/<model>/<track>/<stem>.wav
    <song>/sep_<model>/<track>_(<Stem>)_<model>.wav

Writes bench/review.html (open it directly in a browser, no server needed).
Ratings are kept in the browser (localStorage) and exported as JSON.

    python3 bench/make_review.py [--out-root bench/out] [--html bench/review.html]
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
from pathlib import Path

STEM_ORDER = ["vocals", "drums", "bass", "other", "instrumental", "guitar", "piano"]


def stem_name(path: Path) -> str:
    m = re.search(r"_\(([^)]+)\)_", path.name)
    raw = (m.group(1) if m else path.stem).lower()
    return {"no vocals": "instrumental", "no_vocals": "instrumental"}.get(raw, raw)


def analysis_line(a: dict | None) -> str:
    if not a:
        return ""
    bad = [f"{k}: {v['reason']}" for k, v in a.get("checks", {}).items() if not v.get("ok")]
    status = ("<b class=fail>would fail — " + html.escape("; ".join(bad)) + "</b>") if bad \
        else "<b class=pass>all checks pass</b>"
    return (f'<p class="meta">{a.get("bpm")} BPM (detected {a.get("detected_bpm")}) · '
            f'{html.escape(str(a.get("meter")))} · lead-in {a.get("lead_in_bars")} bars · {status}</p>')


def scan(out_root: Path, with_stabilised: bool = False) -> list[dict]:
    songs = []
    for song_dir in sorted(p for p in out_root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        times = {}
        tf = song_dir / "times.txt"
        if tf.exists():
            for line in tf.read_text().splitlines():
                parts = line.split()
                if len(parts) >= 3:
                    times[parts[0]] = (parts[1], parts[-1])
        models = []
        for mdir in sorted(p for p in song_dir.iterdir() if p.is_dir()):
            if mdir.name == "stabilised" and not with_stabilised:
                continue
            wavs = sorted(list(mdir.rglob("*.wav")) + list(mdir.rglob("*.m4a")))
            if not wavs:
                continue
            stems = {}
            for w in wavs:
                n = stem_name(w)
                # kuielab "no <stem>" outputs are residuals; skip them.
                if n.startswith("no "):
                    continue
                stems[n] = w
            ordered = sorted(stems.items(),
                             key=lambda kv: STEM_ORDER.index(kv[0]) if kv[0] in STEM_ORDER else 99)
            rc, secs = times.get(mdir.name, ("", ""))
            models.append({"id": mdir.name, "time": secs, "rc": rc, "stems": ordered})
        # Stabilised track first, then separators.
        models.sort(key=lambda m: (m["id"] != "stabilised", m["id"]))
        original = next((p for p in [song_dir / "original.wav"] if p.exists()), None)
        analysis = None
        af = song_dir / "analysis.json"
        if af.exists():
            analysis = json.loads(af.read_text())
        if not models:
            continue
        songs.append({"id": song_dir.name, "original": original, "models": models,
                      "analysis": analysis if with_stabilised else None})
    return songs


def render(songs: list[dict], html_path: Path) -> str:
    base = html_path.parent.resolve()

    def rel(p: Path) -> str:
        return html.escape(os.path.relpath(p.resolve(), base))

    parts = []
    for s in songs:
        orig = (f'<div class="row orig"><span class="stem">original mix</span>'
                f'<audio controls preload="none" src="{rel(s["original"])}"></audio></div>'
                if s["original"] else "")
        mparts = []
        for m in s["models"]:
            rows = []
            for name, path in m["stems"]:
                key = f'{s["id"]}|{m["id"]}|{name}'
                rows.append(
                    f'<div class="row" data-key="{html.escape(key)}">'
                    f'<span class="stem">{html.escape(name)}</span>'
                    f'<audio controls preload="none" src="{rel(path)}"></audio>'
                    f'<span class="rate">'
                    f'<button data-v="usable">usable</button>'
                    f'<button data-v="borderline">borderline</button>'
                    f'<button data-v="unusable">unusable</button></span>'
                    f'<input class="note" placeholder="note"></div>')
            t = f'{m["time"]}' + ("" if m["rc"] in ("", "rc=0") else f" ({m['rc']})")
            mparts.append(f'<section class="model"><h3>{html.escape(m["id"])}'
                          f'<span class="time">{html.escape(t)}</span></h3>{"".join(rows)}</section>')
        parts.append(f'<article><h2>{html.escape(s["id"])}</h2>{analysis_line(s["analysis"])}'
                     f'{orig}{"".join(mparts)}</article>')

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Stem Review</title>
<style>
:root {{ --bg:#fafaf9; --fg:#1c1917; --mute:#78716c; --line:#e7e5e4; --card:#fff;
        --ok:#15803d; --mid:#b45309; --bad:#b91c1c; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg:#1c1917; --fg:#f5f5f4; --mute:#a8a29e; --line:#44403c; --card:#292524;
          --ok:#4ade80; --mid:#fbbf24; --bad:#f87171; }} }}
body {{ background:var(--bg); color:var(--fg); font:14px/1.4 -apple-system,system-ui,sans-serif;
       margin:0 auto; max-width:1100px; padding:16px; }}
header {{ display:flex; gap:12px; align-items:center; flex-wrap:wrap; margin-bottom:8px; }}
header p {{ color:var(--mute); margin:0; flex:1; min-width:260px; }}
article {{ background:var(--card); border:1px solid var(--line); border-radius:10px;
          padding:12px 16px; margin:16px 0; }}
h2 {{ margin:4px 0 8px; font-size:18px; }}
h3 {{ margin:14px 0 6px; font-size:14px; display:flex; gap:10px; align-items:baseline; }}
.time {{ color:var(--mute); font-weight:400; font-variant-numeric:tabular-nums; }}
.row {{ display:flex; gap:10px; align-items:center; padding:4px 0; flex-wrap:wrap;
       border-top:1px solid var(--line); }}
.stem {{ width:110px; font-weight:600; }}
audio {{ height:32px; flex:1; min-width:220px; }}
.rate button {{ border:1px solid var(--line); background:transparent; color:var(--fg);
               border-radius:6px; padding:3px 8px; cursor:pointer; font-size:12px; }}
.rate button.on[data-v=usable] {{ background:var(--ok); color:#fff; border-color:var(--ok); }}
.rate button.on[data-v=borderline] {{ background:var(--mid); color:#fff; border-color:var(--mid); }}
.rate button.on[data-v=unusable] {{ background:var(--bad); color:#fff; border-color:var(--bad); }}
.note {{ flex:0 1 200px; min-width:120px; background:transparent; color:var(--fg);
        border:1px solid var(--line); border-radius:6px; padding:3px 6px; }}
#summary {{ font-variant-numeric:tabular-nums; }}
.meta {{ color:var(--mute); margin:0 0 6px; }}
.pass {{ color:var(--ok); }} .fail {{ color:var(--bad); }}
#summary table {{ border-collapse:collapse; margin:8px 0; }}
#summary td, #summary th {{ border:1px solid var(--line); padding:3px 8px; text-align:right; }}
#summary th:first-child, #summary td:first-child {{ text-align:left; }}
.hdrbtn {{ border:1px solid var(--line); background:var(--card); color:var(--fg);
          border-radius:6px; padding:6px 10px; cursor:pointer; }}
</style></head><body>
<header><h1 style="margin:0;font-size:20px">Stem review</h1>
<p>Per stem: can a musician record with it? (Usable for recording, not pristine.) Playing any
track pauses the others and continues from the same position, so you can A/B between models.</p>
<button class="hdrbtn" id="export">Export ratings JSON</button></header>
<div id="summary"></div>
{"".join(parts)}
<script>
const KEY = "stem-review-v1";
let st = {{}};
try {{ st = JSON.parse(localStorage.getItem(KEY) || "{{}}"); }} catch (e) {{}}
const save = () => {{ try {{ localStorage.setItem(KEY, JSON.stringify(st)); }} catch (e) {{}} summary(); }};

let pos = 0;
document.querySelectorAll("audio").forEach(a => {{
  a.addEventListener("play", () => {{
    document.querySelectorAll("audio").forEach(o => {{ if (o !== a && !o.paused) {{ pos = o.currentTime; o.pause(); }} }});
    if (pos && Math.abs(a.currentTime - pos) > 0.3) a.currentTime = pos;
  }});
  a.addEventListener("timeupdate", () => {{ if (!a.paused) pos = a.currentTime; }});
  a.addEventListener("seeked", () => {{ pos = a.currentTime; }});
}});

document.querySelectorAll(".row[data-key]").forEach(r => {{
  const k = r.dataset.key, cur = st[k] || {{}};
  r.querySelectorAll("button").forEach(b => {{
    if (cur.rating === b.dataset.v) b.classList.add("on");
    b.onclick = () => {{
      const on = b.classList.contains("on");
      r.querySelectorAll("button").forEach(x => x.classList.remove("on"));
      st[k] = {{ ...(st[k] || {{}}), rating: on ? null : b.dataset.v }};
      if (!on) b.classList.add("on");
      save();
    }};
  }});
  const n = r.querySelector(".note");
  n.value = cur.note || "";
  n.oninput = () => {{ st[k] = {{ ...(st[k] || {{}}), note: n.value }}; save(); }};
}});

function summary() {{
  const agg = {{}};
  for (const [k, v] of Object.entries(st)) {{
    if (!v.rating) continue;
    const model = k.split("|")[1];
    agg[model] ??= {{ usable: 0, borderline: 0, unusable: 0 }};
    agg[model][v.rating]++;
  }}
  const rows = Object.entries(agg).map(([m, c]) =>
    `<tr><td>${{m}}</td><td>${{c.usable}}</td><td>${{c.borderline}}</td><td>${{c.unusable}}</td></tr>`).join("");
  document.getElementById("summary").innerHTML = rows
    ? `<table><tr><th>model</th><th>usable</th><th>borderline</th><th>unusable</th></tr>${{rows}}</table>` : "";
}}
summary();

document.getElementById("export").onclick = () => {{
  const blob = new Blob([JSON.stringify(st, null, 2)], {{ type: "application/json" }});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = "stem-ratings.json"; a.click();
}};
</script></body></html>
"""


def main() -> None:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", default=str(here / "out"))
    p.add_argument("--html", default=str(here / "review.html"))
    p.add_argument("--with-stabilised", action="store_true",
                   help="Also list stabilised tracks + analysis (default: stems only)")
    a = p.parse_args()
    html_path = Path(a.html)
    songs = scan(Path(a.out_root), a.with_stabilised)
    html_path.write_text(render(songs, html_path))
    n = sum(len(m["stems"]) for s in songs for m in s["models"])
    print(f"{html_path}  ({len(songs)} songs, {n} stems)")


if __name__ == "__main__":
    main()
