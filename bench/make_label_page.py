#!/usr/bin/env python3
"""make_label_page.py — Build a page to label test candidates by ear.

One question per file: is the scenario's feature there? For scenarios that
should not be 4/4, also: 3/4 or 4/4? Ratings stay in localStorage; export JSON.

    python3 bench/make_label_page.py "<candidates dir>"
Writes <candidates dir>/label.html (plays _preview/*.m4a).
"""
from __future__ import annotations

import html
import re
import sys
from pathlib import Path

FEATURE = {
    1: "Beatless intro (~30 s, no drums) before the band enters",
    2: "Long pad intro with no kick before the drop",
    3: "Waltz feel (3 beats per bar)",
    4: "Waltz feel (3 beats per bar)",
    5: "6/8 feel (two groups of three)",
    6: "6/8 feel, fast jig-like",
    7: "Swing feel",
    8: "Shuffle / 12/8 feel",
    9: "Half-time trap drums",
    10: "Drum and bass",
    11: "Reggae one-drop (kick on beat 3)",
    12: "Bossa nova",
    13: "Odd meter (7/8 or similar)",
    14: "Odd meter (5/4)",
    15: "Big ritardando at the end",
    16: "Clear tempo change partway (e.g. slow verse, fast chorus)",
    17: "Gradual speed-up (accelerando)",
    18: "Stop-time breaks (band stops, restarts)",
    19: "Lo-fi with tape wobble",
    20: "No beat at all (no drums, no pulse)",
    21: "A cappella, no instruments, free tempo",
    22: "Free rubato tempo, no steady pulse",
    23: "Spoken word, weak or sparse beat",
    24: "Vocal pickup before the band enters on beat 1",
    25: "Long song (8+ min)",
}
ASK_METER = {3, 4, 5, 6, 8, 13, 14}


def main() -> None:
    d = Path(sys.argv[1])
    rows = {}
    for line in (d / "candidates.md").read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells and re.fullmatch(r"\d\d[ab]\.\w+", cells[0].strip("`")):
            rows[cells[0].strip("`")[:3]] = cells
    items = []
    for p in sorted((d / "_preview").glob("*.m4a")):
        fid = p.stem
        n = int(fid[:2])
        cells = rows.get(fid, [])
        title = cells[2] if len(cells) > 2 else ""
        meter_q = ""
        if n in ASK_METER:
            meter_q = ('<span class="q">Meter: '
                       + "".join(f'<button data-k="meter" data-v="{v}">{v}</button>'
                                 for v in ("3/4", "4/4", "other")) + "</span>")
        items.append(f'''<article data-id="{fid}">
<h2>{fid} <small>{html.escape(title)}</small></h2>
<p class="feat">#{n}: {html.escape(FEATURE.get(n, ""))}</p>
<audio controls preload="none" src="_preview/{fid}.m4a"></audio>
<div class="qs"><span class="q">Feature there? {"".join(f'<button data-k="feature" data-v="{v}">{v}</button>' for v in ("yes", "no", "unsure"))}</span>
{meter_q}<input class="note" placeholder="note (optional)"></div></article>''')
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Feature Labels</title>
<style>
:root {{ --bg:#fafaf9; --fg:#1c1917; --mute:#78716c; --line:#e7e5e4; --card:#fff; --on:#1d4ed8; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#1c1917; --fg:#f5f5f4; --mute:#a8a29e; --line:#44403c; --card:#292524; --on:#60a5fa; }} }}
body {{ background:var(--bg); color:var(--fg); font:14px/1.4 -apple-system,system-ui,sans-serif; max-width:900px; margin:0 auto; padding:16px; }}
header {{ display:flex; gap:12px; align-items:center; flex-wrap:wrap; position:sticky; top:0; background:var(--bg); padding:8px 0; z-index:1; }}
article {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:10px 14px; margin:12px 0; }}
article.done {{ opacity:.55; }}
h2 {{ margin:0; font-size:16px; }} h2 small {{ color:var(--mute); font-weight:400; }}
.feat {{ margin:4px 0 8px; font-weight:600; }}
audio {{ width:100%; height:32px; }}
.qs {{ display:flex; gap:16px; flex-wrap:wrap; align-items:center; margin-top:8px; }}
button {{ border:1px solid var(--line); background:transparent; color:var(--fg); border-radius:6px; padding:4px 10px; margin-left:4px; cursor:pointer; }}
button.on {{ background:var(--on); border-color:var(--on); color:#fff; }}
.note {{ flex:1; min-width:140px; background:transparent; color:var(--fg); border:1px solid var(--line); border-radius:6px; padding:4px 6px; }}
#count {{ color:var(--mute); flex:1; }}
</style></head><body>
<header><h1 style="margin:0;font-size:20px">Feature labels</h1><span id="count"></span>
<button id="export">Export labels JSON</button></header>
<p style="color:var(--mute)">Skim each file and answer whether the feature in bold is really there. Playing one file pauses the others.</p>
{"".join(items)}
<script>
const KEY="feature-labels-v1"; let st={{}};
try {{ st=JSON.parse(localStorage.getItem(KEY)||"{{}}"); }} catch(e) {{}}
const save=()=>{{ try {{ localStorage.setItem(KEY, JSON.stringify(st)); }} catch(e) {{}} refresh(); }};
const auds=[...document.querySelectorAll("audio")];
auds.forEach(a=>a.addEventListener("play",()=>auds.forEach(o=>{{ if(o!==a) o.pause(); }})));
function refresh() {{
  let n=0;
  document.querySelectorAll("article").forEach(art=>{{
    const s=st[art.dataset.id]||{{}}, needM=!!art.querySelector('[data-k=meter]');
    const done=s.feature && (!needM || s.meter); art.classList.toggle("done", !!done); if(done) n++;
  }});
  document.getElementById("count").textContent=`${{n}} / ${{document.querySelectorAll("article").length}} done`;
}}
document.querySelectorAll("article").forEach(art=>{{
  const id=art.dataset.id;
  art.querySelectorAll("button").forEach(b=>{{
    if((st[id]||{{}})[b.dataset.k]===b.dataset.v) b.classList.add("on");
    b.onclick=()=>{{
      const k=b.dataset.k, on=b.classList.contains("on");
      art.querySelectorAll(`[data-k=${{k}}]`).forEach(x=>x.classList.remove("on"));
      st[id]={{...(st[id]||{{}}), [k]: on?null:b.dataset.v}}; if(!on) b.classList.add("on"); save();
    }};
  }});
  const nt=art.querySelector(".note"); nt.value=(st[id]||{{}}).note||"";
  nt.oninput=()=>{{ st[id]={{...(st[id]||{{}}), note:nt.value}}; save(); }};
}});
refresh();
document.getElementById("export").onclick=()=>{{
  const a=document.createElement("a");
  a.href=URL.createObjectURL(new Blob([JSON.stringify(st,null,2)],{{type:"application/json"}}));
  a.download="feature-labels.json"; a.click();
}};
</script></body></html>'''
    out = d / "label.html"
    out.write_text(page)
    print(out, len(items), "files")


if __name__ == "__main__":
    main()
