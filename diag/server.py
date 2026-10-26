"""Local HTTP service: a small web page for technicians and a JSON API, no internet needed.

    uvicorn diag.server:app --host 0.0.0.0 --port 8080

POST /api/ask {"question": "...", "device": optional} returns the answer, its sources (manual
and page), the matched fault's reaction and acknowledgment copied from the manual, and the
checks on the answer. GET /api/health reports model and index state.
"""

import os
import threading
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from diag.agent import DEFAULT_MODEL, Agent

app = FastAPI(title="Offline drive diagnostic agent")
_agent = None
_lock = threading.Lock()  # one generation at a time: the device has a few CPU cores, not a GPU


def agent() -> Agent:
    global _agent
    if _agent is None:
        _agent = Agent(kb=Path(os.environ.get("DIAG_KB", "data/kb")), index=Path(os.environ.get("DIAG_INDEX", "data/index")),
                       model=os.environ.get("DIAG_MODEL", DEFAULT_MODEL), n_threads=int(os.environ.get("DIAG_THREADS", 4)))
    return _agent


class Ask(BaseModel):
    question: str = Field(description="Technician's question, e.g. a fault code and what happened.")
    device: str | None = Field(default=None, description="'SINAMICS G120C' or 'SINAMICS G120 CU240B-2/CU240E-2'.")


@app.on_event("startup")
def warm() -> None:
    agent()


@app.get("/api/health")
def health() -> dict:
    a = agent()
    return {"model": a.model, "documents": len(a.retriever.docs)}


@app.post("/api/ask")
def ask(req: Ask) -> dict:
    with _lock:
        ans = agent().ask(req.question, device=req.device)
    return {"answer": ans.text, "sources": ans.sources, "facts": ans.facts, "exact": ans.exact,
            "passed": ans.passed, "grounded": ans.grounded, "mentioned": ans.mentioned,
            "unsupported": ans.unsupported, "invented": ans.invented, "ack_conflict": ans.ack_conflict,
            "timing_ms": {"retrieve": round(ans.retrieve_ms), "generate": round(ans.generate_ms)}}


PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Drive diagnostics</title><style>
body{font:15px/1.5 system-ui,sans-serif;max-width:820px;margin:24px auto;padding:0 16px;color:#1b1f24;background:#f6f7f9}
textarea,select,button{font:inherit;width:100%;box-sizing:border-box;margin:6px 0;padding:8px;border:1px solid #c5ccd6;border-radius:6px}
button{background:#00646e;color:#fff;border:0;cursor:pointer}button:disabled{opacity:.6}
pre{white-space:pre-wrap;background:#fff;border:1px solid #dde2e8;border-radius:6px;padding:12px}
.ok{color:#0a6b2d}.bad{color:#a4161a}small{color:#5b6573}
.fact{background:#e8f1f2;border-left:4px solid #00646e;padding:8px 12px;margin:8px 0;border-radius:4px}</style></head><body>
<h2>Drive diagnostics (offline)</h2><small>Answers come only from the SINAMICS List Manuals stored on this device.</small>
<select id="d"><option value="">Device: detect from question</option><option>SINAMICS G120C</option>
<option>SINAMICS G120 CU240B-2/CU240E-2</option></select>
<textarea id="q" rows="4" placeholder="e.g. G120C shows F7900 when the conveyor starts in automatic mode"></textarea>
<button id="b" onclick="go()">Ask</button><div id="o"></div>
<script>
async function go(){const b=document.getElementById('b'),o=document.getElementById('o');b.disabled=true;o.innerHTML='<small>Working on it...</small>';
try{const r=await fetch('api/ask',{method:'POST',headers:{'Content-Type':'application/json'},
body:JSON.stringify({question:document.getElementById('q').value,device:document.getElementById('d').value||null})});const j=await r.json();
const esc=s=>s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const facts=j.facts.map(f=>'<div class="fact"><b>'+esc(f.code+' '+f.title)+'</b><br>From the manual ('+esc(f.cite)+'): reaction '+
esc(f.reaction)+', acknowledge '+esc(f.acknowledge)+'</div>').join('');
const checks=[];if(!j.exact)checks.push('No fault code or parameter in the question matched the manuals; the entries below are the closest text matches and may not apply.');
const missing=j.unsupported.concat(j.invented);if(missing.length)checks.push('Not in the cited pages: '+esc(missing.join(', '))+'.');
if(j.ack_conflict.length)checks.push('The answer states a different acknowledgment than the manual for '+esc(j.ack_conflict.join(', '))+'; follow the manual line above.');
o.innerHTML=facts+'<pre>'+esc(j.answer)+'</pre>'+(checks.length?'<p class="bad">Check before acting: '+checks.join(' ')+'</p>':
'<p class="ok">Every code and parameter in the answer is in the cited manual pages.</p>')+'<small>'+
j.sources.map(s=>s.tag+': '+s.id+' ('+s.cite+')').join('<br>')+'<br>retrieve '+j.timing_ms.retrieve+' ms, generate '+j.timing_ms.generate+' ms</small>';}
catch(e){o.textContent='Error: '+e}b.disabled=false}
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def page() -> str:
    return PAGE
