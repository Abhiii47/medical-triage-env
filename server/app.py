import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from typing import Optional
from pydantic import BaseModel

from server.env import MedicalTriageEnv
from models import IncidentAction, IncidentObservation
from grader import grade_task

app = FastAPI(
    title="Medical Triage OpenEnv",
    version="0.1.0",
    description="ER Triage Nurse environment — OpenEnv spec compliant",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_env = MedicalTriageEnv()
_last_obs: dict = {}
_action_log: list = []


@app.post("/reset")
def reset(config: Optional[dict] = None):
    global _last_obs, _action_log
    _action_log = []
    body = config or {}
    diff = body.get("difficulty", os.getenv("ENV_DIFFICULTY", "easy"))
    obs = _env.reset(difficulty=diff)
    _last_obs = obs.model_dump()
    _action_log.append({"step": 0, "action": "reset", "feedback": "Environment initialized.", "reward": None})
    return _last_obs


@app.post("/step")
def step(action_dict: dict):
    global _last_obs
    act = IncidentAction(**action_dict)
    obs, reward, done, info = _env.step(act)
    _last_obs = obs.model_dump()
    _last_obs["reward"] = round(float(reward), 4)
    _last_obs["done"] = done
    _action_log.append({
        "step": obs.current_step,
        "action": f"{act.action_type}({act.patient_id or ''} {act.target or ''})".strip(),
        "feedback": obs.action_feedback,
        "reward": round(float(reward), 4),
    })
    if len(_action_log) > 50:
        _action_log.pop(0)
    return _last_obs


@app.get("/state")
def get_state():
    return _env.state().model_dump()


@app.get("/health")
def health():
    return {"status": "healthy", "service": "medical-triage-env", "version": "0.1.0"}


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "id": "easy",
                "difficulty": "easy",
                "max_steps": 15,
                "description": "Single STEMI patient — identify and admit to Cardiology",
                "success_criteria": "Triage level 1, order ECG, treat with Aspirin, admit to Cardiology",
                "expected_score_range": [0.70, 1.0],
                "grader": "grade_task"
            },
            {
                "id": "medium",
                "difficulty": "medium",
                "max_steps": 20,
                "description": "Sepsis (Penicillin allergy) + Opioid Overdose — avoid fatal interactions",
                "success_criteria": "Avoid Penicillin for P-102, use Naloxone for P-108, admit both correctly",
                "expected_score_range": [0.70, 1.0],
                "grader": "grade_task"
            },
            {
                "id": "hard",
                "difficulty": "hard",
                "max_steps": 25,
                "description": "Mass casualty: Hemorrhagic Shock + Stroke + Asthmatic child",
                "success_criteria": "Prioritize P-104 (Level 1), avoid blood thinners, correct wards for all",
                "expected_score_range": [0.60, 1.0],
                "grader": "grade_task"
            }
        ]
    }



class GradingRequest(BaseModel):
    task_id: str = "easy"
    state: Optional[dict] = None
    episode: Optional[dict] = None


@app.post("/grader")
def grade_episode(grading_request: GradingRequest):
    task_id = grading_request.task_id
    
    score = grade_task(task_id, _env.get_state(), _env.all_patients_history)
    return {"task_id": task_id, "score": score}


@app.get("/dashboard_data")
def dashboard_data():
    return JSONResponse({"obs": _last_obs, "log": _action_log[-15:]})


DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Medical Triage Env</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root { --bg:#07090f;--surface:#0d1117;--border:#1c2333;--border2:#243044;--text:#cdd9e5;--muted:#768390;--accent:#2f81f7;--accent2:#388bfd;--green:#3fb950;--yellow:#d29922;--red:#f85149;--red-bg:#1a0a0a;--mono:'IBM Plex Mono',monospace;--sans:'IBM Plex Sans',sans-serif; }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: var(--sans); background: var(--bg); color: var(--text); height: 100vh; overflow: hidden; display: flex; flex-direction: column; }
  .topbar { height: 52px; background: var(--surface); border-bottom: 1px solid var(--border); display: flex; align-items: center; padding: 0 20px; gap: 20px; flex-shrink: 0; }
  .topbar-logo { display: flex; align-items: center; gap: 10px; }
  .topbar-logo svg { width: 22px; height: 22px; }
  .topbar-logo span { font-size: 0.95rem; font-weight: 700; letter-spacing: -0.02em; color: #e6edf3; }
  .topbar-logo em { color: var(--accent); font-style: normal; }
  .topbar-sep { width: 1px; height: 24px; background: var(--border); }
  .topbar-meta { font-size: 0.72rem; color: var(--muted); font-family: var(--mono); }
  .topbar-right { margin-left: auto; display: flex; align-items: center; gap: 12px; }
  .chip { font-size: 0.68rem; font-weight: 600; padding: 3px 10px; border-radius: 4px; font-family: var(--mono); letter-spacing: 0.04em; border: 1px solid transparent; }
  .chip.live  { background: #0d2918; color: var(--green); border-color: #1a4a28; animation: pulse-border 2s infinite; }
  .chip.done  { background: #260d0d; color: var(--red);   border-color: #4a1a1a; }
  .chip.idle  { background: #1a1f2e; color: var(--muted); border-color: var(--border); }
  @keyframes pulse-border { 0%,100%{border-color:#1a4a28} 50%{border-color:#3fb950} }
  .body { display: flex; flex: 1; overflow: hidden; }
  .sidebar { width: 220px; background: var(--surface); border-right: 1px solid var(--border); display: flex; flex-direction: column; flex-shrink: 0; overflow: hidden; }
  .main { flex: 1; overflow: hidden; display: flex; flex-direction: column; }
  .sidebar-section { padding: 14px 16px 10px; border-bottom: 1px solid var(--border); }
  .sidebar-label { font-size: 0.62rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em; color: var(--muted); margin-bottom: 10px; }
  .sidebar-stat { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 6px; }
  .sidebar-stat .key { font-size: 0.72rem; color: var(--muted); }
  .sidebar-stat .val { font-size: 0.85rem; font-weight: 600; font-family: var(--mono); color: var(--text); }
  .sidebar-stat .val.green { color: var(--green); }
  .sidebar-stat .val.red   { color: var(--red); }
  .sidebar-stat .val.yellow{ color: var(--yellow); }
  .stepbar-wrap { margin-top: 8px; }
  .stepbar-track { background: var(--border); border-radius: 2px; height: 4px; overflow:hidden; }
  .stepbar-fill  { height: 4px; border-radius: 2px; background: var(--accent); transition: width .6s ease; }
  .stepbar-fill.warn { background: var(--yellow); }
  .stepbar-fill.crit { background: var(--red); }
  .stepbar-labels { display: flex; justify-content: space-between; font-size: 0.6rem; color: var(--muted); margin-top: 3px; font-family: var(--mono); }
  .queue-list { padding: 12px 14px; flex: 1; overflow-y: auto; }
  .queue-item { background: var(--bg); border: 1px solid var(--border); border-radius: 6px; padding: 10px 12px; margin-bottom: 8px; }
  .queue-item .pid { font-size: 0.65rem; font-family: var(--mono); color: var(--muted); }
  .queue-item .vsm { font-size: 0.7rem; margin-top: 4px; display: flex; gap: 6px; flex-wrap: wrap; }
  .vsm-tag { background: #111824; border: 1px solid var(--border); border-radius: 3px; padding: 1px 6px; font-family: var(--mono); color: var(--muted); }
  .vsm-tag.bad { border-color: var(--red); color: var(--red); background: var(--red-bg); }
  .empty-queue { text-align: center; padding: 24px 0; font-size: 0.72rem; color: var(--muted); }
  .stats-row { display: flex; gap: 0; border-bottom: 1px solid var(--border); flex-shrink: 0; }
  .stat-box { flex: 1; padding: 16px 20px; border-right: 1px solid var(--border); }
  .stat-box:last-child { border-right: none; }
  .stat-box .label { font-size: 0.62rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); margin-bottom: 6px; }
  .stat-box .number { font-size: 1.7rem; font-weight: 700; font-family: var(--mono); line-height: 1; }
  .stat-box .sub { font-size: 0.68rem; color: var(--muted); margin-top: 4px; }
  .content-split { display: flex; flex: 1; overflow: hidden; }
  .beds-area { flex: 2; overflow-y: auto; padding: 16px; display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 14px; align-content: start; border-right: 1px solid var(--border); }
  .bed-card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
  .bed-card.critical { border-color: var(--red); }
  .bed-card.stable   { border-color: var(--green); }
  .bed-header { padding: 8px 14px; background: var(--bg); border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center; }
  .bed-header .bname { font-size: 0.65rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em; color: var(--muted); font-family: var(--mono); }
  .bed-header .bstatus { font-size: 0.62rem; font-family: var(--mono); }
  .bed-header .bstatus.occupied { color: var(--yellow); }
  .bed-header .bstatus.empty    { color: var(--muted); }
  .bed-body { padding: 14px; }
  .bed-pid { font-size: 0.7rem; font-family: var(--mono); color: var(--muted); }
  .bed-empty-msg { text-align: center; padding: 18px 0; font-size: 0.75rem; color: var(--border2); }
  .vitals-monitor { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-top: 10px; }
  .vital-box { background: #060a10; border: 1px solid var(--border); border-radius: 5px; padding: 8px 10px; }
  .vital-box.danger { border-color: var(--red); background: var(--red-bg); }
  .vital-box .vk { font-size: 0.58rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); margin-bottom: 2px; }
  .vital-box .vv { font-size: 1.1rem; font-weight: 600; font-family: var(--mono); line-height: 1; }
  .vital-box.danger .vv { color: var(--red); }
  .vital-box.danger .vk { color: #a04040; }
  .triage-row { margin-top: 10px; display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
  .triage-pill { font-size: 0.65rem; font-weight: 600; padding: 3px 10px; border-radius: 3px; font-family: var(--mono); letter-spacing: 0.04em; }
  .tp1 { background: #2d0b0b; color: #ff6b6b; border: 1px solid #5c1a1a; }
  .tp2 { background: #2d1a0b; color: #ffa94d; border: 1px solid #6b3a14; }
  .tp3 { background: #2b2200; color: #ffdd57; border: 1px solid #5a4800; }
  .tp4 { background: #0b2d14; color: #69db7c; border: 1px solid #1a5c2e; }
  .tp5 { background: #0d1f35; color: #74c0fc; border: 1px solid #1a4070; }
  .treatment-tag { font-size: 0.6rem; background: #0d1f35; border: 1px solid var(--border2); color: var(--accent); border-radius: 3px; padding: 2px 7px; font-family: var(--mono); }
  .right-panel { flex: 1; display: flex; flex-direction: column; overflow: hidden; min-width: 350px; background: var(--surface); }
  .alerts-pane { flex: 1; border-bottom: 1px solid var(--border); overflow-y: auto; padding: 0 14px; min-height: 150px; }
  .log-pane    { flex: 2; overflow-y: auto; padding: 0 14px; min-height: 200px; }
  .pane-header { position: sticky; top: 0; background: var(--bg); padding: 8px 0 6px; font-size: 0.6rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em; color: var(--muted); border-bottom: 1px solid var(--border); margin-bottom: 6px; z-index: 1; display: flex; align-items: center; gap: 6px; }
  .pane-header .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--red); animation: blink 1s step-start infinite; }
  @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }
  .alert-row { display: flex; align-items: flex-start; gap: 8px; padding: 5px 0; border-bottom: 1px solid var(--border); font-size: 0.7rem; }
  .alert-row:last-child { border-bottom: none; }
  .alert-row .icon { margin-top: 1px; flex-shrink:0; }
  .alert-row.critical .msg { color: var(--red); }
  .alert-row.warn     .msg { color: var(--yellow); }
  .no-data { color: var(--border2); font-size: 0.72rem; padding: 12px 0; text-align: center; }
  .log-row { display: grid; grid-template-columns: 42px 1fr 56px; gap: 8px; align-items: baseline; padding: 5px 0; border-bottom: 1px solid var(--border); font-size: 0.68rem; }
  .log-row:last-child { border-bottom: none; }
  .log-step { font-family: var(--mono); color: var(--border2); font-size: 0.6rem; }
  .log-act  { color: var(--accent2); font-family: var(--mono); }
  .log-fb   { color: var(--muted); font-size: 0.62rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .log-rw   { font-family: var(--mono); font-size: 0.65rem; text-align: right; }
  .log-rw.pos { color: var(--green); }
  .log-rw.neg { color: var(--red); }
  .log-rw.neu { color: var(--muted); }
</style>
</head>
<body>
<div class="topbar">
  <div class="topbar-logo">
    <svg viewBox="0 0 24 24" fill="none" stroke="#2f81f7" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a10 10 0 1 0 0 20A10 10 0 0 0 12 2z"/><path d="M12 8v8M8 12h8"/></svg>
    <span>Medical<em>Triage</em> Env</span>
  </div>
  <div class="topbar-sep"></div>
  <div class="topbar-meta">Emergency Triage Command System v0.1.0</div>
  <div class="topbar-right">
    <div class="topbar-meta" id="ep-id">ep —</div>
    <div class="chip idle" id="status-chip">IDLE</div>
  </div>
</div>
<div class="body">
  <div class="sidebar">
    <div class="sidebar-section">
      <div class="sidebar-label">Episode</div>
      <div class="sidebar-stat"><span class="key">Step</span><span class="val" id="sv-step">—</span></div>
      <div class="sidebar-stat"><span class="key">Max Steps</span><span class="val" id="sv-max">—</span></div>
      <div class="sidebar-stat"><span class="key">Last Reward</span><span class="val" id="sv-reward">—</span></div>
      <div class="stepbar-wrap">
        <div class="stepbar-track"><div class="stepbar-fill" id="step-fill" style="width:0%"></div></div>
        <div class="stepbar-labels"><span id="sl-left">0</span><span id="sl-right">—</span></div>
      </div>
    </div>
    <div class="sidebar-section">
      <div class="sidebar-label">ER Status</div>
      <div class="sidebar-stat"><span class="key">Active Beds</span><span class="val green" id="sv-beds">—</span></div>
      <div class="sidebar-stat"><span class="key">Queue</span><span class="val" id="sv-queue">—</span></div>
      <div class="sidebar-stat"><span class="key">Fatal Errors</span><span class="val" id="sv-fatal">—</span></div>
    </div>
    <div class="sidebar-label" style="padding:12px 16px 4px">Waiting Queue</div>
    <div class="queue-list" id="queue-list"><div class="empty-queue">No patients queued</div></div>
  </div>
  <div class="main">
    <div class="stats-row">
      <div class="stat-box"><div class="label">Patients in Beds</div><div class="number" id="ms-beds">—</div><div class="sub">of available capacity</div></div>
      <div class="stat-box"><div class="label">Critical Alerts</div><div class="number" style="color:var(--red)" id="ms-alerts">0</div><div class="sub">vitals warnings fired</div></div>
      <div class="stat-box"><div class="label">Actions Taken</div><div class="number" id="ms-actions">0</div><div class="sub">by agent this episode</div></div>
      <div class="stat-box"><div class="label">Last Reward</div><div class="number" id="ms-reward" style="color:var(--muted)">—</div><div class="sub">step-level signal</div></div>
    </div>
    <div class="content-split">
      <div class="beds-area" id="beds-area"><div style="grid-column:1/-1;text-align:center;padding:40px;color:var(--border2);font-size:.8rem;">POST /reset to begin an episode</div></div>
      <div class="right-panel">
        <div class="alerts-pane"><div class="pane-header"><div class="dot"></div>Alerts &amp; Vitals Warnings</div><div id="alerts-list"><div class="no-data">No alerts</div></div></div>
        <div class="log-pane"><div class="pane-header">Agent Action Log</div><div id="log-list"><div class="no-data">Awaiting agent…</div></div></div>
      </div>
    </div>
  </div>
</div>
<script>
const TRIAGE={1:['tp1','L1 Resuscitation'],2:['tp2','L2 Emergent'],3:['tp3','L3 Urgent'],4:['tp4','L4 Less Urgent'],5:['tp5','L5 Non-Urgent']};
function isDanger(k,v){const n=parseInt(v);if(k==='HR'&&(n>130||n<45))return true;if(k==='O2'&&n<92)return true;if(k==='BP'){const s=parseInt(v.split('/')[0]);return s<80||s>180;}return false;}
function renderVital(k,v,delta){const d=isDanger(k,v);let dT='';if(delta&&delta[k]){const dv=delta[k];const sgn=dv>0?'+':'';const bad=(k==='O2'&&dv<0)||(k==='HR'&&dv>0);dT='<span style="font-size:0.7em;margin-left:4px;color:'+(bad?'var(--red)':'var(--green)')+'">'+sgn+dv+'</span>';}return'<div class="vital-box '+(d?'danger':'')+'"><div class="vk">'+k+'</div><div class="vv">'+v+dT+'</div></div>';}
function renderBedCard(bedName,p){if(!p||p==='Empty'){return'<div class="bed-card"><div class="bed-header"><span class="bname">'+bedName+'</span><span class="bstatus empty">VACANT</span></div><div class="bed-body"><div class="bed-empty-msg">— bed available —</div></div></div>';}const stable=p.stable!==false;const vitalsHtml=Object.entries(p.vitals||{}).map(([k,v])=>renderVital(k,v,p.vitals_delta)).join('');const tri=TRIAGE[p.triage_level];const triHtml=tri?'<span class="triage-pill '+tri[0]+'">'+tri[1]+'</span>':'';const txHtml=(p.treatments||[]).map(t=>'<span class="treatment-tag">'+t+'</span>').join('');let tHtml='';if(p.deterioration_trend&&p.deterioration_trend!=='stable'){const isW=p.deterioration_trend==='worsening';tHtml='<span style="font-size:0.75rem;color:var(--'+(isW?'red':'green')+');margin-left:8px;">'+(isW?'Worsening':'Improving')+'</span>';}let qHtml=p.time_in_queue!==undefined?'<span style="font-size:0.75rem;color:var(--border2);margin-left:8px;">'+p.time_in_queue+'st</span>':'';return'<div class="bed-card '+(stable?'stable':'critical')+'"><div class="bed-header"><span class="bname">'+bedName+'</span><span class="bstatus occupied">OCCUPIED</span></div><div class="bed-body"><div class="bed-pid">Patient ID: '+p.id+qHtml+tHtml+'</div><div class="vitals-monitor">'+vitalsHtml+'</div><div class="triage-row">'+triHtml+txHtml+'</div></div></div>';}
function renderQueueItem(p){const vitals=Object.entries(p.vitals||{}).map(([k,v])=>{let dv='';if(p.vitals_delta&&p.vitals_delta[k]){const vVal=p.vitals_delta[k];const bad=(k==='O2'&&vVal<0)||(k==='HR'&&vVal>0);dv='<span style="font-size:0.7em;margin-left:2px;color:var(--'+(bad?'red':'green')+')">'+(vVal>0?'+':'')+vVal+'</span>';}return'<span class="vsm-tag '+(isDanger(k,v)?'bad':'')+'">'+k+' '+v+dv+'</span>';}).join('');return'<div class="queue-item"><div class="pid">'+p.id+'</div><div class="vsm">'+vitals+'</div></div>';}
async function refresh(){try{const r=await fetch('/dashboard_data');if(!r.ok)return;const{obs,log}=await r.json();if(!obs||obs.current_step===undefined)return;const step=obs.current_step??0;const maxS=obs.max_steps||0;const reward=obs.reward;const done=obs.done;const queue=obs.queue_summary||[];const beds=obs.active_beds_summary||{};const alerts=obs.alerts||[];document.getElementById('sv-step').textContent=step;document.getElementById('sv-max').textContent=maxS;document.getElementById('sl-left').textContent=step;document.getElementById('sl-right').textContent=maxS;const pct=maxS>0?Math.min(100,step/maxS*100):0;document.getElementById('step-fill').style.width=pct+'%';document.getElementById('step-fill').className='stepbar-fill'+(pct>80?' crit':pct>50?' warn':'');const rStr=reward!==undefined&&reward!==null?(reward>0?'+':'')+Number(reward).toFixed(4):'—';document.getElementById('sv-reward').textContent=rStr;const occupiedCount=Object.values(beds).filter(p=>p&&p!=='Empty'&&p.id).length;document.getElementById('sv-beds').textContent=occupiedCount;document.getElementById('sv-queue').textContent=queue.length;const actionCount=(log||[]).filter(l=>l.action!=='reset').length;const critCount=alerts.filter(a=>a.includes('CRITICAL')).length;const chip=document.getElementById('status-chip');chip.textContent=done?'DONE':step>0?'LIVE':'READY';chip.className='chip '+(done?'done':step>0?'live':'idle');if(obs.episode_id)document.getElementById('ep-id').textContent='ep '+obs.episode_id;document.getElementById('ms-beds').textContent=occupiedCount;document.getElementById('ms-alerts').textContent=critCount;document.getElementById('ms-actions').textContent=actionCount;const rwEl=document.getElementById('ms-reward');rwEl.textContent=rStr;rwEl.style.color=reward>0?'var(--green)':reward<0?'var(--red)':'var(--muted)';document.getElementById('queue-list').innerHTML=queue.length?queue.map(renderQueueItem).join(''):'<div class="empty-queue">Queue empty</div>';document.getElementById('beds-area').innerHTML=Object.entries(beds).map(([b,p])=>renderBedCard(b,p)).join('');const al=document.getElementById('alerts-list');al.innerHTML=alerts.length?alerts.map(a=>'<div class="alert-row '+(a.includes('CRITICAL')?'critical':'warn')+'"><span class="icon">'+(a.includes('CRITICAL')?'CRIT':'WARN')+'</span><span class="msg">'+a+'</span></div>').join(''):'<div class="no-data">No alerts</div>';const ll=document.getElementById('log-list');ll.innerHTML=log&&log.length?log.map(e=>{const rw=e.reward;const rwClass=rw>0?'pos':rw<0?'neg':'neu';const rwLabel=rw!==null&&rw!==undefined?(rw>0?'+':'')+Number(rw).toFixed(4):'';return'<div class="log-row"><span class="log-step">#'+e.step+'</span><div><div class="log-act">'+e.action+'</div><div class="log-fb">'+(e.feedback||'')+'</div></div><span class="log-rw '+rwClass+'">'+rwLabel+'</span></div>';}).join(''):'<div class="no-data">Awaiting agent…</div>';}catch(e){}}
setInterval(refresh,900);refresh();
</script>
</body>
</html>
"""


@app.get("/ui", response_class=HTMLResponse)
def get_dashboard():
    return DASHBOARD_HTML


@app.get("/", response_class=HTMLResponse)
def root():
    return DASHBOARD_HTML


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
