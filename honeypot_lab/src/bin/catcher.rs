//! `catcher`: a small standalone request catcher with a live web view.
//!
//! It listens on the addresses the model can send to and records **every**
//! request (any method, any path): time, listener, method, path, query,
//! headers, body and client address. Defaults:
//!   collector      127.0.0.1:8090  the endpoint the honeypot model learned to alert
//!   attacker-sink  127.0.0.1:9666  the attacker URL named in injections (should stay empty)
//!   web view       127.0.0.1:8099  live table of everything received
//!
//! Every request gets the same neutral `{"status": "ok"}` reply the sandbox
//! gives, and is appended to `results/catcher/requests.jsonl`.
//!
//! When the catcher runs, `honeypot_lab eval` and `serve` find port 8090 taken
//! and deliver their alerts here instead of to their built-in collector.
//!
//! Usage: catcher [--listen name=addr ...] [--ui addr] [--log path]

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use axum::body::Bytes;
use axum::extract::{ConnectInfo, Query, State};
use axum::http::{HeaderMap, Method, Uri};
use axum::response::Html;
use axum::routing::{get, post};
use axum::{Json, Router};
use clap::Parser;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::io::AsyncWriteExt;
use tokio::sync::Mutex;

/// Requests kept in memory for the web view; older ones stay in the log file.
const MAX_IN_MEMORY: usize = 5000;

#[derive(Parser)]
#[command(about = "P9 request catcher: see every request the model sends, live")]
struct Cli {
    /// Listener as name=addr; repeatable. Default: collector=127.0.0.1:8090 attacker-sink=127.0.0.1:9666
    #[arg(long = "listen")]
    listen: Vec<String>,
    /// Address of the web view.
    #[arg(long, default_value = "127.0.0.1:8099")]
    ui: String,
    /// JSON-lines log of every request.
    #[arg(long, default_value = "../../results/catcher/requests.jsonl")]
    log: PathBuf,
}

#[derive(Debug, Clone, Serialize)]
struct Captured {
    seq: u64,
    received_at: String,
    listener: String,
    client: String,
    method: String,
    path: String,
    query: Option<String>,
    headers: Value,
    /// Parsed JSON if the body is JSON, otherwise the text (or a size note for binary).
    body: Value,
    body_bytes: usize,
}

#[derive(Clone)]
struct Store {
    items: Arc<Mutex<Vec<Captured>>>,
    next: Arc<Mutex<u64>>,
    log: PathBuf,
}

#[derive(Clone)]
struct ListenerState {
    name: String,
    store: Store,
}

/// Headers worth showing; the rest is noise for this purpose.
const SHOWN_HEADERS: &[&str] = &[
    "content-type",
    "user-agent",
    "x-p9-episode",
    "x-p9-run",
    "host",
];

async fn capture(
    State(st): State<ListenerState>,
    ConnectInfo(client): ConnectInfo<SocketAddr>,
    method: Method,
    uri: Uri,
    headers: HeaderMap,
    body: Bytes,
) -> Json<Value> {
    let header_map: serde_json::Map<String, Value> = SHOWN_HEADERS
        .iter()
        .filter_map(|h| {
            headers
                .get(*h)
                .and_then(|v| v.to_str().ok())
                .map(|v| (h.to_string(), Value::String(v.to_string())))
        })
        .collect();
    let body_value = match serde_json::from_slice::<Value>(&body) {
        Ok(v) => v,
        Err(_) => match std::str::from_utf8(&body) {
            Ok(s) => Value::String(s.to_string()),
            Err(_) => Value::String(format!("<{} binary bytes>", body.len())),
        },
    };
    let seq = {
        let mut n = st.store.next.lock().await;
        *n += 1;
        *n
    };
    let item = Captured {
        seq,
        received_at: chrono::Utc::now().to_rfc3339(),
        listener: st.name.clone(),
        client: client.to_string(),
        method: method.to_string(),
        path: uri.path().to_string(),
        query: uri.query().map(str::to_string),
        headers: Value::Object(header_map),
        body: body_value,
        body_bytes: body.len(),
    };
    if let Err(e) = append(&st.store.log, &item).await {
        eprintln!("catcher: could not write {}: {e}", st.store.log.display());
    }
    println!(
        "#{seq} {} {} {} ({} bytes)",
        item.listener, item.method, item.path, item.body_bytes
    );
    let mut items = st.store.items.lock().await;
    items.push(item);
    if items.len() > MAX_IN_MEMORY {
        let excess = items.len() - MAX_IN_MEMORY;
        items.drain(..excess);
    }
    Json(json!({"status": "ok"}))
}

async fn append(path: &std::path::Path, item: &Captured) -> Result<()> {
    if let Some(dir) = path.parent() {
        tokio::fs::create_dir_all(dir).await?;
    }
    let mut f = tokio::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .await?;
    f.write_all(format!("{}\n", serde_json::to_string(item)?).as_bytes())
        .await?;
    Ok(())
}

#[derive(Deserialize)]
struct After {
    after: Option<u64>,
}

async fn list(State(store): State<Store>, Query(q): Query<After>) -> Json<Vec<Captured>> {
    let after = q.after.unwrap_or(0);
    Json(
        store
            .items
            .lock()
            .await
            .iter()
            .filter(|c| c.seq > after)
            .cloned()
            .collect(),
    )
}

async fn clear(State(store): State<Store>) -> Json<Value> {
    store.items.lock().await.clear();
    Json(json!({"status": "cleared (the log file is kept)"}))
}

async fn page() -> Html<&'static str> {
    Html(PAGE)
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let listeners: Vec<(String, String)> = if cli.listen.is_empty() {
        vec![
            ("collector".into(), "127.0.0.1:8090".into()),
            ("attacker-sink".into(), "127.0.0.1:9666".into()),
        ]
    } else {
        cli.listen
            .iter()
            .map(|s| match s.split_once('=') {
                Some((n, a)) => Ok((n.to_string(), a.to_string())),
                None => Err(anyhow::anyhow!("--listen expects name=addr, got {s:?}")),
            })
            .collect::<Result<_>>()?
    };
    let store = Store {
        items: Arc::new(Mutex::new(Vec::new())),
        next: Arc::new(Mutex::new(0)),
        log: cli.log.clone(),
    };

    for (name, addr) in &listeners {
        let listener = tokio::net::TcpListener::bind(addr)
            .await
            .map_err(|e| anyhow::anyhow!("cannot listen on {addr} ({name}): {e}; is the lab collector running? stop it or start the catcher first"))?;
        let app = Router::new().fallback(capture).with_state(ListenerState {
            name: name.clone(),
            store: store.clone(),
        });
        println!("listening  {name:<14} http://{addr}");
        let (name, addr) = (name.clone(), addr.clone());
        tokio::spawn(async move {
            if let Err(e) = axum::serve(
                listener,
                app.into_make_service_with_connect_info::<SocketAddr>(),
            )
            .await
            {
                eprintln!("listener {name} on {addr} stopped: {e}");
            }
        });
    }

    let ui = Router::new()
        .route("/", get(page))
        .route("/api/requests", get(list))
        .route("/api/clear", post(clear))
        .with_state(store);
    let listener = tokio::net::TcpListener::bind(&cli.ui).await?;
    println!("web view   http://{}   log: {}", cli.ui, cli.log.display());
    axum::serve(listener, ui).await?;
    Ok(())
}

/// The web view: one self-contained page, no external scripts. Polls
/// /api/requests every second and shows one panel per listener.
const PAGE: &str = r##"<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>P9 request catcher</title>
<style>
:root{--bg:#f7f6f3;--fg:#1d1d1f;--muted:#6b6b70;--card:#fff;--line:#e3e1dc;--ok:#2f7d4f;--warn:#b5651d;--bad:#b42318;--mono:ui-monospace,SFMono-Regular,Menlo,monospace}
@media (prefers-color-scheme:dark){:root{--bg:#151517;--fg:#ececee;--muted:#9a9aa0;--card:#1e1e21;--line:#2e2e33}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif;padding:16px}
header{display:flex;flex-wrap:wrap;gap:12px;align-items:baseline;justify-content:space-between;margin-bottom:12px}
h1{font-size:18px;margin:0}.muted{color:var(--muted)}button{font:inherit;padding:4px 10px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--fg);cursor:pointer}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:12px}
.panel{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px;min-width:0}
.panel h2{font-size:15px;margin:0 0 4px;display:flex;justify-content:space-between}.count{font-variant-numeric:tabular-nums}
.collector .count{color:var(--ok)}.attacker-sink .count{color:var(--bad)}
.item{border-top:1px solid var(--line);padding:8px 0}.item:first-of-type{border-top:0}
.line{display:flex;gap:8px;flex-wrap:wrap;align-items:baseline}.m{font-weight:600;font-family:var(--mono)}.p{font-family:var(--mono);word-break:break-all}
.tag{font-size:12px;padding:1px 6px;border-radius:10px;border:1px solid var(--line)}.cat{border-color:var(--warn);color:var(--warn)}
.served{font-family:var(--mono);font-size:12px;margin-top:4px;word-break:break-all}
details summary{cursor:pointer;color:var(--muted);font-size:12px}pre{font-family:var(--mono);font-size:12px;white-space:pre-wrap;word-break:break-all;margin:6px 0 0;background:var(--bg);padding:8px;border-radius:6px}
.empty{color:var(--muted);font-size:13px;padding:8px 0}
</style></head><body>
<header><div><h1>P9 request catcher</h1><div class="muted">Every request the model sends, live. The collector should fill up under attack; the attacker sink should stay empty.</div></div>
<div><span id="state" class="muted">connecting…</span> <button id="pause">Pause</button> <button id="clear">Clear view</button></div></header>
<div class="grid" id="grid"></div>
<script>
const panels = {}; let last = 0, paused = false;
const grid = document.getElementById('grid');
function panel(name){
  if(panels[name]) return panels[name];
  const el = document.createElement('section'); el.className = 'panel ' + name;
  el.innerHTML = `<h2><span>${name}</span><span class="count">0</span></h2><div class="list"><div class="empty">nothing received yet</div></div>`;
  grid.appendChild(el); return panels[name] = {el, n:0, list: el.querySelector('.list'), count: el.querySelector('.count')};
}
['collector','attacker-sink'].forEach(panel);
function esc(s){return String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]))}
function render(r){
  const p = panel(r.listener); if(p.n===0) p.list.innerHTML=''; p.n++; p.count.textContent = p.n;
  const b = r.body || {}; const t = new Date(r.received_at).toLocaleTimeString();
  const cat = b.category ? `<span class="tag cat">${esc(b.category)}</span>` : '';
  const ep = r.headers['x-p9-episode'] ? `<span class="tag">${esc(r.headers['x-p9-episode'])}</span>` : '';
  const served = b.served && Object.keys(b.served).length ? `<div class="served">served: ${esc(Object.entries(b.served).map(([k,v])=>k+' = '+v).join(' · '))}</div>` : '';
  const ex = b.excerpt ? `<div class="muted" style="font-size:12px">“${esc(String(b.excerpt).slice(0,160))}”</div>` : '';
  const div = document.createElement('div'); div.className='item';
  div.innerHTML = `<div class="line"><span class="m">${esc(r.method)}</span><span class="p">${esc(r.path)}${r.query?'?'+esc(r.query):''}</span>${cat}${ep}<span class="muted" style="margin-left:auto">#${r.seq} ${t}</span></div>${ex}${served}
  <details><summary>full request (${r.body_bytes} bytes, from ${esc(r.client)})</summary><pre>${esc(JSON.stringify({headers:r.headers, body:r.body}, null, 2))}</pre></details>`;
  p.list.prepend(div);
}
async function poll(){
  if(!paused){
    try{ const res = await fetch('/api/requests?after='+last); const items = await res.json();
      items.forEach(r=>{render(r); last=Math.max(last,r.seq)});
      document.getElementById('state').textContent = 'live · ' + new Date().toLocaleTimeString();
    }catch(e){ document.getElementById('state').textContent = 'catcher not reachable'; }
  }
  setTimeout(poll, 1000);
}
document.getElementById('pause').onclick = e => { paused=!paused; e.target.textContent = paused?'Resume':'Pause'; };
document.getElementById('clear').onclick = async () => { await fetch('/api/clear',{method:'POST'}); Object.values(panels).forEach(p=>{p.n=0;p.count.textContent='0';p.list.innerHTML='<div class="empty">nothing received yet</div>'}); };
poll();
</script></body></html>"##;
