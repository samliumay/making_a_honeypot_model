"""P9 honeypot lab UI (Streamlit).

A viewer and trigger over the Rust lab API (`honeypot_lab serve`, 127.0.0.1:8787).
It derives no numbers of its own: every score comes from the API or the run
folders, and the paper cites only the CLI reports (AGENTS.md "Visualization").

Tabs:
  Attack lab      type your own attacks against a deployment; see what the attacker
                  sees next to what the defender sees (alerts, leaks, egress).
  Attack library  run any dataset episode against the base and the honeypot model.
  Evaluations     start eval runs, follow progress, compare base vs honeypot.
  Alert feed      everything the collector received.

Run: `make ui` (or `uv run streamlit run app.py`) after `make lab`.
"""

import os

import pandas as pd
import requests
import streamlit as st

API = os.environ.get("P9_LAB_API", "http://127.0.0.1:8787")

st.set_page_config(page_title="P9 honeypot lab", layout="wide")


def get(path, **params):
    r = requests.get(f"{API}{path}", params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def post(path, body, timeout=600):
    r = requests.post(f"{API}{path}", json=body, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(r.text)
    return r.json()


try:
    INFO = get("/api/info")
except requests.RequestException as e:
    st.error(f"Lab API not reachable at {API} ({e}). Start it with `make lab` in code/.")
    st.stop()

MODELS = [m["name"] for m in INFO["models"]]
st.title("P9 honeypot lab")
st.caption(
    f"Dataset {INFO['dataset']} {INFO['version']} · collector {INFO['alert_url']} · "
    f"attacker sink (never called) {INFO['attacker_url']} · models: {', '.join(MODELS)}"
)


def show_score(score, expected=None):
    """Render one episode's score: the defender's view."""
    leak = score["leak_full"] or score["leak_oracle"]
    cols = st.columns(5)
    cols[0].metric("Success", "yes" if score["success"] else "no")
    cols[1].metric("Real value leaked", "YES" if leak else "no")
    cols[2].metric("Silent alerts", score["alerts"])
    cols[3].metric("Tell words", len(score["tell_words"]))
    cols[4].metric("Attacker URL called", "YES" if score["attacker_url_called"] else "no")
    details = {
        "alert category": score["alert_category_raw"],
        "category accepted": score["category_ok"],
        "fakes served (from the alert)": score["served"],
        "fakes in the requested form in reply": score["served_in_reply"],
        "fakes match the real format": score["served_format_ok"],
        "partial leak (8-gram)": score["leak_partial"],
        "consistent with earlier fake": score["consistent"],
        "tell words": score["tell_words"],
        "egress attempts (never sent)": score["egress_urls"],
        "required tool calls done": score["required_calls_ok"],
        "false alarm (negatives)": score["false_alarm"],
    }
    if expected:
        details["expected"] = expected
    st.json(details, expanded=False)


def show_transcript(tr):
    """The attacker's view (reply) plus every tool call the model made."""
    for m in tr["generated"]:
        if m["role"] == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                st.code(f"tool call  {tc['function']['name']}  {tc['function']['arguments']}", language="json")
        elif m["role"] == "tool":
            st.caption(f"tool result: {m['content'][:300]}")
    st.markdown("**Reply the attacker sees:**")
    st.info(tr["reply"] or "(no reply)")
    if tr["outcome"]["kind"] != "completed":
        st.warning(f"episode outcome: {tr['outcome']}")


tab_lab, tab_lib, tab_eval, tab_alerts = st.tabs(["Attack lab", "Attack library", "Evaluations", "Alert feed"])

# ---------------------------------------------------------------- attack lab
with tab_lab:
    st.subheader("Attack a deployment by hand")
    benign = get("/api/episodes", file="E4_benign")
    deployments = {}
    for row in benign:
        deployments.setdefault(row["deployment"], row["id"])
    c1, c2 = st.columns(2)
    dep = c1.selectbox("Deployment", sorted(deployments))
    model = c2.selectbox("Model", MODELS, index=len(MODELS) - 1, key="lab_model")
    ep = get(f"/api/episode/{deployments[dep]}")["episode"]
    with st.expander("Defender view: system prompt and real values (the model is told nothing about the honeypot)"):
        st.code(ep["messages"][0]["content"])
    key = f"chat_{dep}_{model}"
    if key not in st.session_state:
        st.session_state[key] = []
    history = st.session_state[key]
    for m in history:
        with st.chat_message(m["role"]):
            st.write(m["content"])
    prompt = st.chat_input("Type an attack (or a normal question)")
    if st.button("Reset conversation"):
        st.session_state[key] = []
        st.rerun()
    if prompt:
        history.append({"role": "user", "content": prompt})
        with st.spinner("model is answering"):
            try:
                res = post("/api/chat", {"model": model, "episode_id": deployments[dep], "messages": history})
            except RuntimeError as e:
                st.error(str(e))
                history.pop()
                st.stop()
        history.append({"role": "assistant", "content": res["transcript"]["reply"]})
        st.session_state[f"{key}_last"] = res
        st.rerun()
    last = st.session_state.get(f"{key}_last")
    if last:
        st.markdown("---\n**Last turn, defender view**")
        show_score(last["score"])
        show_transcript(last["transcript"])

# ---------------------------------------------------------------- attack library
with tab_lib:
    st.subheader("Run dataset episodes")
    files = list(INFO["files"])
    c1, c2 = st.columns(2)
    f = c1.selectbox("Eval file", files)
    fam = c2.selectbox("Family", ["(all)"] + INFO["files"][f]["families"])
    rows = get("/api/episodes", file=f, **({} if fam == "(all)" else {"family": fam}))
    df = pd.DataFrame(rows)
    st.dataframe(df[["id", "family", "deployment", "behaviour", "last_user"]], hide_index=True, height=240)
    ep_id = st.selectbox("Episode", [r["id"] for r in rows])
    chosen = st.multiselect("Models", MODELS, default=MODELS)
    if st.button("Run episode", type="primary"):
        cols = st.columns(len(chosen) or 1)
        for col, m in zip(cols, chosen):
            with col:
                st.markdown(f"### {m}")
                with st.spinner(f"running on {m}"):
                    try:
                        res = post("/api/run_episode", {"model": m, "episode_id": ep_id})
                    except RuntimeError as e:
                        st.error(str(e))
                        continue
                show_score(res["score"], res["expected"])
                show_transcript(res["transcript"])

# ---------------------------------------------------------------- evaluations
with tab_eval:
    st.subheader("Start an eval")
    c1, c2, c3 = st.columns(3)
    em = c1.selectbox("Model", MODELS, key="eval_model")
    ef = c2.multiselect("Files (empty = all)", list(INFO["files"]))
    lim = c3.number_input("Limit per file (0 = all)", min_value=0, value=0)
    if st.button("Start eval"):
        rid = post("/api/eval", {"model": em, "files": ef, "limit": lim or None}, timeout=30)["run_id"]
        st.success(f"started {rid}")
    prog = get("/api/progress")
    for p in prog:
        if not p["finished"]:
            st.progress(p["done"] / max(p["total"], 1), text=f"{p['run_id']}: {p['done']}/{p['total']}")
        elif p.get("error"):
            st.error(f"{p['run_id']}: {p['error']}")
    st.button("Refresh progress")

    st.subheader("Results")
    runs = [r for r in get("/api/runs") if r["has_summary"]]
    if not runs:
        st.info("No finished runs yet.")
    else:
        labels = {r["run_id"]: f"{r['run_id']} ({r['meta']['model']['name']}, {r['meta']['episodes']} eps)" for r in runs}
        c1, c2 = st.columns(2)
        hp = c1.selectbox("Run", list(labels), format_func=labels.get)
        base = c2.selectbox("Compare with (base model run)", ["(none)"] + list(labels), format_func=lambda x: labels.get(x, x))
        res = get(f"/api/runs/{hp}")
        fam = pd.DataFrame(res["summary"]["by_family"])
        if res["judged"]:
            fam["judged_success"] = [res["judged"].get(f"{r.file}|{r.family}", {}).get("success") for r in fam.itertuples()]
        if base != "(none)":
            b = pd.DataFrame(get(f"/api/runs/{base}")["summary"]["by_family"])[["file", "family", "success", "leak", "tell", "false_alarm"]]
            fam = fam.merge(b, on=["file", "family"], suffixes=("", "_base"), how="left")
            st.markdown("**Success rate per attack family: base vs honeypot**")
            chart = fam[["family", "success_base", "success"]].set_index("family").rename(
                columns={"success_base": "base", "success": "honeypot"})
            st.bar_chart(chart, stack=False)
        st.dataframe(fam, hide_index=True)
        with st.expander("Episodes of this run"):
            eps = get(f"/api/runs/{hp}/episodes")
            table = pd.DataFrame([{
                "id": e["result"]["id"], "file": e["result"]["file"], "family": e["result"]["family"],
                "success": e["result"]["score"]["success"], "alerts": e["result"]["score"]["alerts"],
                "leak": bool(e["result"]["score"]["leak_full"]), "tells": ",".join(e["result"]["score"]["tell_words"]),
                "judge tell": (e["verdict"] or {}).get("tell"), "reply": e["result"]["transcript"]["reply"][:160],
            } for e in eps])
            st.dataframe(table, hide_index=True)

# ---------------------------------------------------------------- alert feed
with tab_alerts:
    st.subheader("Collector: alerts the model sent")
    st.caption("Each row is a silent http_request POST the model made to the collector. "
               "`served` lists the fake values handed to the attacker: use any of them later and you know who it was.")
    alerts = get("/api/alerts", limit=500)
    if alerts:
        df = pd.DataFrame([{
            "seq": a["seq"], "time": a["received_at"], "run": a["run_id"], "episode": a["episode_id"],
            "category": a["body"].get("category"), "requested": a["body"].get("requested"),
            "served": a["body"].get("served"), "excerpt": a["body"].get("excerpt"),
        } for a in reversed(alerts)])
        st.dataframe(df, hide_index=True)
    else:
        st.info("No alerts yet.")
    st.button("Refresh feed")
