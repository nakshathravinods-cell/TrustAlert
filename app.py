"""
TrustAlert - Streamlit dashboard (research prototype).

Run on a laptop:
    pip install streamlit scikit-learn pandas matplotlib joblib
    streamlit run app.py

Folder layout (from the zip the notebook downloads):
    app.py
    sepsisguard.py
    outputs/sepsisguard_bundle.joblib
    outputs/demo_patients.csv      (optional, real held-out test patients)
    outputs/metrics.json           (optional)

Sound: a beep plays only when an alert is RAISED (the start of a high-risk episode),
not every hour. Priority 3 = three high beeps, priority 2 = one lower beep.
This is a DEMONSTRATION on stored data, not a live hospital alarm.
"""
import base64
import io
import json
import os
import time
import wave

import joblib
import numpy as np
import pandas as pd
import sklearn
import streamlit as st
import streamlit.components.v1 as components

import sepsisguard as sg

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
st.set_page_config(page_title="TrustAlert - Research Prototype", layout="wide")

COLORS = {3: "#c0392b", 2: "#e67e22", 1: "#d4a017", 0: "#27ae60"}
ICONS = {3: "🔴", 2: "⚠️", 1: "🟠", 0: "🟢"}


# ------------------------------------------------------------------ sound
def make_wav(pattern, sr=22050):
    """pattern = list of (frequency_hz, duration_ms, gap_ms). Returns WAV bytes."""
    parts = []
    for freq, dur, gap in pattern:
        t = np.arange(int(sr * dur / 1000)) / sr
        tone = 0.5 * np.sin(2 * np.pi * freq * t)
        k = max(1, int(sr * 0.01))                       # 10 ms fade in/out avoids clicks
        env = np.ones_like(tone)
        env[:k], env[-k:] = np.linspace(0, 1, k), np.linspace(1, 0, k)
        parts += [tone * env, np.zeros(int(sr * gap / 1000))]
    pcm = (np.concatenate(parts) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


SOUNDS = {
    3: make_wav([(880, 180, 120)] * 3),   # three high beeps: high-priority alert
    2: make_wav([(520, 350, 0)]),         # one lower beep: verify data
}


def play_sound(slot, priority):
    """Play the beep for this priority inside `slot` (an st.empty())."""
    wav = SOUNDS.get(int(priority))
    if wav is None:
        slot.empty()
        return
    try:
        slot.audio(wav, format="audio/wav", autoplay=True)
    except TypeError:                                    # older Streamlit without autoplay
        b64 = base64.b64encode(wav).decode()
        with slot:
            components.html(f'<audio autoplay src="data:audio/wav;base64,{b64}"></audio>', height=0)


# ------------------------------------------------------------------ model + data
@st.cache_resource
def load_bundle():
    return joblib.load(os.path.join(OUT, "sepsisguard_bundle.joblib"))


st.title("TrustAlert - Early Sepsis Risk Assessment")
st.warning("**Academic research prototype - NOT a diagnostic tool.** Trained on a retrospective public dataset, "
           "with no clinical validation. It must not be used for patient care. The replay below is a "
           "demonstration on stored data, not a live alarm system.")

try:
    bundle = load_bundle()
except Exception as e:
    st.error(f"Could not load outputs/sepsisguard_bundle.joblib: {e}\n\nRun the notebook, then copy its "
             "'outputs' folder next to app.py.")
    st.stop()

if bundle.get("sklearn_version") != sklearn.__version__:
    st.info(f"The model was saved with scikit-learn {bundle.get('sklearn_version')} but this computer has "
            f"{sklearn.__version__}. If you see errors, install the same version.")

cfg, H = bundle["alert_cfg"], bundle["horizon"]

st.sidebar.header("Mode and sound")
mode = st.sidebar.radio("Mode", ["Live monitor (enter vitals each hour)", "Manual (pick an hour)", "Replay hour by hour"])
sound_on = st.sidebar.checkbox("Beep when an alert is raised", value=True)
if st.sidebar.button("🔊 Test sound"):
    play_sound(st.sidebar.empty(), 3)
st.sidebar.caption("Priority 3 = three high beeps. Priority 2 (verify data) = one lower beep. "
                   "Sound plays only when a NEW alert is raised, not every hour. Your browser may need one "
                   "click on the page before it allows sound.")

LIVE_VITALS = [("HR", "Heart rate (beats/min)", 80.0), ("SBP", "Systolic BP (mmHg)", 120.0),
               ("MAP", "Mean BP (mmHg)", 85.0), ("DBP", "Diastolic BP (mmHg)", 70.0),
               ("Resp", "Breathing rate (per min)", 16.0), ("Temp", "Temperature (°C)", 37.0),
               ("O2Sat", "Oxygen saturation (%)", 97.0)]


def status_cards(r):
    """The four headline numbers for one hour (r = one row of `frame`)."""
    p = int(r["priority"])
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"### Alert\n<h2 style='color:{COLORS[p]};margin:0'>{ICONS[p]} {r['alert_type']}</h2>",
                unsafe_allow_html=True)
    c2.metric(f"Estimated chance label turns on within {H} h", f"{r['calibrated']*100:.1f}%",
              help="Calibrated on validation data. Describes this dataset and horizon, not real-world clinical risk.")
    c3.metric("Risk tier / trend", f"{r['tier']} / {str(r['trend']).title()}")
    c4.metric("Data reliability", f"{r['reliability']*100:.0f}%",
              "reliable" if r["reliable"] else "VERIFY DATA",
              delta_color="normal" if r["reliable"] else "inverse")


def risk_chart(fr, c, slot=None, height=270):
    """Interactive risk chart: hover any point for details, scroll to zoom, drag to pan.
    Red triangles mark hours where an alert was raised. Falls back to a plain chart if needed."""
    target = slot if slot is not None else st
    try:
        import altair as alt
        d_ = fr[[sg.TIME, "score", "calibrated", "tier", "policy_B"]].copy()
        d_.columns = ["hour", "score", "chance", "tier", "raised"]
        d_["chance"] = (d_["chance"] * 100).round(1)
        d_["score"] = d_["score"].round(3)
        d_["raised"] = d_["raised"].astype(bool)
        tips = ["hour:Q", "score:Q", "chance:Q", "tier:N"]
        x = alt.X("hour:Q", title="ICU hour")
        y = alt.Y("score:Q", title="Risk score")
        layers = [alt.Chart(d_).mark_line(color="#0a6c86").encode(x=x, y=y),
                  alt.Chart(d_).mark_circle(size=60, color="#0a6c86").encode(x=x, y=y, tooltip=tips)]
        if d_["raised"].any():
            layers.append(alt.Chart(d_[d_["raised"]]).mark_point(
                shape="triangle-up", size=260, filled=True, color="#c0392b").encode(x=x, y=y, tooltip=tips))
        thr = pd.DataFrame({"y": [float(c["thr_high"]), float(c["thr_mod"])],
                            "label": ["HIGH threshold", "MODERATE threshold"]})
        layers.append(alt.Chart(thr).mark_rule(strokeDash=[6, 4]).encode(
            y="y:Q", color=alt.Color("label:N", legend=alt.Legend(title=None, orient="bottom"),
                                     scale=alt.Scale(domain=["HIGH threshold", "MODERATE threshold"],
                                                     range=["#c0392b", "#d4a017"]))))
        target.altair_chart(alt.layer(*layers).properties(width="container", height=height).interactive())
    except Exception:
        tl_ = fr.set_index(sg.TIME)[["score"]].copy()
        tl_["HIGH threshold"], tl_["MODERATE threshold"] = c["thr_high"], c["thr_mod"]
        target.line_chart(tl_)


RANGES = {"HR": (30.0, 200.0), "SBP": (50.0, 220.0), "MAP": (30.0, 150.0), "DBP": (20.0, 130.0),
          "Resp": (6.0, 50.0), "Temp": (34.0, 42.0), "O2Sat": (60.0, 100.0)}


# ------------------------------------------------------------------ LIVE MONITOR MODE
PRESETS = {   # one-click example values for the form (typed-in numbers still work)
    "stable":   dict(HR=80.0,  SBP=120.0, MAP=85.0, DBP=70.0, Resp=16.0, Temp=37.0, O2Sat=97.0),
    "worse":    dict(HR=108.0, SBP=104.0, MAP=72.0, DBP=58.0, Resp=22.0, Temp=38.1, O2Sat=94.0),
    "critical": dict(HR=128.0, SBP=84.0,  MAP=56.0, DBP=42.0, Resp=30.0, Temp=39.0, O2Sat=89.0),
}
SENSITIVITY = {"Fewer alerts (stricter)": 1.4, "Standard": 1.0, "More alerts (earlier warning)": 0.7}

if mode.startswith("Live"):
    ss = st.session_state
    ss.setdefault("live", [])                       # one dict per ICU hour entered so far
    ss.setdefault("play", False)                    # auto-play of a demo patient is running
    ss.setdefault("ack", None)                      # ICU hour at which the user acknowledged an alert
    for key, _, default in LIVE_VITALS:
        ss.setdefault(f"v_{key}", float(default))

    # --- button actions (run before the page redraws) ---
    def set_preset(name):
        for k, v in PRESETS[name].items():
            ss[f"v_{k}"] = v

    def copy_last():
        if ss.live:
            for key, _, _ in LIVE_VITALS:
                v = ss.live[-1].get(key)
                if v is not None and not pd.isna(v):
                    lo, hi = RANGES[key]
                    ss[f"v_{key}"] = float(min(max(float(v), lo), hi))

    def undo_last():
        ss.live = ss.live[:-1]
        ss.ack = None

    def reset_all():
        ss.live = []
        ss.ack = None

    def stop_play():
        ss.play = False

    def acknowledge():
        ss.ack = len(ss.live)

    # --- sidebar: how sensitive should the alarm be? ---
    st.sidebar.header("Alert sensitivity")
    sens = st.sidebar.select_slider("Choose", options=list(SENSITIVITY), value="Standard")
    k_sens = SENSITIVITY[sens]
    cfg_live = dict(cfg)
    cfg_live["thr_high"] = min(0.99, cfg["thr_high"] * k_sens)
    cfg_live["thr_mod"] = min(cfg["thr_mod"] * k_sens, cfg_live["thr_high"])
    bundle_live = {**bundle, "alert_cfg": cfg_live}
    st.sidebar.caption(f"HIGH threshold {cfg_live['thr_high']:.2f}. Moving this slider re-checks every hour you "
                       "entered. 'More alerts' warns earlier but raises more false alarms.")
    age = st.sidebar.number_input("Patient age (years)", 0, 120, 60)

    def live_add(rows):
        """Append hours, re-score the whole stay, return the priority of the strongest NEW alert (or None)."""
        n0 = len(ss.live)
        ss.live = ss.live + rows
        fr, _, _ = sg.score_history(bundle_live, pd.DataFrame(ss.live))
        fr = sg.apply_policies(fr, cfg_live)
        new = fr.iloc[n0:]
        fired = new[new["policy_B"]]
        return int(fired["priority"].max()) if len(fired) else None

    st.subheader("Live monitor: type in each new hour's measurements")
    st.caption("Every time you add an hour, the model re-scores the patient using only the hours entered so far, "
               "and TrustAlert decides whether to raise an alert. A beep plays when a NEW alert is raised. "
               "During the first %d hours no alerts are raised (too little history)." % sg.WARMUP_HOURS)
    new_pri = None

    st.markdown("**Quick fill** (then press *Add this hour*):")
    q1, q2, q3, q4 = st.columns(4)
    q1.button("🟢 Stable example", on_click=set_preset, args=("stable",), use_container_width=True)
    q2.button("🟠 Getting worse", on_click=set_preset, args=("worse",), use_container_width=True)
    q3.button("🔴 Critical example", on_click=set_preset, args=("critical",), use_container_width=True)
    q4.button("↩ Same as last hour", on_click=copy_last, use_container_width=True)

    st.markdown(f"**Measurements for ICU hour {len(ss.live) + 1}**  (move a slider: the preview below updates at once)")
    cols = st.columns(4)
    vals = {}
    for k, (key, label, default) in enumerate(LIVE_VITALS):
        lo, hi = RANGES[key]
        vals[key] = cols[k % 4].slider(label, lo, hi, step=0.1 if key == "Temp" else 1.0, key=f"v_{key}")
    missing = st.multiselect("Not measured this hour (leave out of the model)", [k for k, _, _ in LIVE_VITALS])
    row = {sg.ID: "live", sg.TIME: len(ss.live) + 1, sg.LABEL: 0, "Age": age}
    for key, v in vals.items():
        row[key] = np.nan if key in missing else float(v)

    # live preview: what WOULD happen if this hour were added (nothing is saved)
    if not ss.play:
        pf, _, _ = sg.score_history(bundle_live, pd.DataFrame(ss.live + [row]))
        pf = sg.apply_policies(pf, cfg_live)
        pr = pf.iloc[-1]
        pc1, pc2, pc3 = st.columns(3)
        delta = f"{(pr['calibrated'] - pf.iloc[-2]['calibrated']) * 100:+.1f} points vs last hour" if len(pf) > 1 else None
        pc1.metric("Preview: estimated chance", f"{pr['calibrated'] * 100:.1f}%", delta, delta_color="inverse")
        pc2.metric("Preview: tier / trend", f"{pr['tier']} / {str(pr['trend']).title()}")
        pc3.metric("Preview: alert", "🔔 would be raised" if bool(pr["policy_B"]) else
                   ("warm-up (none yet)" if bool(pr["warmup"]) else "no new alert"))
    add = st.button("➕ Add this hour and check", type="primary", use_container_width=True)
    if add:
        new_pri = live_add([row])

    demo_p = os.path.join(OUT, "demo_patients.csv")
    if os.path.exists(demo_p):
        with st.expander("No data to hand? Feed hours from a real held-out patient", expanded=True):
            demo = pd.read_csv(demo_p)
            demo = sg.standardise(demo)
            dpid = st.selectbox("Demo patient", sorted(demo[sg.ID].astype(str).unique()))
            dd = demo[demo[sg.ID].astype(str) == dpid].sort_values(sg.TIME).reset_index(drop=True)
            nfeed = st.select_slider("Hours to feed", options=[1, 3, 6, 12], value=1)
            if ss.play and len(ss.live) < len(dd):      # auto-play: add ONE hour per refresh
                j = len(ss.live)
                r1 = dd.iloc[j].to_dict()
                r1.update({sg.ID: "live", sg.TIME: j + 1, sg.LABEL: 0})
                new_pri = live_add([r1])
            elif ss.play:
                ss.play = False
                st.info("Auto-play finished: that stay has no more hours.")
            pa, pb = st.columns(2)
            if pa.button("▶ Auto-play this patient", use_container_width=True, disabled=ss.play):
                ss.play = True
                st.rerun()
            pb.button("⏸ Stop auto-play", on_click=stop_play, use_container_width=True, disabled=not ss.play)
            ss.play_speed = st.select_slider("Seconds per hour (auto-play)", options=[1.5, 2.0, 3.0, 5.0], value=2.0)
            if ss.play:
                st.progress(min(1.0, len(ss.live) / max(1, len(dd))), text=f"Hour {len(ss.live)} of {len(dd)}")
            if st.button("⏩ Feed next hour(s) from this patient"):
                n0 = len(ss.live)
                rows = []
                for j in range(n0, min(n0 + nfeed, len(dd))):
                    r = dd.iloc[j].to_dict()
                    r.update({sg.ID: "live", sg.TIME: j + 1, sg.LABEL: 0})
                    rows.append(r)
                if rows:
                    new_pri = live_add(rows)
                else:
                    st.info("That demo stay has no more hours. Press Reset to start again.")

    b1, b2 = st.columns(2)
    b1.button("⏪ Undo last hour", on_click=undo_last, use_container_width=True, disabled=not ss.live)
    b2.button("🔄 Reset (new patient)", on_click=reset_all, use_container_width=True)

    if not ss.live:
        st.info("Add the first hour above to begin.")
        st.stop()

    frame, feats, d = sg.score_history(bundle_live, pd.DataFrame(ss.live))
    frame = sg.apply_policies(frame, cfg_live)
    n = len(frame)
    r = frame.iloc[-1]
    st.markdown(f"### ICU hour {int(r[sg.TIME])}")
    if new_pri == 3:
        st.error("🔔 NEW HIGH-PRIORITY ALERT: high chance of sepsis within the next %d hours. Check this patient now." % H)
    elif new_pri == 2:
        st.warning("🔔 NEW ALERT: risk looks high but some data is old or missing. Verify the vital signs first.")
    elif r["warmup"]:
        st.info("Warming up: collecting the first %d hours. No alerts are raised yet." % sg.WARMUP_HOURS)
    elif int(r["priority"]) >= 2:
        st.warning("Still high risk. An alert was already raised, so no repeat alarm.")
    status_cards(r)
    if new_pri is not None and sound_on:
        play_sound(st.empty(), new_pri)
    if int(r["priority"]) >= 2:
        a1, a2 = st.columns([1, 3])
        a1.button("✅ Acknowledge alert", on_click=acknowledge)
        if ss.ack is not None:
            a2.caption(f"Alert acknowledged at ICU hour {ss.ack}. (Demo only: nothing is sent to anyone.)")

    t_over, t_insp, t_what, t_rep = st.tabs(["📈 Overview", "🔎 Inspect any hour", "🎚 What-if", "📄 Report"])

    with t_over:
        left, right = st.columns([3, 2])
        with left:
            st.subheader("Risk score so far")
            risk_chart(frame, cfg_live)
            st.caption("Hover a point for details. Scroll to zoom, drag to pan, double-click to reset. "
                       "Red triangles = alerts raised.")
            st.subheader("Vital signs entered")
            vit_all = [c for c in ["HR", "SBP", "MAP", "Resp", "Temp", "O2Sat"] if c in feats.columns]
            vit = st.multiselect("Vitals to show", vit_all, default=vit_all[:3]) or vit_all[:1]
            vt = feats[vit].copy()
            vt.index = d[sg.TIME].to_numpy()
            st.line_chart(vt)
        with right:
            st.subheader("Alerts raised so far")
            al = frame[frame["policy_B"]]
            if len(al):
                st.dataframe(pd.DataFrame({"ICU hour": al[sg.TIME].astype(int), "alert": al["alert_type"],
                                           "est. chance %": (al["calibrated"] * 100).round(1)}),
                             hide_index=True, use_container_width=True)
            else:
                st.write("None yet.")
            st.subheader("Top model factors right now")
            contrib = sg.explain_row(bundle["model"], feats.iloc[-1], bundle["medians"], bundle["features"], top_k=5)
            contrib["value"] = contrib["value"].round(2)
            contrib["score_change"] = contrib["score_change"].round(3)
            st.dataframe(contrib, hide_index=True, use_container_width=True)
            st.caption("Occlusion: how much the score drops if a feature is replaced by its training median. It "
                       "describes the model, not causes of sepsis.")

    with t_insp:
        if ss.play:
            st.info("Paused while auto-play is running. Press Stop to inspect any hour.")
        else:
            st.caption("Drag the slider to look back at any earlier hour: what the system saw and why.")
            ih = st.slider("ICU hour to inspect", 1, n, n) if n > 1 else 1
            ri = frame.iloc[ih - 1]
            status_cards(ri)
            vit_i = [c for c in ["HR", "SBP", "MAP", "Resp", "Temp", "O2Sat"] if c in feats.columns]
            st.write(pd.DataFrame({"vital (last known value)": vit_i,
                                   "value": [round(float(feats.iloc[ih - 1][c]), 1) if pd.notna(feats.iloc[ih - 1][c]) else None
                                             for c in vit_i]}).T)
            ci = sg.explain_row(bundle["model"], feats.iloc[ih - 1], bundle["medians"], bundle["features"], top_k=6)
            ci["value"] = ci["value"].round(2)
            ci["score_change"] = ci["score_change"].round(3)
            st.dataframe(ci, hide_index=True, use_container_width=True)
            st.bar_chart(ci.set_index("feature")["score_change"])

    with t_what:
        if ss.play:
            st.info("Paused while auto-play is running. Press Stop to try what-if changes.")
        else:
            st.caption("Change the latest hour's numbers with the sliders and watch the estimated risk move. "
                       "This does NOT change the saved record.")
            last = ss.live[-1]
            wcols = st.columns(4)
            wv = {}
            ranges = {"HR": (30, 200), "SBP": (50, 220), "MAP": (30, 150), "DBP": (20, 130),
                      "Resp": (6, 50), "Temp": (34.0, 42.0), "O2Sat": (60, 100)}
            for k, (key, label, default) in enumerate(LIVE_VITALS):
                lo, hi = ranges[key]
                cur = last.get(key)
                cur = float(default) if cur is None or pd.isna(cur) else float(cur)
                cur = min(max(cur, lo), hi)
                if key == "Temp":
                    wv[key] = wcols[k % 4].slider(label, float(lo), float(hi), cur, 0.1, key=f"wi_{key}_{n}")
                else:
                    wv[key] = wcols[k % 4].slider(label, float(lo), float(hi), cur, 1.0, key=f"wi_{key}_{n}")
            alt = [dict(x) for x in ss.live]
            alt[-1] = {**alt[-1], **{k: float(v) for k, v in wv.items()}}
            fa, _, _ = sg.score_history(bundle_live, pd.DataFrame(alt))
            ra = fa.iloc[-1]
            w1, w2, w3 = st.columns(3)
            w1.metric("Chance now (as entered)", f"{r['calibrated']*100:.1f}%")
            w2.metric("Chance with your changes", f"{ra['calibrated']*100:.1f}%",
                      f"{(ra['calibrated'] - r['calibrated'])*100:+.1f} points", delta_color="inverse")
            w3.metric("Tier with your changes", str(ra["tier"]))

    with t_rep:
        st.caption("Download this stay's hour-by-hour results to include in your report.")
        rep = frame[[sg.TIME, "score", "calibrated", "tier", "trend", "reliability", "warmup", "priority",
                     "alert_type", "policy_B"]].rename(columns={"policy_B": "alert_raised"})
        vcols = [c for c in ["HR", "SBP", "MAP", "DBP", "Resp", "Temp", "O2Sat"] if c in d.columns]
        rep = pd.concat([rep, d[vcols].reset_index(drop=True)], axis=1)
        st.dataframe(rep.round(3), hide_index=True, use_container_width=True)
        st.download_button("⬇ Download CSV", rep.round(4).to_csv(index=False).encode("utf-8"),
                           file_name="trustalert_stay.csv", mime="text/csv")
    st.divider()
    st.caption("Research prototype for a course project. Trained on the PhysioNet 2019 sepsis challenge data. "
               "Not validated for clinical use.")
    if ss.play:                                     # wait, then add the next hour
        time.sleep(float(ss.get("play_speed", 2.0)) + (1.3 if new_pri in (2, 3) else 0.0))
        st.rerun()
    st.stop()

st.sidebar.header("Patient data")
options = []
demo_path = os.path.join(OUT, "demo_patients.csv")
if os.path.exists(demo_path):
    options.append("Held-out demo patients (real dataset rows)")
options.append("Upload a CSV with one patient's hourly history")
source = st.sidebar.radio("Source", options)

raw = None
if source.startswith("Held-out"):
    demo = pd.read_csv(demo_path)
    pid = st.sidebar.selectbox("Patient", sorted(demo[sg.ID].astype(str).unique()))
    raw = demo[demo[sg.ID].astype(str) == pid].reset_index(drop=True)
else:
    up = st.sidebar.file_uploader("CSV with the same columns as the training data (one row per hour)", type="csv")
    if up is not None:
        raw = pd.read_csv(up)
        if sg.ID in raw.columns and raw[sg.ID].nunique() > 1:
            pid = st.sidebar.selectbox("Patient", sorted(raw[sg.ID].astype(str).unique()))
            raw = raw[raw[sg.ID].astype(str) == pid].reset_index(drop=True)

if raw is None:
    st.info("Choose a patient in the sidebar to begin.")
    st.stop()

try:
    frame, feats, d = sg.score_history(bundle, raw)
except Exception as e:
    st.error(f"Could not score this file: {e}")
    st.stop()

frame = sg.apply_policies(frame, cfg)     # adds policy_A / policy_B / policy_C alert flags
hmin, hmax = int(frame[sg.TIME].min()), int(frame[sg.TIME].max())

# ------------------------------------------------------------------ REPLAY MODE
if mode.startswith("Replay"):
    st.subheader("Replay: the alert system hour by hour")
    st.caption("This steps through a stored patient stay as if the hours were arriving live. At each hour the "
               "system sees only data up to that hour.")
    speed = st.sidebar.select_slider("Seconds per hour", options=[0.3, 0.5, 1.0, 2.0], value=1.0)
    start_h = st.sidebar.slider("Start at ICU hour", hmin, hmax, hmin)
    go = st.sidebar.button("▶ Start replay")
    if not go:
        st.info("Press **Start replay** in the sidebar.")
        st.stop()

    cards, chart, snd, logbox = st.empty(), st.empty(), st.empty(), st.empty()
    log = []
    for j in np.flatnonzero(frame[sg.TIME].to_numpy() >= start_h):
        r = frame.iloc[j]
        with cards.container():
            st.markdown(f"**ICU hour {int(r[sg.TIME])}**")
            status_cards(r)
        risk_chart(frame.iloc[: j + 1], cfg, slot=chart)
        if bool(r["policy_B"]):                          # an alert is RAISED this hour
            log.append({"ICU hour": int(r[sg.TIME]), "alert": r["alert_type"],
                        "est. chance %": round(float(r["calibrated"]) * 100, 1),
                        "trend": r["trend"], "reliability %": round(float(r["reliability"]) * 100)})
            logbox.dataframe(pd.DataFrame(log), hide_index=True, use_container_width=True)
            if sound_on:
                play_sound(snd, r["priority"])
        else:
            snd.empty()
        time.sleep(speed)
    st.success(f"Replay finished. Alerts raised: {len(log)} (a naive 'alert every high hour' rule would raise "
               f"{int(frame['policy_A'].sum())}).")
    if sg.LABEL in d.columns and d[sg.LABEL].max() > 0:
        st.caption(f"In this stay the dataset's sepsis label turns on at ICU hour "
                   f"{int(d.loc[d[sg.LABEL] == 1, sg.TIME].min())}.")
    st.stop()

# ------------------------------------------------------------------ MANUAL MODE
hour = st.sidebar.slider("ICU hour to assess (uses only data up to this hour)", hmin, hmax, hmax)
i = int(np.where(frame[sg.TIME].to_numpy() == hour)[0][-1])
row = frame.iloc[i]
status_cards(row)
st.caption("Priority 3 = high risk with reliable data - 2 = high risk, verify the data first - 1 = moderate risk "
           "and rising - 0 = no alert. Tier, trend and reliability cut-offs were learned from validation data; the "
           "reliability weights are a design choice, not a clinical standard.")
if bool(row["policy_B"]):
    st.info("An alert is raised at this hour by TrustAlert.")
    if sound_on and st.button("🔊 Play this alert's sound"):
        play_sound(st.empty(), row["priority"])

left, right = st.columns([3, 2])
with left:
    st.subheader("Model score over the ICU stay")
    risk_chart(frame, cfg)
    n_a, n_b = int(frame["policy_A"].sum()), int(frame["policy_B"].sum())
    st.caption(f"For this patient: a naive 'alert every high hour' rule would raise {n_a} alerts; TrustAlert "
               f"(episode + quiet period) raises {n_b}.")
    st.subheader("Vital signs (last known value carried forward)")
    vit = [c for c in ["HR", "SBP", "MAP", "Resp", "Temp", "O2Sat"] if c in feats.columns]
    vt = feats[vit].copy()
    vt.index = d[sg.TIME].to_numpy()
    st.line_chart(vt)

with right:
    st.subheader(f"Top model factors at hour {hour}")
    contrib = sg.explain_row(bundle["model"], feats.iloc[i], bundle["medians"], bundle["features"], top_k=6)
    contrib["value"] = contrib["value"].round(2)
    contrib["score_change"] = contrib["score_change"].round(3)
    st.dataframe(contrib, hide_index=True, use_container_width=True)
    st.bar_chart(contrib.set_index("feature")["score_change"])
    st.caption("Occlusion: how much the score drops if a feature is replaced by its training median. It shows what "
               "the trained model uses - not causes of sepsis and not clinical importance.")
    st.subheader("Why the data reliability is what it is")
    st.write(pd.DataFrame({
        "component": ["coverage (vitals available)", "freshness (recent measurements)", "plausibility (values in range)"],
        "score": [row["coverage"], row["freshness"], row["plausibility"]]}).round(2))
    if sg.LABEL in d.columns:
        st.caption(f"Dataset label at this hour (demo comparison only): {int(d[sg.LABEL].iloc[i])}")

mpath = os.path.join(OUT, "metrics.json")
if os.path.exists(mpath):
    with st.expander("How was this evaluated? (held-out test patients, computed in the notebook)"):
        m = json.load(open(mpath))
        t = m["test_metrics"][m["best_model"]]
        st.write(f"**Model:** {m['best_model']} - **horizon:** {m['config']['horizon_hours']} h - "
                 "**split:** by patient (train / validation / test)")
        st.write(pd.DataFrame({"ROC-AUC": [t["roc_auc"]], "PR-AUC": [t["pr_auc"]],
                               "Sensitivity": [t["recall_sensitivity"]], "Precision": [t["precision"]],
                               "Specificity": [t["specificity"]], "F1": [t["f1"]]}).round(3))
        st.write("Alert-policy comparison on test patients:")
        st.write(pd.DataFrame(m["policy_comparison"]).T)

st.divider()
st.caption("Research prototype for a Concepts of Machine Learning course project. Trained on the PhysioNet 2019 "
           "sepsis challenge data. Not validated for clinical use.")
