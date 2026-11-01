from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from safety_model import (
    CLASS_COLORS,
    CLASS_NAMES,
    FEATURE_COLUMNS,
    build_model,
    classify_reading,
    generate_reading,
)

st.set_page_config(
    page_title="SchoolWatch AI | Safety Monitor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { --ink:#172b3a; --muted:#6b7b88; --line:#e7edf0; --teal:#168568; }
    html, body, [class*="css"] { font-family:Arial,sans-serif; color:var(--ink); }
    .stApp { background:#f6f8f8; }
    [data-testid="stSidebar"] { background:#102a38; border-right:0; }
    [data-testid="stSidebar"] * { color:#e7f0f2 !important; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label { padding:7px 10px; }
    .brand { display:flex; align-items:center; gap:11px; margin:4px 0 34px; }
    .brand-mark { width:42px;height:42px;border-radius:14px;background:#1a8b73;display:grid;place-items:center;font-size:21px; }
    .brand-name { color:white;font:800 17px Arial,sans-serif;letter-spacing:-.4px; }
    .brand-sub { color:#9fb5bd;font-size:11px;letter-spacing:1.6px;text-transform:uppercase; }
    .eyebrow { color:#168568; font-size:11px; font-weight:700; letter-spacing:1.5px; text-transform:uppercase; margin-bottom:8px; }
    .page-title { font:800 30px Arial,sans-serif; letter-spacing:-1.1px; margin:0; color:#172b3a; }
    .page-subtitle { color:#71808a; margin:5px 0 24px; font-size:14px; }
    .panel { background:white;border:1px solid #e8edef;border-radius:16px;padding:20px 22px; }
    .panel-title { font:700 15px Arial,sans-serif;margin:0 0 14px;color:#203746; }
    .status-card { border-radius:17px;padding:22px 24px;color:white;margin-bottom:16px; }
    .status-card.safe { background:linear-gradient(120deg,#147c67,#249b79); }
    .status-card.alert { background:linear-gradient(120deg,#b87919,#df9b32); }
    .status-card.emergency { background:linear-gradient(120deg,#a94444,#d45d57); }
    .status-card .small { opacity:.78;font-size:11px;letter-spacing:1.5px;text-transform:uppercase;font-weight:700; }
    .status-card .big { font:800 25px Arial,sans-serif;margin:5px 0 3px; }
    .status-card .detail { opacity:.88;font-size:13px; }
    .note { background:#eef5f3;border:1px solid #dcebe6;border-radius:12px;padding:12px 14px;color:#45655f;font-size:12px; }
    .stButton>button { border-radius:10px;font-weight:700; }
    div[data-testid="stMetric"] { background:white;border:1px solid #e8edef;border-radius:13px;padding:14px 16px; }
    div[data-testid="stMetricLabel"] { color:#71808a; }
    .stTabs [data-baseweb="tab"] { font-weight:600; }
    footer { visibility:hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def load_model():
    return build_model()


model, model_metrics, training_data = load_model()

if "history" not in st.session_state:
    st.session_state.history = []
if "active_alert" not in st.session_state:
    st.session_state.active_alert = False
if "stream_step" not in st.session_state:
    st.session_state.stream_step = 0


def add_assessment(reading: dict[str, float | int | str]) -> dict[str, object]:
    result = classify_reading(model, reading)
    status = str(result["status"])
    if status == "Safe":
        action = "CLEAR: Previous alert closed" if st.session_state.active_alert else "SAFE: No active alert"
        st.session_state.active_alert = False
    elif status == "Normal Alert":
        action = "NORMAL ALERT: Inspection required"
        st.session_state.active_alert = True
    else:
        action = "HIGH-PRIORITY: Immediate human review"
        st.session_state.active_alert = True

    st.session_state.stream_step += 1
    record = {
        "step": st.session_state.stream_step,
        "time": datetime.now().strftime("%H:%M:%S"),
        "incident_type": str(reading.get("incident_type", "Manual assessment")),
        "temperature_c": float(reading["temperature_c"]),
        "smoke_level": float(reading["smoke_level"]),
        "co_ppm": float(reading["co_ppm"]),
        "gas_level": float(reading["gas_level"]),
        "sound_db": float(reading["sound_db"]),
        "crowd_count": int(reading["crowd_count"]),
        "motion_index": float(reading["motion_index"]),
        "flame_detected": int(reading["flame_detected"]),
        "event_duration_min": float(reading["event_duration_min"]),
        "predicted_status": status,
        "model_score": float(result["scores"][status]),
        "system_action": action,
    }
    st.session_state.history.insert(0, record)
    return result


def page_header(eyebrow: str, title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="eyebrow">{eyebrow}</div><h1 class="page-title">{title}</h1>'
        f'<div class="page-subtitle">{subtitle}</div>',
        unsafe_allow_html=True,
    )


def status_style(status: str) -> str:
    if status == "Safe":
        return "safe"
    if status == "Normal Alert":
        return "alert"
    return "emergency"


with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="brand-mark">🛡️</div><div>'
        '<div class="brand-name">SchoolWatch AI</div>'
        '<div class="brand-sub">Safety intelligence</div></div></div>',
        unsafe_allow_html=True,
    )
    st.caption("MONITORING CONSOLE")
    page = st.radio(
        "Navigation",
        ["Overview", "Assess a reading", "Live simulation", "Model insights", "Alert history"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("**Prototype status**")
    st.caption("●  Demo model ready")
    st.caption("Synthetic sensor data · No live devices")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div class="note">For academic demonstration only. This prototype is not an '
        'operational life-safety or emergency-dispatch system.</div>',
        unsafe_allow_html=True,
    )

if page == "Overview":
    page_header("Campus safety · prototype", "Good day. Here’s the safety overview.", "A model-powered demonstration of sensor-based campus monitoring.")
    history = pd.DataFrame(st.session_state.history)
    latest = history.iloc[0] if not history.empty else None
    current_status = str(latest["predicted_status"]) if latest is not None else "Safe"
    if latest is None:
        card_detail = "No assessments yet. Start with a sample reading or run the live simulation."
    else:
        card_detail = f'{latest["system_action"]} · Last reading at {latest["time"]}'
    icon = {"Safe": "✓", "Normal Alert": "!", "High-Priority Emergency": "!"}[current_status]
    st.markdown(
        f'<div class="status-card {status_style(current_status)}"><div class="small">{icon} &nbsp; Current model status</div>'
        f'<div class="big">{current_status}</div><div class="detail">{card_detail}</div></div>',
        unsafe_allow_html=True,
    )

    total = len(history)
    alert_count = int((history["predicted_status"] != "Safe").sum()) if total else 0
    emergency_count = int((history["predicted_status"] == "High-Priority Emergency").sum()) if total else 0
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Readings assessed", total)
    col2.metric("Alerts in history", alert_count)
    col3.metric("High-priority", emergency_count)
    col4.metric("Demo model accuracy", f'{model_metrics["accuracy"]:.0%}')
    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([1.5, 1])
    with left:
        st.markdown('<div class="panel"><div class="panel-title">Recent monitoring activity</div>', unsafe_allow_html=True)
        if history.empty:
            st.info("Your alert history is empty. Use **Assess a reading** to create the first prediction.")
        else:
            chart_data = history.head(12).iloc[::-1].set_index("time")[["temperature_c", "smoke_level", "gas_level"]]
            st.line_chart(chart_data, height=235, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel"><div class="panel-title">Quick actions</div>', unsafe_allow_html=True)
        st.write("Explore the prototype workflow:")
        st.markdown("- **Assess a reading** with editable sensor inputs\n- **Live simulation** with generated scenarios\n- **Model insights** and test-set metrics")
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div class="note"><strong>Important:</strong> The scores and readings are produced from a '
        'synthetic dataset for learning. They must not be used to make real-world safety decisions.</div>',
        unsafe_allow_html=True,
    )

elif page == "Assess a reading":
    page_header("Manual assessment", "Evaluate a sensor snapshot", "Enter a sample of sensor readings and see the model’s three-class result.")
    with st.form("assessment_form"):
        st.markdown('<div class="panel-title">Environmental indicators</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            temperature = st.slider("Temperature", 15.0, 85.0, 29.0, 0.5, format="%.1f °C")
            smoke = st.slider("Smoke level", 0.0, 100.0, 8.0, 1.0)
            co_ppm = st.slider("CO level", 0.0, 100.0, 5.0, 1.0, format="%.0f ppm")
        with c2:
            gas = st.slider("Gas level", 0.0, 120.0, 18.0, 1.0)
            sound = st.slider("Sound level", 25.0, 130.0, 60.0, 1.0, format="%.0f dB")
            crowd = st.slider("Crowd count", 0, 120, 25, 1)
        with c3:
            motion = st.slider("Motion index", 0.0, 100.0, 35.0, 1.0)
            duration = st.slider("Event duration", 0.1, 30.0, 1.0, 0.1, format="%.1f min")
            flame = st.selectbox("Flame detector", ["No flame", "Flame detected"])
        submitted = st.form_submit_button("Analyze sensor reading", type="primary", use_container_width=True)
    if submitted:
        reading = {
            "temperature_c": temperature,
            "smoke_level": smoke,
            "co_ppm": co_ppm,
            "gas_level": gas,
            "sound_db": sound,
            "crowd_count": crowd,
            "motion_index": motion,
            "flame_detected": int(flame == "Flame detected"),
            "event_duration_min": duration,
            "incident_type": "Manual assessment",
        }
        result = add_assessment(reading)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f'<div class="status-card {status_style(str(result["status"]))}"><div class="small">Prediction</div>'
            f'<div class="big">{result["status"]}</div><div class="detail">{st.session_state.history[0]["system_action"]}</div></div>',
            unsafe_allow_html=True,
        )
        st.caption("Model scores are illustrative outputs from simulated training data, not calibrated risk probabilities.")
        score_cols = st.columns(3)
        for col, status in zip(score_cols, CLASS_NAMES.values()):
            col.metric(status, f'{result["scores"][status]:.1%}')

elif page == "Live simulation":
    page_header("Simulated sensor stream", "Run a safety scenario", "Generate a synthetic sensor snapshot and send it through the trained model.")
    st.markdown(
        '<div class="note">This is a deterministic demo workflow, not a connection to a camera, sensor, '
        'or emergency service. Every result is simulated.</div><br>',
        unsafe_allow_html=True,
    )
    scenario = st.selectbox(
        "Choose a scenario",
        ["Routine activity", "Minor smoke", "Crowd disturbance", "Possible gas issue", "Fire emergency", "Severe heat event"],
    )
    if st.button("Generate reading & analyze", type="primary"):
        reading = generate_reading(scenario, seed=st.session_state.stream_step + 140)
        result = add_assessment(reading)
        st.markdown("<br>", unsafe_allow_html=True)
        left, right = st.columns([1, 1.4])
        with left:
            st.markdown(
                f'<div class="status-card {status_style(str(result["status"]))}"><div class="small">Model result</div>'
                f'<div class="big">{result["status"]}</div><div class="detail">{st.session_state.history[0]["system_action"]}</div></div>',
                unsafe_allow_html=True,
            )
        with right:
            st.markdown('<div class="panel"><div class="panel-title">Generated sensor snapshot</div>', unsafe_allow_html=True)
            sensor_items = [
                ("Temperature", "temperature_c", "°C"),
                ("Smoke", "smoke_level", ""),
                ("CO", "co_ppm", "ppm"),
                ("Gas", "gas_level", ""),
                ("Sound", "sound_db", "dB"),
                ("Crowd", "crowd_count", ""),
                ("Motion", "motion_index", ""),
                ("Flame", "flame_detected", ""),
                ("Duration", "event_duration_min", "min"),
            ]
            for offset in range(0, len(sensor_items), 3):
                sensor_columns = st.columns(3)
                for col, (label, key, unit) in zip(sensor_columns, sensor_items[offset : offset + 3]):
                    value = reading[key]
                    if key == "flame_detected":
                        display_value = "Detected" if value else "No"
                    elif key == "crowd_count":
                        display_value = str(int(value))
                    else:
                        display_value = f"{float(value):.1f}{unit}"
                    col.metric(label, display_value)
            st.markdown("</div>", unsafe_allow_html=True)
        st.caption("Use the history page to review and export this assessment.")

elif page == "Model insights":
    page_header("Machine learning", "Understand the demo model", "Multiclass Logistic Regression with feature scaling and a stratified held-out test set.")
    report = model_metrics["report"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test accuracy", f'{model_metrics["accuracy"]:.1%}')
    c2.metric("Test examples", str(model_metrics["test_size"]))
    c3.metric("Training examples", f"{len(training_data) - int(model_metrics['test_size'])}")
    c4.metric("Input features", str(len(FEATURE_COLUMNS)))
    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([1.05, 1])
    with left:
        st.markdown('<div class="panel"><div class="panel-title">Per-class performance</div>', unsafe_allow_html=True)
        rows = []
        for label in CLASS_NAMES.values():
            rows.append(
                {
                    "Class": label,
                    "Precision": report[label]["precision"],
                    "Recall": report[label]["recall"],
                    "F1 score": report[label]["f1-score"],
                }
            )
        st.dataframe(pd.DataFrame(rows).set_index("Class").style.format("{:.2f}"), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel"><div class="panel-title">Confusion matrix</div>', unsafe_allow_html=True)
        matrix = pd.DataFrame(
            model_metrics["confusion_matrix"],
            index=list(CLASS_NAMES.values()),
            columns=list(CLASS_NAMES.values()),
        )
        st.dataframe(matrix, use_container_width=True)
        st.caption("Rows are actual classes; columns are model predictions.")
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Features used**")
    st.write(", ".join(FEATURE_COLUMNS))
    st.markdown(
        '<div class="note"><strong>Evaluation limitation:</strong> The model was trained and tested on '
        'generated data with labels derived from scenario profiles. Reported metrics show code workflow '
        'only and do not establish real-world accuracy, safety, or readiness.</div>',
        unsafe_allow_html=True,
    )

else:
    page_header("Event log", "Alert history", "Review assessments generated during this session. Export the log for your project report.")
    history = pd.DataFrame(st.session_state.history)
    if history.empty:
        st.info("No readings assessed yet. Run an assessment or create a simulated reading first.")
    else:
        export = history.to_csv(index=False).encode("utf-8")
        st.download_button("Download history as CSV", export, "schoolwatch_alert_history.csv", "text/csv")
        st.markdown("<br>", unsafe_allow_html=True)
        display_columns = [
            "time", "incident_type", "predicted_status", "model_score",
            "temperature_c", "smoke_level", "flame_detected", "system_action",
        ]
        st.dataframe(
            history[display_columns].rename(
                columns={
                    "time": "Time",
                    "incident_type": "Scenario",
                    "predicted_status": "Prediction",
                    "model_score": "Model score",
                    "temperature_c": "Temp °C",
                    "smoke_level": "Smoke",
                    "flame_detected": "Flame",
                    "system_action": "System action",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        if st.button("Clear session history"):
            st.session_state.history = []
            st.session_state.active_alert = False
            st.session_state.stream_step = 0
            st.rerun()
