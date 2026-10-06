"""Shared baseline and temporal patient replay dashboard."""
from pathlib import Path
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sepsis.serving import load_bundle, predict, model_metadata
from sepsis.schema import FEATURE_COLUMNS


@st.cache_resource
def cached_bundle(path, modified):
    return load_bundle(path)


@st.cache_data(show_spinner=False)
def trajectory(path, modified, history):
    bundle = cached_bundle(path, modified)
    return pd.DataFrame([{"Hour": i, "Probability": predict(bundle, history[:i], explain=False)["risk_probability"]}
                         for i in range(1, len(history) + 1)])


def main():
    st.set_page_config(page_title="ICU Sepsis Research", page_icon="🩺", layout="wide")
    st.title("Continuous ICU monitoring")
    st.caption("Made by [bhavya605](https://github.com/bhavya605) · Research demonstration")
    st.caption("Hourly patient replay · Early-warning research · Model explanations")
    st.warning("Research prototype. Scores are not validated for patient care.")
    with st.sidebar:
        st.header("Patient replay")
        artifacts = sorted([*Path("artifacts").rglob("*.joblib"), *Path("artifacts").rglob("*.pt")])
        artifact = st.selectbox("Trained model", artifacts, format_func=lambda p: str(p.relative_to("artifacts"))) if artifacts else None
        uploaded = st.file_uploader("Patient PSV or CSV", type=["psv", "csv"])
        demo_files = sorted(Path("data/demo").glob("*.psv"))
        demo_path = st.selectbox("Synthetic example", demo_files, format_func=lambda p: p.stem) if demo_files else None
        st.caption("Each row represents the next consecutive hour. Training labels are excluded from inputs.")
    if artifact is None:
        st.info("No model found. Follow the demo commands in README.md to train one.")
        return
    if uploaded is None and demo_path is None:
        st.info("Upload a patient record or generate synthetic examples.")
        return
    try:
        frame = pd.read_csv(uploaded if uploaded is not None else demo_path,
                            sep="," if uploaded is not None and uploaded.name.endswith(".csv") else "|")
        missing = set(FEATURE_COLUMNS) - set(frame.columns)
        extra = set(frame.columns) - set(FEATURE_COLUMNS) - {"SepsisLabel"}
        if missing or extra:
            raise ValueError(f"Invalid columns. Missing: {sorted(missing)}; unexpected: {sorted(extra)}")
        values = frame[list(FEATURE_COLUMNS)].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
        if not 1 <= len(values) <= 1000 or np.isinf(values).any():
            raise ValueError("Record must have 1–1000 hourly rows and finite observed readings")
        los = values[:, list(FEATURE_COLUMNS).index("ICULOS")]
        if np.isfinite(los).all() and len(los) > 1 and not np.all(np.diff(los) == 1):
            raise ValueError("ICULOS must advance by one hour per row")
        path, modified = str(artifact.resolve()), artifact.stat().st_mtime_ns
        bundle = cached_bundle(path, modified)
    except Exception as exc:
        st.error(f"Unable to load record or model: {exc}")
        return
    hour = st.slider("Replay through record hour", 1, len(values), min(12, len(values))) if len(values) > 1 else 1
    try:
        output = predict(bundle, values[:hour])
        risks = trajectory(path, modified, values[:hour])
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        return
    if output.get("source_type") == "synthetic_demo" or uploaded is None:
        st.info("Synthetic demonstration: the patient or model uses generated data. Scores do not establish medical performance.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model probability", f"{output['risk_probability']:.1%}")
    c2.metric("Persistent alert", "Active" if output["alert"] else "Inactive")
    c3.metric("Threshold", f"{output['threshold']:.1%}")
    c4.metric("Historical context", f"{bundle.window_hours} hours")
    if output.get("calibrated"):
        raw_probability = output.get("raw_probability", output.get("raw_risk_probability"))
        st.caption(f"Calibrated probability shown. Raw model probability: {raw_probability:.1%}. Explanations describe the raw model calculation.")
    st.caption(f"Target: published label {bundle.horizon_hours} additional hours ahead. Labels already precede recorded onset by six hours. Alerts require {bundle.persistence_hours} consecutive threshold crossings.")
    st.subheader("Hourly risk trajectory")
    chart = alt.Chart(risks).mark_line(color="#176b87", point=True).encode(
        x=alt.X("Hour:Q", title="Record hour"), y=alt.Y("Probability:Q", scale=alt.Scale(domain=[0, 1])),
        tooltip=["Hour", alt.Tooltip("Probability:Q", format=".1%")])
    threshold = alt.Chart(pd.DataFrame({"Threshold": [output["threshold"]]})).mark_rule(color="#c05246", strokeDash=[5, 4]).encode(y="Threshold:Q")
    st.altair_chart(chart + threshold, width="stretch")
    trends, explanations = st.tabs(["Patient readings", "Model explanations"])
    with trends:
        selected = st.multiselect("Plot observed readings", ["HR", "MAP", "SBP", "Temp", "Resp", "O2Sat", "Lactate", "Creatinine"], default=["HR", "MAP"])
        if selected:
            st.line_chart(frame.iloc[:hour][selected].set_axis(range(1, hour + 1)), x_label="Record hour")
        st.dataframe(frame.iloc[[hour - 1]][list(FEATURE_COLUMNS)], hide_index=True, width="stretch")
        st.caption("Blank cells indicate missing readings. Imputation uses past readings and training-derived fallback values.")
    with explanations:
        rows = pd.DataFrame(output.get("feature_contributions", []))
        if not rows.empty:
            st.dataframe(rows.head(20), hide_index=True, width="stretch")
        shap_details = output.get("shap_explanation")
        if shap_details:
            st.caption("SHAP attributes the raw model output relative to training-window backgrounds. Calibration, when present, transforms the final probability separately.")
            with st.expander("SHAP details"):
                st.json(shap_details)
        temporal_rows = pd.DataFrame(output.get("temporal_contributions", []))
        if not temporal_rows.empty and {"feature", "lag", "contribution"} <= set(temporal_rows):
            st.altair_chart(alt.Chart(temporal_rows).mark_rect().encode(
                x="lag:O", y="feature:N", color=alt.Color("contribution:Q", scale=alt.Scale(scheme="redblue", domainMid=0)),
                tooltip=["feature", "lag", "contribution"]), width="stretch")
        st.caption(f"Method: {output.get('explanation_method', output.get('contribution_scale', 'model attribution'))}. Contributions describe the model calculation rather than clinical causation.")
        mask_rows = pd.DataFrame(output.get("missingness_feature_time_contributions", []))
        if not mask_rows.empty:
            st.subheader("Measurement availability explanation")
            st.altair_chart(alt.Chart(mask_rows).mark_rect().encode(
                x="lag:O", y="feature:N",
                color=alt.Color("contribution:Q", scale=alt.Scale(scheme="redblue", domainMid=0)),
                tooltip=["feature", "lag", "contribution", "observed"]), width="stretch")
            st.caption("Values stay fixed while measurement flags vary from an all-missing reference to the recorded flags. This is a separate raw-model comparison; do not add it to the value explanations.")
    with st.expander("Model provenance and prediction metadata"):
        st.json(model_metadata(bundle))
        st.json({k: v for k, v in output.items() if k not in {"feature_contributions", "temporal_contributions", "feature_time_contributions", "missingness_feature_time_contributions"}})
