from pathlib import Path
import os

PYCOX_CACHE_DIR = "/tmp/pycox_data"
os.makedirs(PYCOX_CACHE_DIR, exist_ok=True)
os.environ["PYCOX_DATA_DIR"] = PYCOX_CACHE_DIR

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torchtuples as tt
from pycox.models import CoxPH

# =========================
# 1. Paths and page config
# =========================
BASE_DIR = Path(__file__).resolve().parent

ARTIFACT_DIR = BASE_DIR / "deepsurv_artifacts"
if not ARTIFACT_DIR.exists():
    ARTIFACT_DIR = BASE_DIR

st.set_page_config(
    page_title="DeepSurv-based Survival Prediction Model for Glioblastoma",
    layout="wide",
)

# =========================
# 2. Global style
# =========================
st.markdown(
    """
    <style>
    .main {
        background-color: #ffffff;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    h1, h2, h3 {
        color: #1f2d3d;
        font-family: Arial, Helvetica, sans-serif;
    }

    p, div, label, span {
        color: #2f3b4a;
        font-family: Arial, Helvetica, sans-serif;
    }

    .panel-title {
        font-size: 1.55rem;
        font-weight: 700;
        color: #1f2d3d;
        margin-top: 0.5rem;
        margin-bottom: 0.9rem;
    }

    .result-section-title {
        font-size: 1.7rem;
        font-weight: 700;
        color: #1f2d3d;
        margin-top: 0.9rem;
        margin-bottom: 0.95rem;
    }

    .metric-card {
        background: linear-gradient(180deg, #fbfdff 0%, #f5f9fc 100%);
        border: 1px solid #d6e4f0;
        border-radius: 14px;
        padding: 0.95rem 0.8rem;
        text-align: center;
        box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
        min-height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    .metric-value {
        font-size: 1.9rem;
        font-weight: 700;
        color: #163a63;
        line-height: 1.2;
        margin-bottom: 0.3rem;
    }

    .metric-label {
        font-size: 0.95rem;
        color: #486581;
        font-weight: 600;
    }

    .instruction-box {
        background: #fcfdff;
        border: 1px solid #d8e3ec;
        border-radius: 14px;
        padding: 1rem 1.05rem 0.95rem 1.05rem;
        margin-top: 0.15rem;
        box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
    }

    .instructions-title {
        margin-bottom: 0.45rem;
    }

    .instructions-title h3 {
        font-size: 1.2rem;
        font-weight: 700;
        color: #1f2d3d;
        margin: 0;
    }

    .instructions-text {
        font-size: 0.97rem;
        line-height: 1.7;
        color: #334e68;
    }

    .note-text {
        color: #5b6b7a;
        font-size: 0.93rem;
        line-height: 1.6;
    }

    .small-help {
        color: #5b6b7a;
        font-size: 0.95rem;
        line-height: 1.6;
    }

    label, .stSelectbox label, .stNumberInput label {
        font-size: 1rem !important;
        font-weight: 600;
    }

    div[data-baseweb="select"] > div {
        font-size: 0.98rem !important;
    }

    input {
        font-size: 0.98rem !important;
    }

    div.stButton > button:first-child {
        width: 180px;
        min-width: 180px;
        border-radius: 10px;
        border: 1px solid #c62828;
        background-color: #d32f2f;
        color: #ffffff;
        font-weight: 700;
        padding: 0.62rem 1rem;
        display: inline-block;
    }

    div.stButton > button:first-child:hover {
        background-color: #b71c1c;
        border-color: #b71c1c;
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# 3. Asset loading
# =========================
def _load_pickle(name: str):
    path = ARTIFACT_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return joblib.load(path)

@st.cache_resource
def load_assets():
    preprocessor = _load_pickle("preprocessor.pkl")
    numeric_cols = _load_pickle("numeric_cols.pkl")
    categorical_cols = _load_pickle("categorical_cols.pkl")
    income_quantiles = _load_pickle("income_quantiles.pkl")
    baseline_hazards = _load_pickle("baseline_hazards.pkl")
    category_options = _load_pickle("category_options.pkl")

    for key in ["Radiotherapy", "Chemotherapy"]:
        if key in category_options:
            category_options[key] = [
                "No" if str(v).strip() == "No/Unknown" else str(v)
                for v in category_options[key]
            ]

    dummy_row = {
        "Age": 50.0,
        "Income": 50000.0,
        "Age2": 2500.0,
        "Income_group": 0,
    }
    for col, vals in category_options.items():
        vals = list(vals)
        dummy_row[col] = vals[0] if len(vals) > 0 else "Unknown"

    ordered_cols = list(numeric_cols) + list(categorical_cols)
    dummy_df = pd.DataFrame([dummy_row])[ordered_cols]
    in_features = preprocessor.transform(dummy_df).shape[1]

    net = tt.practical.MLPVanilla(
        in_features=in_features,
        num_nodes=[128, 64, 32],
        out_features=1,
        batch_norm=True,
        dropout=0.10,
    )

    model = CoxPH(net)
    model.load_net(
        str(ARTIFACT_DIR / "deepsurv_net.pt"),
        map_location=torch.device("cpu"),
        weights_only=False,
    )
    model.net.to(torch.device("cpu"))
    model.net.eval()

    if isinstance(baseline_hazards, pd.Series):
        baseline_hazards = baseline_hazards.astype(float).sort_index()
        baseline_cumulative_hazards = baseline_hazards.cumsum()
    else:
        baseline_hazards = pd.Series(baseline_hazards).astype(float).sort_index()
        baseline_cumulative_hazards = baseline_hazards.cumsum()

    model.baseline_hazards_ = baseline_hazards
    model.baseline_cumulative_hazards_ = baseline_cumulative_hazards

    return (
        model,
        preprocessor,
        list(numeric_cols),
        list(categorical_cols),
        np.asarray(income_quantiles, dtype=float),
        category_options,
        baseline_hazards,
    )

# =========================
# 4. Helper functions
# =========================
def calc_income_group(income, quantiles):
    bins = np.array(quantiles, dtype=float).copy()

    for i in range(1, len(bins)):
        if bins[i] <= bins[i - 1]:
            bins[i] = bins[i - 1] + 1e-6

    group = pd.cut(
        pd.Series([income]),
        bins=bins,
        labels=False,
        include_lowest=True,
    ).iloc[0]

    if pd.isna(group):
        if income < bins[0]:
            return 0
        return len(bins) - 2

    return int(group)

def build_patient_df(
    age,
    sex,
    year,
    race,
    laterality,
    eor,
    radiotherapy,
    marital,
    chemotherapy,
    t_site,
    histologic,
    extension,
    income,
):
    income_group = calc_income_group(income, income_quantiles)

    radiotherapy_model_value = "No/Unknown" if radiotherapy == "No" else radiotherapy
    chemotherapy_model_value = "No/Unknown" if chemotherapy == "No" else chemotherapy

    row = {
        "Age": float(age),
        "Income": float(income),
        "Age2": float(age) ** 2,
        "Sex": str(sex),
        "Year": str(year),
        "Race": str(race),
        "Laterality": str(laterality),
        "EOR": str(eor),
        "Radiotherapy": str(radiotherapy_model_value),
        "Marital": str(marital),
        "Chemotherapy": str(chemotherapy_model_value),
        "T_site": str(t_site),
        "Histologic": str(histologic),
        "Extension": str(extension),
        "Income_group": int(income_group),
    }

    ordered_cols = numeric_cols + categorical_cols
    return pd.DataFrame([row])[ordered_cols]

def predict_survival(patient_df):
    x = preprocessor.transform(patient_df[numeric_cols + categorical_cols]).astype("float32")
    surv_raw = model.predict_surv_df(x)
    surv_df = pd.DataFrame(surv_raw)

    if surv_df.shape[1] == 1:
        curve_df = surv_df.copy()
        try:
            curve_df.index = pd.Index(np.asarray(curve_df.index, dtype=float))
        except Exception:
            curve_df.index = pd.Index(np.arange(len(curve_df)), dtype=float)
        curve_df = curve_df.sort_index()
    else:
        curve_df = surv_df.T.copy()
        curve_df.index = pd.Index(np.arange(len(curve_df)), dtype=float)

    curve_df = curve_df.apply(pd.to_numeric, errors="coerce")
    curve_df = curve_df.dropna(how="all")

    if curve_df.shape[0] == 0 or curve_df.shape[1] == 0:
        raise ValueError("No valid survival curve points were returned by the model.")

    curve_series = curve_df.iloc[:, 0].astype(float).dropna()

    if len(curve_series) == 0:
        raise ValueError("All predicted survival probabilities are NaN.")

    curve_series = curve_series.clip(lower=0, upper=1)

    if curve_series.index.min() > 0:
        curve_series.loc[0.0] = 1.0
        curve_series = curve_series.sort_index()

    time_points = np.asarray(curve_series.index, dtype=float)
    values = np.asarray(curve_series.values, dtype=float)

    if np.min(values) > 0.5:
        median_survival_time = np.nan
    else:
        median_survival_time = np.interp(0.5, values[::-1], time_points[::-1])

    def nearest_prob(month):
        idx = np.argmin(np.abs(time_points - float(month)))
        return float(values[idx])

    p6 = nearest_prob(6)
    p12 = nearest_prob(12)
    p24 = nearest_prob(24)

    risk_score = model.predict(x)
    try:
        risk_score = float(np.asarray(risk_score).reshape(-1)[0])
    except Exception:
        risk_score = np.nan

    return curve_series, p6, p12, p24, median_survival_time, x.shape, risk_score

def make_survival_figure(curve_series):
    x_vals = np.asarray(curve_series.index, dtype=float)
    y_vals = np.asarray(curve_series.values, dtype=float)

    mask = x_vals <= 24
    x_plot = x_vals[mask]
    y_plot = y_vals[mask]

    fig, ax = plt.subplots(figsize=(10.6, 5.8), dpi=160)
    ax.step(x_plot, y_plot, where="post", color="#d62728", linewidth=2.2)

    def nearest_point(month):
        idx = np.argmin(np.abs(x_vals - float(month)))
        return x_vals[idx], y_vals[idx]

    x6, y6 = nearest_point(6)
    x12, y12 = nearest_point(12)
    x24, y24 = nearest_point(24)

    ax.scatter([x6, x12, x24], [y6, y12, y24], color="#d62728", s=34, zorder=3)

    ax.set_title(
        "Predicted Survival Probability",
        fontsize=18,
        fontweight="semibold",
        color="#1f2d3d",
        pad=12,
    )
    ax.set_xlabel(
        "Time (months)",
        fontsize=14,
        fontweight="semibold",
        color="#1f2d3d",
    )
    ax.set_ylabel(
        "Survival probability",
        fontsize=14,
        fontweight="semibold",
        color="#1f2d3d",
    )

    ax.set_xlim(0, 24.5)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.tick_params(axis="both", labelsize=12, colors="#243b53")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.1)
    ax.spines["bottom"].set_linewidth(1.1)
    ax.grid(False)

    plt.tight_layout()
    return fig

# =========================
# 5. Load everything
# =========================
try:
    (
        model,
        preprocessor,
        numeric_cols,
        categorical_cols,
        income_quantiles,
        category_options,
        baseline_hazards_loaded,
    ) = load_assets()
except Exception as e:
    st.error(f"Failed to load model assets: {e}")
    st.stop()

# =========================
# 6. Page title
# =========================
st.title("DeepSurv-based Survival Prediction Model for Glioblastoma")

left_col, right_col = st.columns([1.0, 2.35])

default_year = (
    category_options["Year"][0]
    if "Year" in category_options and len(category_options["Year"]) > 0
    else "Unknown"
)

default_t_site = (
    category_options["T_site"][0]
    if "T_site" in category_options and len(category_options["T_site"]) > 0
    else "Unknown"
)

default_histologic = (
    category_options["Histologic"][0]
    if "Histologic" in category_options and len(category_options["Histologic"]) > 0
    else "Unknown"
)

# =========================
# 7. Left input panel
# =========================
with left_col:
    st.markdown('<div class="panel-title">Patient Input</div>', unsafe_allow_html=True)

    age = st.number_input("Age (years)", min_value=18.0, max_value=120.0, value=66.0, step=1.0)
    income = st.number_input(
        "County-level median household income (USD)",
        min_value=0.0,
        value=50000.0,
        step=1000.0
    )

    sex = st.selectbox("Sex", category_options["Sex"])
    race = st.selectbox("Race", category_options["Race"])
    laterality = st.selectbox("Laterality", category_options["Laterality"])
    eor = st.selectbox("Extent of resection", category_options["EOR"])
    radiotherapy = st.selectbox("Radiotherapy", category_options["Radiotherapy"])
    marital = st.selectbox("Marital status", category_options["Marital"])
    chemotherapy = st.selectbox("Chemotherapy", category_options["Chemotherapy"])
    extension = st.selectbox("Tumor extension", category_options["Extension"])

    predict_btn = st.button("Predict")

# =========================
# 8. Prediction
# =========================
if "pred_done" not in st.session_state:
    st.session_state.pred_done = False

if predict_btn or not st.session_state.pred_done:
    try:
        patient_df = build_patient_df(
            age,
            sex,
            default_year,
            race,
            laterality,
            eor,
            radiotherapy,
            marital,
            chemotherapy,
            default_t_site,
            default_histologic,
            extension,
            income,
        )
        curve_series, p6, p12, p24, median_survival_time, x_shape, risk_score = predict_survival(patient_df)

        st.session_state.pred_done = True
        st.session_state.patient_df = patient_df
        st.session_state.curve_series = curve_series
        st.session_state.p6 = p6
        st.session_state.p12 = p12
        st.session_state.p24 = p24
        st.session_state.median_survival_time = median_survival_time
        st.session_state.x_shape = x_shape
        st.session_state.risk_score = risk_score
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

# =========================
# 9. Right result panel
# =========================
with right_col:
    curve_series = st.session_state.curve_series
    p6 = st.session_state.p6
    p12 = st.session_state.p12
    p24 = st.session_state.p24

    fig = make_survival_figure(curve_series)
    st.pyplot(fig)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="result-section-title">Predicted Probabilities</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{p6 * 100:.1f}%</div>
                <div class="metric-label">6-month survival</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{p12 * 100:.1f}%</div>
                <div class="metric-label">12-month survival</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{p24 * 100:.1f}%</div>
                <div class="metric-label">24-month survival</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="instruction-box">
            <div class="instructions-title"><h3>Instructions</h3></div>
            <div class="instructions-text">
                1. Select the patient's information in the left panel.<br>
                2. Click <strong>Predict</strong>.<br>
                3. Review the survival curve and predicted probabilities.
            </div>
            <div style="height: 10px;"></div>
            <div class="note-text">
                <strong>Note:</strong> This model is intended for research use only, and predictive accuracy is not guaranteed.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
