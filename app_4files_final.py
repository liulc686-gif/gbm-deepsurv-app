from pathlib import Path
import os

PYCOX_CACHE_DIR = "/tmp/pycox_data"
os.makedirs(PYCOX_CACHE_DIR, exist_ok=True)
os.environ["PYCOX_DATA_DIR"] = PYCOX_CACHE_DIR

import numpy as np
import pandas as pd
import streamlit as st
import torch
import joblib
import torchtuples as tt
import plotly.graph_objects as go
from pycox.models import CoxPH

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(page_title="GBM生存预测工具", layout="wide")
st.title("Glioblastoma (GBM) 生存预测在线工具")
st.markdown("请输入患者临床信息以获取实时生存概率预测。")

# 只有 4 个文件时，界面选项必须手动提供。
# 这些选项来自你当前仓库里曾经能跑起来的那版 app.py。
UI_OPTIONS = {
    "Sex": ["Female", "Male"],
    "Race": ["White", "Black", "Asian or P"],
    "Laterality": ["Midline", "Left", "Right"],
    "EOR": ["GTR", "STR", "No Surger", "Biopsy", "Unknown"],
    "Radiotherapy": ["Yes", "No/Unknc"],
    "Marital": ["Married", "Unmarriec"],
    "Chemotherapy": ["Yes", "No/Unknc"],
    "Extension": ["Confined t", "Midline Cr", "Ventricles", "Extra-cere", "Unknown"],
}

RAW_FEATURE_COLS = [
    "Age",
    "Income",
    "Sex",
    "Race",
    "Laterality",
    "EOR",
    "Radiotherapy",
    "Marital",
    "Chemotherapy",
    "Extension",
    "Age2",
    "Income_group",
]

@st.cache_resource
def load_assets():
    preprocessor = joblib.load(BASE_DIR / "preprocessor.joblib")
    income_quantiles = joblib.load(BASE_DIR / "income_quantiles.joblib")
    baseline_obj = joblib.load(BASE_DIR / "baseline_hazards.joblib")

    # 用一个 dummy 输入动态推断预处理后的输入维度，避免写死 34。
    dummy_row = {
        "Age": 60.0,
        "Income": 50000.0,
        "Sex": UI_OPTIONS["Sex"][0],
        "Race": UI_OPTIONS["Race"][0],
        "Laterality": UI_OPTIONS["Laterality"][0],
        "EOR": UI_OPTIONS["EOR"][0],
        "Radiotherapy": UI_OPTIONS["Radiotherapy"][0],
        "Marital": UI_OPTIONS["Marital"][0],
        "Chemotherapy": UI_OPTIONS["Chemotherapy"][0],
        "Extension": UI_OPTIONS["Extension"][0],
        "Age2": 60.0 ** 2,
        "Income_group": 0,
    }
    dummy_df = pd.DataFrame([dummy_row], columns=RAW_FEATURE_COLS)
    x_dummy = preprocessor.transform(dummy_df).astype("float32")
    in_features = x_dummy.shape[1]

    net = tt.practical.MLPVanilla(
        in_features=in_features,
        num_nodes=[512, 256, 128, 64, 32],
        out_features=1,
        batch_norm=True,
        dropout=0.2,
    )
    model = CoxPH(net)

    # 云端是 CPU 环境，必须 map 到 cpu。
    model.load_net(
        str(BASE_DIR / "deepsurv_net.pt"),
        map_location=torch.device("cpu"),
        weights_only=False,
    )
    model.net.to(torch.device("cpu"))
    model.net.eval()

    # 兼容不同 pycox 版本。
    model.baseline_hazards_ = baseline_obj
    model.baseline_cumulative_hazards_ = baseline_obj

    return model, preprocessor, income_quantiles, in_features


def calc_income_group(income: float, quantiles) -> int:
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


def build_patient_df(age, income, sex, race, laterality, eor, radiotherapy, marital, chemotherapy, extension):
    income_group = calc_income_group(income, income_quantiles)
    row = {
        "Age": float(age),
        "Income": float(income),
        "Sex": str(sex),
        "Race": str(race),
        "Laterality": str(laterality),
        "EOR": str(eor),
        "Radiotherapy": str(radiotherapy),
        "Marital": str(marital),
        "Chemotherapy": str(chemotherapy),
        "Extension": str(extension),
        "Age2": float(age) ** 2,
        "Income_group": int(income_group),
    }
    return pd.DataFrame([row], columns=RAW_FEATURE_COLS)


def predict_survival(patient_df: pd.DataFrame):
    x_final = preprocessor.transform(patient_df).astype("float32")
    surv_raw = model.predict_surv_df(x_final)
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

    curve_df = curve_df.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    if curve_df.shape[0] == 0 or curve_df.shape[1] == 0:
        raise ValueError("No valid survival curve points were returned by the model.")

    curve_series = curve_df.iloc[:, 0].astype(float).dropna().clip(lower=0, upper=1)
    if len(curve_series) == 0:
        raise ValueError("All predicted survival probabilities are NaN.")

    if curve_series.index.min() > 0:
        curve_series.loc[0.0] = 1.0
        curve_series = curve_series.sort_index()

    time_points = np.asarray(curve_series.index, dtype=float)
    values = np.asarray(curve_series.values, dtype=float)

    def nearest_prob(month: float) -> float:
        idx = np.argmin(np.abs(time_points - float(month)))
        return float(values[idx])

    if np.min(values) > 0.5:
        median_survival_time = np.nan
    else:
        median_survival_time = np.interp(0.5, values[::-1], time_points[::-1])

    return {
        "curve_series": curve_series,
        "p6": nearest_prob(6),
        "p12": nearest_prob(12),
        "p24": nearest_prob(24),
        "median_survival_time": median_survival_time,
        "x_final_shape": tuple(x_final.shape),
    }


model, preprocessor, income_quantiles, inferred_in_features = load_assets()

with st.sidebar:
    st.header("患者临床指标")
    age = st.slider("Age (年龄)", 18, 95, 60)
    income = st.number_input("Income (家庭年收入, $)", min_value=0.0, value=100000.0, step=1000.0)
    sex = st.selectbox("Sex (性别)", UI_OPTIONS["Sex"])
    race = st.selectbox("Race (种族)", UI_OPTIONS["Race"])
    laterality = st.selectbox("Laterality (侧性)", UI_OPTIONS["Laterality"])
    eor = st.selectbox("EOR (切除程度)", UI_OPTIONS["EOR"])
    radiotherapy = st.selectbox("Radiotherapy (放疗)", UI_OPTIONS["Radiotherapy"])
    marital = st.selectbox("Marital (婚姻)", UI_OPTIONS["Marital"])
    chemotherapy = st.selectbox("Chemotherapy (化疗)", UI_OPTIONS["Chemotherapy"])
    extension = st.selectbox("Extension (肿瘤累及)", UI_OPTIONS["Extension"])
    run_btn = st.button("点击开始预测")

if "pred_done" not in st.session_state:
    st.session_state.pred_done = False

if run_btn or not st.session_state.pred_done:
    try:
        patient_df = build_patient_df(
            age,
            income,
            sex,
            race,
            laterality,
            eor,
            radiotherapy,
            marital,
            chemotherapy,
            extension,
        )
        result = predict_survival(patient_df)
        st.session_state.pred_done = True
        st.session_state.patient_df = patient_df
        st.session_state.result = result
    except Exception as e:
        st.error(f"处理数据时出错: {e}")

if st.session_state.get("pred_done"):
    result = st.session_state["result"]
    curve_series = result["curve_series"]

    c1, c2, c3 = st.columns(3)
    c1.metric("0.5年生存率", f"{result['p6']:.1%}")
    c2.metric("1年生存率", f"{result['p12']:.1%}")
    c3.metric("2年生存率", f"{result['p24']:.1%}")

    if np.isnan(result["median_survival_time"]):
        st.info("Median Survival Time: Not reached")
    else:
        st.success(f"Median Survival Time: {result['median_survival_time']:.2f} months")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=np.asarray(curve_series.index, dtype=float),
            y=np.asarray(curve_series.values, dtype=float),
            mode="lines",
            name="Survival Probability",
        )
    )
    fig.update_layout(
        title="Predicted Survival Curve",
        xaxis_title="Time (Months)",
        yaxis_title="Probability",
        yaxis=dict(range=[0, 1.05]),
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("调试信息（看结果是否真的在变化）"):
        st.write("Raw patient_df", st.session_state["patient_df"])
        st.write("Transformed feature shape", result["x_final_shape"])
        st.write(
            "提示：如果你改了很多输入，但结果还是几乎不变，最可能是这 4 个文件并不是一套完整训练产物。"
        )
else:
    st.info("请在左侧填写信息并点击‘开始预测’。")
