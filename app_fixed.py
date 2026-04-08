from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent
PYCOX_CACHE_DIR = "/tmp/pycox_data"
os.makedirs(PYCOX_CACHE_DIR, exist_ok=True)
os.environ["PYCOX_DATA_DIR"] = PYCOX_CACHE_DIR

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torchtuples as tt
import plotly.graph_objects as go
from pycox.models import CoxPH

st.set_page_config(page_title="GBM生存预测工具", layout="wide")
st.title("Glioblastoma (GBM) 生存预测在线工具")
st.markdown("请输入患者临床信息以获取实时生存概率预测。")


@st.cache_resource
def load_models():
    """加载模型与预处理器。"""
    in_features = 34
    net = tt.practical.MLPVanilla(in_features, [512, 256, 128, 64, 32], 1, True, 0.2)
    model = CoxPH(net)

    try:
        model.load_net(str(BASE_DIR / "deepsurv_net.pt"), weights_only=False)
        preprocessor = joblib.load(BASE_DIR / "preprocessor.joblib")
        model.baseline_hazards_ = joblib.load(BASE_DIR / "baseline_hazards.joblib")
        income_quantiles = joblib.load(BASE_DIR / "income_quantiles.joblib")
    except Exception as e:
        raise RuntimeError(
            "模型文件加载失败。最常见原因是部署环境的依赖版本与训练环境不一致，"
            "尤其是 scikit-learn / torch / pycox / torchtuples 版本不一致。\n"
            f"原始错误: {e}"
        ) from e

    return model, preprocessor, income_quantiles


try:
    model, preprocessor, income_quantiles = load_models()
except Exception as e:
    st.error(str(e))
    st.stop()


st.sidebar.header("患者临床指标")
age = st.sidebar.slider("Age (年龄)", 18, 95, 60)
income = st.sidebar.number_input("Income (家庭年收入, $)", min_value=0, value=100000, step=1000)

sex = st.sidebar.selectbox("Sex (性别)", ["Female", "Male"])
race = st.sidebar.selectbox("Race (种族)", ["White", "Black", "Asian or P"])
laterality = st.sidebar.selectbox("Laterality (侧性)", ["Midline", "Left", "Right"])
eor = st.sidebar.selectbox("EOR (切除程度)", ["GTR", "STR", "No Surger", "Biopsy", "Unknown"])
radio = st.sidebar.selectbox("Radiotherapy (放疗)", ["Yes", "No/Unknc"])
chemo = st.sidebar.selectbox("Chemotherapy (化疗)", ["Yes", "No/Unknc"])
marital = st.sidebar.selectbox("Marital (婚姻)", ["Married", "Unmarriec"])
extension = st.sidebar.selectbox(
    "Extension (肿瘤累及)",
    ["Confined t", "Midline Cr", "Ventricles", "Extra-cere", "Unknown"],
)


def make_income_group(value: float, quantiles: np.ndarray) -> int:
    """把收入分到完整区间，避免范围外样本都变成 0。"""
    bins = np.concatenate(([-np.inf], np.asarray(quantiles, dtype=float), [np.inf]))
    group = pd.cut([value], bins=bins, labels=False, include_lowest=True)[0]
    return int(group)


if st.sidebar.button("点击开始预测"):
    raw_df = pd.DataFrame([
        {
            "Age": age,
            "Income": income,
            "Sex": sex,
            "Race": race,
            "Laterality": laterality,
            "EOR": eor,
            "Radiotherapy": radio,
            "Marital": marital,
            "Chemotherapy": chemo,
            "Extension": extension,
        }
    ])

    raw_df["Age2"] = raw_df["Age"] ** 2
    raw_df["Income_group"] = make_income_group(income, income_quantiles)

    try:
        x_final = preprocessor.transform(raw_df).astype("float32")
        surv_df = model.predict_surv_df(x_final)

        st.subheader("预测结果")
        c1, c2, c3 = st.columns(3)

        time_points = surv_df.index.to_numpy(dtype=float)
        surv_values = surv_df.iloc[:, 0].to_numpy(dtype=float)

        def get_p(months: float) -> float:
            idx = int(np.abs(time_points - months).argmin())
            return float(surv_values[idx])

        c1.metric("1年生存率", f"{get_p(12):.1%}")
        c2.metric("3年生存率", f"{get_p(36):.1%}")
        c3.metric("5年生存率", f"{get_p(60):.1%}")

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=time_points,
                y=surv_values,
                name="Survival Prob",
                line=dict(width=3),
            )
        )
        fig.update_layout(
            title="Predicted Survival Curve",
            xaxis_title="Time (Months)",
            yaxis_title="Probability",
            yaxis=dict(range=[0, 1.05]),
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"处理数据时出错，请检查输入是否正确: {e}")
else:
    st.info("请在左侧填写信息并点击‘开始预测’。")
