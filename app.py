import os

PYCOX_CACHE_DIR = "/tmp/pycox_data"
os.makedirs(PYCOX_CACHE_DIR, exist_ok=True)
os.environ["PYCOX_DATA_DIR"] = PYCOX_CACHE_DIR

import streamlit as st
import pandas as pd
import numpy as np
import torch
import joblib
import torchtuples as tt
import plotly.graph_objects as go
from pycox.models import CoxPH
# ==========================================
# 1. 页面基本配置
# ==========================================
st.set_page_config(page_title="GBM生存预测工具", layout="wide")
st.title("Glioblastoma (GBM) 生存预测在线工具")
st.markdown("请输入患者临床信息以获取实时生存概率预测。")


# ==========================================
# 2. 加载已经集齐的 4 个零件
# ==========================================

def load_models():
    # 注意：这里的 34 必须对应你训练时的特征数，如果不匹配，运行会报错
    in_features = 34
    net = tt.practical.MLPVanilla(in_features, [512, 256, 128, 64, 32], 1, True, 0.2)
    model = CoxPH(net)
    model.load_net('deepsurv_net.pt', weights_only=False)

    # 加载其他 joblib 文件
    preprocessor = joblib.load('preprocessor.joblib')
    model.baseline_hazards_ = joblib.load('baseline_hazards.joblib')
    income_quantiles = joblib.load('income_quantiles.joblib')

    return model, preprocessor, income_quantiles


model, preprocessor, income_quantiles = load_models()

# ==========================================
# 3. 侧边栏输入 (根据你的原始 Excel 数据列名)
# ==========================================
st.sidebar.header("患者临床指标")

# 数值输入
age = st.sidebar.slider("Age (年龄)", 18, 95, 60)
income = st.sidebar.number_input("Income (家庭年收入, $)", value=100000)

# 分类输入 (选项字符串必须与你 Excel 里的完全一致)
sex = st.sidebar.selectbox("Sex (性别)", ["Female", "Male"])
race = st.sidebar.selectbox("Race (种族)", ["White", "Black", "Asian or P"])
laterality = st.sidebar.selectbox("Laterality (侧性)", ["Midline", "Left", "Right"])
eor = st.sidebar.selectbox("EOR (切除程度)", ["GTR", "STR", "No Surger", "Biopsy", "Unknown"])
radio = st.sidebar.selectbox("Radiotherapy (放疗)", ["Yes", "No/Unknc"])
chemo = st.sidebar.selectbox("Chemotherapy (化疗)", ["Yes", "No/Unknc"])
marital = st.sidebar.selectbox("Marital (婚姻)", ["Married", "Unmarriec"])
extension = st.sidebar.selectbox("Extension (肿瘤累及)",
                                 ["Confined t", "Midline Cr", "Ventricles", "Extra-cere", "Unknown"])

# ==========================================
# 4. 预测与绘图逻辑
# ==========================================
if st.sidebar.button("点击开始预测"):
    # 构造原始数据
    raw_df = pd.DataFrame([{
        "Age": age, "Income": income, "Sex": sex, "Race": race,
        "Laterality": laterality, "EOR": eor, "Radiotherapy": radio,
        "Marital": marital, "Chemotherapy": chemo, "Extension": extension
    }])

    # 后台处理衍生变量
    raw_df["Age2"] = raw_df["Age"] ** 2
    raw_df["Income_group"] = pd.cut(raw_df["Income"], bins=income_quantiles, labels=False, include_lowest=True).fillna(
        0).astype(int)

    # 特征转换
    try:
        x_final = preprocessor.transform(raw_df).astype("float32")
        surv_df = model.predict_surv_df(x_final)

        # 结果展示
        st.subheader("预测结果")
        c1, c2, c3 = st.columns(3)


        # 提取 1, 3, 5 年节点 (假设单位是月)
        def get_p(m):
            return surv_df.iloc[(surv_df.index - m).abs().argmin(), 0]


        c1.metric("1年生存率", f"{get_p(12):.1%}")
        c2.metric("3年生存率", f"{get_p(36):.1%}")
        c3.metric("5年生存率", f"{get_p(60):.1%}")

        # 绘制交互式曲线
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=surv_df.index, y=surv_df.iloc[:, 0], name="Survival Prob",
                                 line=dict(color='#007bff', width=3)))
        fig.update_layout(title="Predicted Survival Curve", xaxis_title="Time (Months)", yaxis_title="Probability",
                          yaxis=dict(range=[0, 1.05]))
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"处理数据时出错，请检查输入是否正确: {e}")

else:
    st.info("请在左侧填写信息并点击‘开始预测’。")