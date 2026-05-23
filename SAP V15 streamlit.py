"""
Streamlit web app for SAP prediction (V15 model).
Loads the best model exported by V15 and provides a bedside risk estimation.
"""

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import shap

# ----------------------------- 页面配置 -----------------------------
st.set_page_config(
    page_title="SAP Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------- 加载模型 -----------------------------
@st.cache_resource
def load_model():
    """Load the exported pipeline, feature list, and optimal threshold."""
    model_pack = joblib.load("best_model.pkl")
    pipe = model_pack["pipe"]
    features = model_pack["features"]
    threshold = model_pack.get("threshold", 0.5)
    return pipe, features, threshold

pipe, features, threshold = load_model()

# ----------------------------- 特征说明 -----------------------------
feature_info = {
    "CRP": ("C-Reactive Protein (mg/L)", 0.0, 300.0),
    "mRS": ("Modified Rankin Scale (0-6)", 0, 6),
    "kwst": ("Wada Drinking Test (0-5)", 0, 5),
    "NIHSS": ("NIH Stroke Scale (0-42)", 0, 42),
    "VTE_score": ("VTE Risk Score", 0, 10),
    "Albumin": ("Serum Albumin (g/L)", 20.0, 50.0),
    "NLR": ("Neutrophil-Lymphocyte Ratio", 0.0, 50.0),
    "age": ("Age (years)", 65, 100),
    "LDL": ("LDL Cholesterol (mmol/L)", 0.5, 8.0),
    "RDW": ("Red Cell Distribution Width (%)", 10.0, 25.0),
    "ppi": ("Proton Pump Inhibitor Use (0=No, 1=Yes)", 0, 1),
    "Urea": ("Blood Urea Nitrogen (mmol/L)", 1.0, 30.0),
}

# ----------------------------- 应用界面 -----------------------------
st.title("🩺 Stroke-Associated Pneumonia (SAP) Risk Predictor")
st.markdown(
    """
    This tool estimates the probability of developing **stroke-associated pneumonia** 
    in older patients with acute ischemic stroke (age ≥65 years). 
    It is based on a linear SVM model trained on 1,011 patients and uses 12 clinical features.
    """
)

st.sidebar.header("Patient Data")
st.sidebar.markdown("Please enter the 12 required clinical features below.")

input_data = {}
for feat in features:
    if feat in feature_info:
        label, min_val, max_val = feature_info[feat]
        if isinstance(min_val, int) and isinstance(max_val, int):
            # integer input
            value = st.sidebar.number_input(
                label, min_value=min_val, max_value=max_val, value=min_val, step=1
            )
        else:
            value = st.sidebar.number_input(
                label, min_value=float(min_val), max_value=float(max_val), value=float(min_val)
            )
        input_data[feat] = value
    else:
        # fallback generic input
        input_data[feat] = st.sidebar.number_input(feat, value=0.0)

# 转为 DataFrame
input_df = pd.DataFrame([input_data])

# ----------------------------- 预测 -----------------------------
if st.sidebar.button("Predict SAP Risk"):
    # 确保列顺序与训练时一致
    input_df = input_df[features]
    
    # 获取预测概率
    proba = pipe.predict_proba(input_df)[:, 1][0]
    pred_class = int(proba >= threshold)
    
    # 显示结果
    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Probability", f"{proba:.3f}")
    col2.metric("Risk Classification", "High Risk" if pred_class == 1 else "Low Risk")
    col3.metric("Threshold Used", f"{threshold:.2f}")
    
    # 进度条
    st.progress(min(proba, 1.0), text=f"Risk: {proba:.1%}")
    
    # 解释
    st.subheader("Top Contributing Features")
    # 使用 SHAP 解释单个预测（可选，需较长时间）
    try:
        # 获取背景数据 (这里简单使用全0背景，实际可预先保存)
        # 因 KernelExplainer 较慢，用 LinearExplainer 或 TreeExplainer 根据模型类型
        classifier = pipe.named_steps["classifier"]
        if hasattr(classifier, "coef_"):
            # 线性模型：显示系数
            coefs = classifier.coef_.flatten()
            importance = pd.DataFrame({"Feature": features, "Coefficient": coefs})
            importance["AbsCoeff"] = np.abs(importance["Coefficient"])
            importance = importance.sort_values("AbsCoeff", ascending=False).head(5)
            fig, ax = plt.subplots()
            colors = ["red" if c < 0 else "green" for c in importance["Coefficient"]]
            ax.barh(importance["Feature"], importance["Coefficient"], color=colors)
            ax.set_xlabel("Coefficient")
            ax.axvline(0, color="black", linewidth=0.8)
            st.pyplot(fig)
            st.caption("Shown are the top 5 features with largest linear model coefficients.")
        else:
            st.info("Feature importance visualization is not available for this model type in the demo.")
    except Exception as e:
        st.warning(f"Could not generate explanation: {e}")

else:
    st.info("👈 Enter patient values in the sidebar and click 'Predict SAP Risk' to see results.")

# ----------------------------- 脚注 -----------------------------
st.markdown("---")
st.markdown(
    """
    **Disclaimer:** This tool is for research purposes only and should not be used as 
    the sole basis for clinical decision-making.
    """
)