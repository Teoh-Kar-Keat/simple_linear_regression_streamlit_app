import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 設定頁面標題與版面
st.set_page_config(page_title="HW1 — 簡單線性迴歸示範", layout="wide")

# ----- 側邊控制欄 -----
st.sidebar.title("參數設定")
# 使用 slider 控制資料點數、斜率與雜訊變異數
num_points = st.sidebar.slider("資料點數 (Number of data points)", min_value=10, max_value=1000, value=100, step=10)
coef = st.sidebar.slider("斜率 (slope a)", min_value=-10.0, max_value=10.0, value=2.0, step=0.1)
noise_var = st.sidebar.slider("雜訊變異數 (noise variance)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

# ----- 資料生成 -----
np.random.seed(42)  # 固定隨機種子以利重現
X = np.linspace(-10, 10, num_points)
noise = np.random.normal(0, noise_var, num_points)
Y = coef * X + noise

# 建立線性迴歸模型並擬合
X_reshaped = X.reshape(-1, 1)
model = LinearRegression()
model.fit(X_reshaped, Y)
Y_pred = model.predict(X_reshaped)

# 殘差計算 & 簡單 outlier 偵測（超過 ±2σ 視為異常點）
residuals = Y - Y_pred
threshold = 2 * np.std(residuals)
outliers = np.abs(residuals) > threshold

# ----- 主畫面布局 -----
st.header("📈 簡單線性迴歸互動式展示")

# 上方結果圖
st.subheader("結果圖 (Result Plot)")
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(X, Y, label="資料點 (Data)", alpha=0.6)
ax.plot(X, Y_pred, color="red", linewidth=2, label="迴歸線 (Fitted line)")
ax.scatter(X[outliers], Y[outliers], color="orange", edgecolors="black", s=80, label="異常點 (Outliers)", zorder=5)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend(frameon=True)
ax.grid(alpha=0.3)
st.pyplot(fig)

# 中間模型係數
st.subheader("模型係數 (Model Coefficients)")
st.markdown(f"**斜率 (â):** {model.coef_[0]:.4f}")
st.markdown(f"**截距 (b̂):** {model.intercept_:.4f}")

# 下方 Outliers 資料表
st.subheader("異常點清單 (Outliers)")
outlier_df = pd.DataFrame({"x": X[outliers], "y": Y[outliers], "residual": residuals[outliers]})
if not outlier_df.empty:
    st.dataframe(outlier_df)
else:
    st.success("沒有偵測到明顯的異常點 🚀")
