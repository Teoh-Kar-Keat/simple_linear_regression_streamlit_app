import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# --- Custom CSS for Better Font Sizes ---
st.markdown(
    """
    <style>
    html, body, [class*="css"]  {
        font-size: 18px;
    }
    .stHeader, .stAppViewContainer .main .block-container h1 {
        font-size: 2.2em !important;
    }
    .stSubheader, .stAppViewContainer .main .block-container h2 {
        font-size: 1.5em !important;
    }
    .stDataFrame, .stTable {
        font-size: 1.1em !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.set_page_config(page_title="HW1 — Simple Linear Regression", layout="wide")

# ----- Sidebar: Config -----
st.sidebar.title("Configuration")
num_points = st.sidebar.slider("Number of data points", min_value=10, max_value=1000, value=100, step=10)
coef = st.sidebar.slider("Coeffici ent (slope a)", min_value=-10.0, max_value=10.0, value=2.0, step=0.1)
noise_var = st.sidebar.slider("Noise variance", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

# ----- Data Generation -----
np.random.seed(42)
X = np.linspace(-10, 10, num_points)
noise = np.random.normal(0, noise_var, num_points)
Y = coef * X + noise

# Fit linear regression
X_reshaped = X.reshape(-1, 1)
model = LinearRegression()
model.fit(X_reshaped, Y)
Y_pred = model.predict(X_reshaped)

# Detect outliers (simple residual threshold)
residuals = Y - Y_pred
threshold = 2 * np.std(residuals)
outliers = np.abs(residuals) > threshold

# ----- Layout -----
col1, = st.columns([1])

with col1:
    st.markdown('<h1 style="font-size:2.5em;">Result Plot</h1>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(X, Y, label="Data", alpha=0.6)
    ax.plot(X, Y_pred, color="red", label="Fitted line")
    ax.scatter(X[outliers], Y[outliers], color="orange", label="Outliers", zorder=5)
    ax.legend(fontsize=16)
    ax.set_xlabel("X", fontsize=16)
    ax.set_ylabel("Y", fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
    st.pyplot(fig)

    st.markdown('<h2 style="font-size:1.7em;">Model Coefficients</h2>', unsafe_allow_html=True)
    st.write(f"<span style='font-size:1.2em;'>Slope (â): <b>{model.coef_[0]:.4f}</b></span>", unsafe_allow_html=True)
    st.write(f"<span style='font-size:1.2em;'>Intercept (b̂): <b>{model.intercept_:.4f}</b></span>", unsafe_allow_html=True)

    st.markdown('<h2 style="font-size:1.7em;">Outliers</h2>', unsafe_allow_html=True)
    outlier_df = pd.DataFrame({"X": X[outliers], "Y": Y[outliers], "Residual": residuals[outliers]})
    if not outlier_df.empty:
        st.dataframe(outlier_df)
    else:
        st.write("<span style='font-size:1.1em;'>No significant outliers detected.</span>", unsafe_allow_html=True)
