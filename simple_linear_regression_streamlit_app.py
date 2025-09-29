import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="HW1 — Simple Linear Regression", layout="wide")

# ----- Sidebar: Config -----
st.sidebar.title("Configuration")
num_points = st.sidebar.slider("Number of data points", min_value=10, max_value=1000, value=100, step=10)
coef = st.sidebar.slider("Coefficient (slope a)", min_value=-10.0, max_value=10.0, value=2.0, step=0.1)
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
    st.header("Result Plot")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(X, Y, label="Data", alpha=0.6)
    ax.plot(X, Y_pred, color="red", label="Fitted line")
    ax.scatter(X[outliers], Y[outliers], color="orange", label="Outliers", zorder=5)
    ax.legend()
    st.pyplot(fig)

    st.subheader("Model Coefficients")
    st.write(f"Slope (â): {model.coef_[0]:.4f}")
    st.write(f"Intercept (b̂): {model.intercept_:.4f}")

    st.subheader("Outliers")
    outlier_df = pd.DataFrame({"x": X[outliers], "y": Y[outliers], "residual": residuals[outliers]})
    if not outlier_df.empty:
        st.dataframe(outlier_df)
    else:
        st.write("No significant outliers detected.")
