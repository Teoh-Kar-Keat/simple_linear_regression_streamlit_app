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

# --- Page config ---
st.set_page_config(page_title="HW1 — Simple Linear Regression", layout="wide")

# ----- Sidebar: Config -----
st.sidebar.title("Configuration")
num_points = st.sidebar.slider("Number of data points", min_value=10, max_value=1000, value=100, step=10)
coef = st.sidebar.slider("Slope (a)", min_value=-10.0, max_value=10.0, value=2.0, step=0.1)
noise_var = st.sidebar.slider("Noise variance", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
seed_value = st.sidebar.number_input("Random seed", min_value=0, max_value=1000, value=42, step=1)

# ----- Data Generation -----
np.random.seed(seed_value)
X = np.linspace(-10, 10, num_points)
noise = np.random.normal(0, noise_var, num_points)
Y = coef * X + noise

# Fit linear regression
X_reshaped = X.reshape(-1, 1)
model = LinearRegression()
model.fit(X_reshaped, Y)
Y_pred = model.predict(X_reshaped)

# Residuals & Outlier Detection
residuals = Y - Y_pred
threshold = 2 * np.std(residuals)
outliers = np.abs(residuals) > threshold

# ----- Layout -----
st.title("📈 Simple Linear Regression with Residual Lines")

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(X, Y, label="Data", alpha=0.6)
ax.plot(X, Y_pred, color="red", linewidth=2, label="Fitted line")

# Residual lines
for xi, yi, y_hat in zip(X, Y, Y_pred):
    ax.plot([xi, xi], [yi, y_hat], color='gray', linestyle='--', alpha=0.5)

# Outliers
ax.scatter(X[outliers], Y[outliers], color="orange", label="Outliers", zorder=5)

ax.set_xlabel("X", fontsize=14)
ax.set_ylabel("Y", fontsize=14)
ax.set_title("Linear Regression with Residual Lines", fontsize=16)
ax.legend(fontsize=12)
ax.grid(alpha=0.3)
st.pyplot(fig)

# Model coefficients
st.subheader("Model Coefficients")
st.markdown(f"**Slope (â):** {model.coef_[0]:.4f}")
st.markdown(f"**Intercept (b̂):** {model.intercept_:.4f}")

# Outliers table
st.subheader("Outliers")
outlier_df = pd.DataFrame({"X": X[outliers], "Y": Y[outliers], "Residual": residuals[outliers]})
if not outlier_df.empty:
    st.dataframe(outlier_df)
else:
    st.success("No significant outliers detected 🚀")

# Optional: download CSV
csv = pd.DataFrame({"X": X, "Y": Y, "Predicted Y": Y_pred, "Residuals": residuals}).to_csv(index=False)
st.download_button("Download Data as CSV", csv, "linear_regression_data.csv", "text/csv")
