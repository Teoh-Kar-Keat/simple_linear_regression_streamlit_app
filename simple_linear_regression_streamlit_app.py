import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import statsmodels.api as sm

st.set_page_config(page_title="HW1 — Simple Linear Regression (CRISP-DM)", layout="wide")

# ----- Sidebar: CRISP-DM parameters -----
st.sidebar.title("Controls — CRISP‑DM & Data Generation")
st.sidebar.markdown("""
- **Business / Data Understanding**: We generate a 1D linear dataset `y = a*x + b + noise`.
- **Data Preparation**: adjust how many points and the noise level.
- **Modeling**: we fit ordinary least squares and show performance.
- **Evaluation & Deployment**: metrics, residuals, and downloadable CSV.
""")

with st.sidebar.form(key="params"):
    st.write("**Data generation parameters**")
    a_true = st.number_input("True slope (a)", value=2.0, format="%.4f")
    b_true = st.number_input("True intercept (b)", value=1.0, format="%.4f")
    noise_std = st.number_input("Noise standard deviation", value=1.0, min_value=0.0, format="%.4f")
    n_points = st.slider("Number of data points", min_value=10, max_value=2000, value=100, step=1)
    x_min, x_max = st.slider("x range", -100.0, 100.0, (-10.0, 10.0))
    seed = st.number_input("Random seed (enter integer)", value=42, step=1)
    fit_intercept = st.checkbox("Force fit intercept (fit_intercept=True)", value=True)
    show_true_line = st.checkbox("Show true line (y = a*x + b)", value=True)
    submitted = st.form_submit_button("Generate / Re-run")

# regenerate on first run
if 'state' not in st.session_state:
    st.session_state.state = 0

if submitted:
    st.session_state.state += 1

# ----- Data generation -----
np.random.seed(int(seed))
X = np.random.uniform(x_min, x_max, size=n_points)
noise = np.random.normal(loc=0.0, scale=noise_std, size=n_points)
Y = a_true * X + b_true + noise

# Put into DataFrame
df = pd.DataFrame({"x": X, "y": Y})

# ----- Modeling -----
X_reshaped = df[["x"]].values
model = LinearRegression(fit_intercept=fit_intercept)
model.fit(X_reshaped, df["y"].values)

a_hat = float(model.coef_[0])
if fit_intercept:
    b_hat = float(model.intercept_)
else:
    b_hat = 0.0

# Predictions
df["y_pred"] = model.predict(X_reshaped)

# Metrics
r2 = r2_score(df["y"], df["y_pred"])
mse = mean_squared_error(df["y"], df["y_pred"])
mae = mean_absolute_error(df["y"], df["y_pred"]) 

# Statsmodels OLS for parameter std errors and CIs
X_sm = sm.add_constant(X_reshaped) if fit_intercept else X_reshaped
ols = sm.OLS(df["y"].values, X_sm).fit()

# Collect coefficient table
if fit_intercept:
    params = pd.DataFrame({
        "coef": ols.params,
        "std_err": ols.bse,
        "t": ols.tvalues,
        "p": ols.pvalues,
        "ci_low": ols.conf_int(alpha=0.05)[0],
        "ci_high": ols.conf_int(alpha=0.05)[1]
    }, index=["intercept", "slope"])
else:
    params = pd.DataFrame({
        "coef": ols.params,
        "std_err": ols.bse,
        "t": ols.tvalues,
        "p": ols.pvalues,
        "ci_low": ols.conf_int(alpha=0.05)[:,0],
        "ci_high": ols.conf_int(alpha=0.05)[:,1]
    }, index=["slope"])

# ----- Layout -----
col1, col2 = st.columns([1, 1.4])

with col1:
    st.header("Dataset & Model")
    st.write("Generated dataset (first 10 rows):")
    st.dataframe(df.head(10))

    st.markdown("**Fitted parameters**")
    st.write(f"Estimated slope (â): **{a_hat:.4f}**")
    st.write(f"Estimated intercept (b̂): **{b_hat:.4f}**")

    st.markdown("**Model evaluation**")
    st.metric("R²", f"{r2:.4f}")
    st.write(f"MSE: {mse:.4f}")
    st.write(f"MAE: {mae:.4f}")

    st.markdown("**Coefficient summary (OLS)**")
    st.table(params)

    csv = df.to_csv(index=False)
    st.download_button(label="Download dataset (CSV)", data=csv, file_name="linear_dataset.csv", mime="text/csv")

with col2:
    st.header("Scatter & Fit")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df["x"], df["y"], alpha=0.6, label="data")

    xs = np.linspace(df["x"].min(), df["x"].max(), 200)
    if show_true_line:
        ax.plot(xs, a_true * xs + b_true, linestyle="--", linewidth=2, label="true: y = a*x + b")

    ax.plot(xs, a_hat * xs + b_hat, linewidth=2, label="fitted: y = â*x + b̂")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.grid(alpha=0.2)
    st.pyplot(fig)

    st.subheader("Residuals")
    fig2, ax2 = plt.subplots(figsize=(7, 3))
    residuals = df["y"] - df["y_pred"]
    ax2.scatter(df["y_pred"], residuals, alpha=0.6)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("Predicted y")
    ax2.set_ylabel("Residual (y - y_pred)")
    ax2.grid(alpha=0.2)
    st.pyplot(fig2)

# ----- CRISP-DM explanation expanders -----
st.markdown("---")
with st.expander("What I implemented — CRISP‑DM steps (brief)"):
    st.write("1. **Business Understanding** — A simple exercise: understand relation between x and y (linear).\n"
             "2. **Data Understanding** — we generate synthetic linear data and observe distribution.\n"
             "3. **Data Preparation** — user controls (n points, noise, x range).\n"
             "4. **Modeling** — OLS is used (scikit-learn and statsmodels to show inference).\n"
             "5. **Evaluation** — R², MSE, MAE, residual plot, parameter CIs.\n"
             "6. **Deployment** — this Streamlit app is a lightweight interactive deployment for teaching and experimentation.")

st.markdown("---")

st.caption("HW1: interactive simple linear regression demo. Modify the left-side controls and click 'Generate / Re-run' to update.")

# End of file
