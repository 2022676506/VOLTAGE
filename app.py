import streamlit as st 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import scipy.stats as stats 
import datetime from matplotlib.lines 
import Line2D from io import BytesIO

st.set_page_config(page_title="Elevation Adjustment via LSA", layout="wide") st.title("📏 Elevation Adjustment using Least Squares Adjustment (LSA)")

========== STEP 1: Input benchmark points ==========

st.header("1️⃣ Input Benchmark Points")

bm_option = st.radio("Select number of benchmark points", ["1", "2", "Custom"]) if bm_option == "1": bm_count = 1 elif bm_option == "2": bm_count = 2 else: bm_count = st.number_input("Enter custom number of benchmark points", min_value=3, step=1)\n known_points = {} st.subheader("Enter Benchmark Elevations") for i in range(bm_count): col1, col2 = st.columns(2) with col1: label = st.text_input(f"Label for BM{i+1}", key=f"bm_label_{i}") with col2: elevation = st.number_input(f"Elevation for {label} (m)", format="%.3f", step=0.001, key=f"bm_elev_{i}") if label: known_points[label] = elevation

========== STEP 2: Unknown points ==========

st.header("2️⃣ Unknown Points") raw_unknowns = st.text_input("Enter unknown point labels (comma-separated)", value="A,B,C") unknown_points = [pt.strip() for pt in raw_unknowns.split(",") if pt.strip()] point_index = {pt: i for i, pt in enumerate(unknown_points)} u = len(unknown_points)

========== STEP 3: Observations ==========

st.header("3️⃣ Observations") n_obs = st.number_input("Number of observations", min_value=1, step=1) observations = []

for i in range(n_obs): with st.expander(f"Observation {i+1}"): frm = st.text_input(f"From point", key=f"from_{i}") to = st.text_input(f"To point", key=f"to_{i}") diff = st.number_input(f"Height difference (m)", format="%.3f", step=0.001, key=f"diff_{i}") if frm and to: observations.append((frm, to, diff))

========== STEP 4: Perform LSA ==========

if st.button("🔍 Perform LSA"): st.header("🧮 Least Squares Adjustment Results") n = len(observations) r = n - u

st.markdown(f"**Redundancy (r)** = n - u = {n} - {u} = **{r}**")
if r <= 0:
    st.error("❌ LSA cannot be performed because redundancy r ≤ 0.")
    st.stop()

A = np.zeros((n, u))
L = np.zeros((n, 1))

for i, (frm, to, dh) in enumerate(observations):
    if frm in point_index:
        A[i, point_index[frm]] = -1
    elif frm in known_points:
        L[i] += known_points[frm]

    if to in point_index:
        A[i, point_index[to]] = 1
    elif to in known_points:
        L[i] -= known_points[to]

    L[i] += dh

st.subheader("Matrix A:")
st.write(np.round(A, 3))

st.subheader("Matrix L:")
st.write(np.round(L, 3))

# LSA computation
AT = A.T
N = AT @ A
U = AT @ L
X = np.linalg.inv(N) @ U
V = A @ X - L
sigma0_squared = (V.T @ V)[0, 0] / r
Cov = sigma0_squared * np.linalg.inv(N)
std_dev = np.sqrt(np.diag(Cov))

st.subheader("📐 Formulas Used")
st.latex(r"A \cdot X = L")
st.latex(r"N = A^T A")
st.latex(r"U = A^T L")
st.latex(r"X = N^{-1} U")
st.latex(r"V = AX - L")
st.latex(r"\hat{\sigma}_0^2 = \frac{V^T V}{r}")
st.latex(r"\text{Cov}(X) = \hat{\sigma}_0^2 \cdot (A^T A)^{-1}")

confidence_level = 0.99
z_score = stats.norm.ppf(1 - (1 - confidence_level) / 2)

df_output = pd.DataFrame({
    'Point': unknown_points,
    'Adjusted Elevation (m)': np.round(X.flatten(), 4),
    'Std Deviation (m)': np.round(std_dev, 4),
    'CI Lower Bound (99%)': np.round(X.flatten() - z_score * std_dev, 4),
    'CI Upper Bound (99%)': np.round(X.flatten() + z_score * std_dev, 4)
})

st.subheader("📋 Adjusted Elevations and Confidence Intervals")
st.dataframe(df_output)

st.success(f"Variance Factor (\u03c3₀²): {sigma0_squared:.5f}")

# ====== CSV Download ======
csv_data = df_output.to_csv(index=False).encode('utf-8')
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f"lsa_adjustment_result_{timestamp}.csv"
st.download_button("📥 Download CSV", data=csv_data, file_name=filename, mime="text/csv")

# Combine all points
elevation_points = unknown_points + list(known_points.keys())
elevation_values = list(X.flatten()) + [known_points[k] for k in known_points]
confidence_intervals = [z_score * e for e in std_dev] + [0 for _ in known_points]

# ===== Residual Plot =====
st.subheader("📊 Residual Plot")
threshold = 3 * np.sqrt(sigma0_squared)
residuals = V.flatten()
outliers = np.abs(residuals) > threshold
normal_idx = np.where(~outliers)[0]
outlier_idx = np.where(outliers)[0]

fig1, ax1 = plt.subplots(figsize=(8, 4))
ax1.bar(normal_idx, residuals[normal_idx], color='orange', label='Residuals')
ax1.bar(outlier_idx, residuals[outlier_idx], color='red', label='Outliers')
ax1.axhline(0

