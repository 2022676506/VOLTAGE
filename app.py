import streamlit as st 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import scipy.stats as stats 

st.set_page_config(page_title="Elevation Adjustment (LSA)", layout="centered") st.title("🏔️ Elevation Adjustment using Least Squares Adjustment (LSA)")

--- Benchmark input ---

st.subheader("📌 Benchmark Points (Known Elevations)") bm_count = st.selectbox("How many benchmarks?", [1, 2, 3, 4, 5], index=1) known_points = {} for i in range(bm_count): col1, col2 = st.columns(2) with col1: label = st.text_input(f"Benchmark {i+1} Name", value=f"BM{i+1}") with col2: elev = st.number_input(f"Elevation of {label} (m)", format="%.3f", key=f"elev_{i}") if label: known_points[label.strip()] = elev

--- Unknown points input ---

st.subheader("🧩 Unknown Points") raw_points = st.text_input("Enter unknown point names (comma-separated)", "A,B,C") unknown_points = [pt.strip() for pt in raw_points.split(",") if pt.strip()] point_index = {pt: i for i, pt in enumerate(unknown_points)} u = len(unknown_points)

--- Observation input ---

st.subheader("📊 Observations") n_obs = st.number_input("How many observations?", min_value=1, step=1) observations = [] for i in range(int(n_obs)): st.markdown(f"Observation {i+1}") col1, col2, col3 = st.columns(3) with col1: frm = st.text_input(f"From", key=f"obs_from_{i}") with col2: to = st.text_input(f"To", key=f"obs_to_{i}") with col3: diff = st.number_input(f"Height Diff (m)", format="%.4f", key=f"obs_diff_{i}") if frm and to: observations.append((frm.strip(), to.strip(), diff))

--- Proceed Button ---

if st.button("🚀 Run LSA Adjustment"): n = len(observations) r = n - u

if r <= 0:
    st.error("❌ LSA cannot be performed because redundancy r ≤ 0.")
    st.stop()

# Matrix A and L
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

# LSA Computation
AT = A.T
N = AT @ A
U = AT @ L
X = np.linalg.inv(N) @ U
V = A @ X - L
sigma0_squared = (V.T @ V)[0, 0] / r
Cov = sigma0_squared * np.linalg.inv(N)
std_dev = np.sqrt(np.diag(Cov))
z_score = stats.norm.ppf(0.995)

# Display formulas
st.subheader("📐 Adjustment Formulas")
st.markdown(r"""
**Normal Equation**:
N = A^T A
U = A^T L
X = N^{-1} U

**Residuals**:
V = AX - L

**A Posteriori Variance**:
\sigma_0^2 = \frac{V^T V}{r}

**Covariance Matrix**:
\Sigma_X = \sigma_0^2 N^{-1}

**Confidence Interval (99%)**:
CI = X \pm Z_{0.995} \times \sqrt{\Sigma_X}
""")

# Result Table
result_df = pd.DataFrame({
    'Point': unknown_points,
    'Adjusted Elevation (m)': X.flatten(),
    'Std Deviation (m)': std_dev,
    'CI Lower Bound (99%)': X.flatten() - z_score * std_dev,
    'CI Upper Bound (99%)': X.flatten() + z_score * std_dev
})
st.success("✅ Adjustment completed successfully!")
st.dataframe(result_df.set_index('Point'), use_container_width=True)

# Residual Plot
st.subheader("📈 Residual Plot")
residuals = V.flatten()
threshold = 3 * np.sqrt(sigma0_squared)
outliers = np.abs(residuals) > threshold
normal_idx = np.where(~outliers)[0]
outlier_idx = np.where(outliers)[0]

fig1, ax1 = plt.subplots(figsize=(8, 4))
ax1.bar(normal_idx, residuals[normal_idx], color='orange', label='Residuals')
ax1.bar(outlier_idx, residuals[outlier_idx], color='red', label='Outliers')
ax1.axhline(0, color='black', linestyle='--', linewidth=1)
ax1.axhline(threshold, color='red', linestyle=':', linewidth=1)
ax1.axhline(-threshold, color='red', linestyle=':', linewidth=1)
ax1.set_title('Residual Plot of Elevation Differences')
ax1.set_xlabel('Observation Index')
ax1.set_ylabel('Residual (m)')
ax1.legend()
ax1.grid(True)
st.pyplot(fig1)

# Elevation Profile Plot
st.subheader("📊 Elevation Profile")
elevation_points = unknown_points + list(known_points.keys())
elevation_values = list(X.flatten()) + [known_points[k] for k in known_points]
confidence_intervals = [z_score * e for e in std_dev] + [0 for _ in known_points]

fig2, ax2 = plt.subplots(figsize=(10, 5))
x_positions = list(range(len(elevation_points)))
colors = ['blue' if pt in unknown_points else 'green' for pt in elevation_points]
markers = ['o' if pt in unknown_points else 's' for pt in elevation_points]

ax2.plot(x_positions, elevation_values, linestyle='-', color='gray', alpha=0.4)

for i, pt in enumerate(elevation_points):
    ax2.errorbar(x_positions[i], elevation_values[i], yerr=confidence_intervals[i],
                 fmt=markers[i], color=

