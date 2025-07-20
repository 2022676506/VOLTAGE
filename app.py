from io import BytesIO
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from datetime import datetime
from matplotlib.lines import Line2D

st.set_page_config(page_title="Elevation Adjustment via LSA", layout="wide")
st.title("📏 Elevation Adjustment using Least Squares Adjustment (LSA)")

# ===============================
# INPUT SECTION
# ===============================
def get_data():
    with st.form("data_form"):
        st.subheader("📌 Input Observations")
        n_obs = st.number_input("Number of Observations (e.g. 5)", min_value=2, max_value=100, value=5)
        st.markdown("Enter data in the format: **From, To, Elevation Difference (m), Std Dev (mm)**")

        obs_data = []
        for i in range(n_obs):
            cols = st.columns(4)
            f = cols[0].text_input(f"From [{i+1}]", key=f"from_{i}")
            t = cols[1].text_input(f"To [{i+1}]", key=f"to_{i}")
            dh = cols[2].number_input(f"Δh (m) [{i+1}]", format="%.3f", key=f"dh_{i}")
            std_mm = cols[3].number_input(f"σ (mm) [{i+1}]", min_value=0.0, format="%.1f", key=f"std_{i}")
            obs_data.append([f.strip(), t.strip(), dh, std_mm / 1000.0])

        st.subheader("🎯 Benchmark Elevation")
        bm_name = st.text_input("Benchmark Point Name", value="BM1")
        bm_elevation = st.number_input("Benchmark Elevation (m)", format="%.3f")
        submitted = st.form_submit_button("🚀 Perform Adjustment")
    return submitted, obs_data, bm_name, bm_elevation

# ===============================
# LSA COMPUTATION
# ===============================
def perform_lsa(obs_data, bm_name, bm_elevation):
    all_points = set()
    for f, t, _, _ in obs_data:
        all_points.update([f, t])
    unknown_points = sorted(list(all_points - {bm_name}))
    point_index = {pt: i for i, pt in enumerate(unknown_points)}
    n = len(unknown_points)
    m = len(obs_data)

    A = np.zeros((m, n))
    L = np.zeros((m, 1))
    P = np.zeros((m, m))

    for i, (f, t, dh, std) in enumerate(obs_data):
        if t != bm_name:
            A[i, point_index[t]] = 1
        if f != bm_name:
            A[i, point_index[f]] = -1
        known_elev = 0
        if t == bm_name:
            known_elev -= bm_elevation
        if f == bm_name:
            known_elev += bm_elevation
        L[i, 0] = dh + known_elev
        P[i, i] = 1 / std**2

    N = A.T @ P @ A
    U = A.T @ P @ L
    X = np.linalg.solve(N, U)
    V = A @ X - L
    sigma0_squared = (V.T @ P @ V) / (m - n)
    Qxx = np.linalg.inv(N)
    S = np.sqrt(np.diag(Qxx) * sigma0_squared.item())

    adjusted_points = {bm_name: bm_elevation}
    for i, pt in enumerate(unknown_points):
        adjusted_points[pt] = X[i, 0]

    confidence_intervals = stats.t.ppf(1 - 0.005, df=m - n) * S

    df = pd.DataFrame({
        "Point": list(adjusted_points.keys()),
        "Elevation (m)": [adjusted_points[pt] for pt in adjusted_points],
        "± CI (99%)": [0] + list(confidence_intervals)
    })
    return df, V, sigma0_squared, unknown_points, adjusted_points, confidence_intervals

# ===============================
# MAIN APP
# ===============================
submitted, obs_data, bm_name, bm_elevation = get_data()
if submitted:
    df, V, sigma0_squared, unknown_points, adjusted_points, confidence_intervals = perform_lsa(obs_data, bm_name, bm_elevation)
    st.success("✅ Adjustment Completed!")

    st.subheader("📄 Adjusted Elevations")
    st.dataframe(df.style.format({"Elevation (m)": "{:.3f}", "± CI (99%)": "{:.3f}"}))

    # Download CSV with timestamp
    csv_data = df.to_csv(index=False).encode("utf-8")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"lsa_adjustment_result_{timestamp}.csv"
    st.download_button("📥 Download CSV", data=csv_data, file_name=filename, mime="text/csv")

    # Plot residuals
    st.subheader("📊 Residual Plot")
    threshold = 3 * np.sqrt(sigma0_squared)
    residuals = V.flatten()
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

    buf1 = BytesIO()
    fig1.savefig(buf1, format="png")
    st.download_button("📥 Download Residual Plot as PNG", buf1.getvalue(), file_name="residual_plot.png", mime="image/png")

    # Plot elevation profile
    st.subheader("📉 Adjusted Elevation Profile")
    elevation_points = list(adjusted_points.keys())
    elevation_values = [adjusted_points[pt] for pt in elevation_points]
    x_positions = list(range(len(elevation_points)))
    colors = ['blue' if pt in unknown_points else 'green' for pt in elevation_points]
    markers = ['o' if pt in unknown_points else 's' for pt in elevation_points]

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(x_positions, elevation_values, linestyle='-', color='gray', alpha=0.4)
    for i, pt in enumerate(elevation_points):
        ax2.errorbar(x_positions[i], elevation_values[i], yerr=confidence_intervals[i],
                     fmt=markers[i], color=colors[i], ecolor='gray', capsize=5, markersize=8)
        ax2.text(x_positions[i], elevation_values[i] + 0.1,
                 f"{pt}\n{elevation_values[i]:.3f} m", ha='center', fontsize=8)
    legend_elements = [
        Line2D([0], [0], marker='o', color='blue', label='Unknown Point', linestyle=''),
        Line2D([0], [0], marker='s', color='green', label='Benchmark (BM)', linestyle='')
    ]
    ax2.legend(handles=legend_elements)
    ax2.set_title('Adjusted Elevation Profile (99% CI Including BM)')
    ax2.set_xlabel('Point Index')
    ax2.set_ylabel('Elevation (m)')
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(elevation_points)
    ax2.grid(True)
    st.pyplot(fig2)

    buf2 = BytesIO()
    fig2.savefig(buf2, format="png")
    st.download_button("📥 Download Elevation Profile as PNG", buf2.getvalue(), file_name="elevation_profile.png", mime="image/png")
