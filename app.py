import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import datetime
import io
import os

# ================================
# CONFIGURATION & LOGO
# ================================
st.set_page_config(page_title="LSA Elevation Adjustment", layout="wide")
st.markdown("<h1 style='color:#2c3e50;'>🛰️ Noah's Ark: LSA Elevation Adjustment Tool</h1>", unsafe_allow_html=True)
st.markdown("Upload data or enter manually for height adjustment using Least Squares.")
st.markdown("---")

# ================================
# FILE UPLOAD SECTION
# ================================
input_method = st.radio("📥 Select Data Input Method:", ['Manual Input', 'Upload CSV', 'Upload TXT'])

def parse_txt(file):
    content = file.read().decode("utf-8").splitlines()
    known_points, unknown_points, observations = {}, [], []
    mode = None
    for line in content:
        line = line.strip()
        if line.lower() == "[benchmarks]":
            mode = "bm"
            continue
        elif line.lower() == "[unknowns]":
            mode = "unknown"
            continue
        elif line.lower() == "[observations]":
            mode = "obs"
            continue
        if not line or line.startswith("#"):
            continue
        if mode == "bm":
            label, val = line.split(',')
            known_points[label.strip()] = float(val)
        elif mode == "unknown":
            unknown_points.append(line.strip())
        elif mode == "obs":
            frm, to, diff = line.split(',')
            observations.append((frm.strip(), to.strip(), float(diff)))
    return known_points, unknown_points, observations

if input_method == "Upload CSV":
    uploaded_file = st.file_uploader("Upload CSV File", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        known_points = dict(zip(df[df['Type'] == 'BM']['Point'], df[df['Type'] == 'BM']['Elevation']))
        unknown_points = df[df['Type'] == 'Unknown']['Point'].tolist()
        observations = list(zip(df['From'], df['To'], df['HeightDiff']))

elif input_method == "Upload TXT":
    txt_file = st.file_uploader("Upload TXT File", type="txt")
    if txt_file:
        known_points, unknown_points, observations = parse_txt(txt_file)

else:
    # Manual input
    st.subheader("Benchmark Points")
    num_bm = st.number_input("Number of Benchmark Points", 1, 10, 2)
    known_points = {}
    for i in range(num_bm):
        col1, col2 = st.columns([1, 2])
        with col1:
            label = st.text_input(f"Label BM {i+1}", value=f"BM{i+1}")
        with col2:
            elevation = st.number_input(f"Elevation of {label} (m)", key=f"elev_{i}", format="%.3f")
        known_points[label] = elevation

    st.subheader("Unknown Points")
    unknown_str = st.text_input("Enter unknown points separated by commas (e.g., A,B,C)", "A,B,C")
    unknown_points = [pt.strip() for pt in unknown_str.split(",") if pt.strip()]

    st.subheader("Observations")
    num_obs = st.number_input("Number of Observations", 1, 100, 5)
    observations = []
    for i in range(num_obs):
        col1, col2, col3 = st.columns(3)
        with col1:
            frm = st.text_input(f"From (Obs {i+1})", key=f"frm_{i}")
        with col2:
            to = st.text_input(f"To (Obs {i+1})", key=f"to_{i}")
        with col3:
            dh = st.number_input(f"ΔH (Obs {i+1})", key=f"dh_{i}", format="%.3f")
        if frm and to:
            observations.append((frm, to, dh))

# ================================
# LSA CALCULATION
# ================================
if 'observations' in locals() and st.button("Run LSA"):
    st.subheader("📊 Adjustment Results")

    u = len(unknown_points)
    point_index = {pt: i for i, pt in enumerate(unknown_points)}
    n = len(observations)
    r = n - u

    if r <= 0:
        st.error("Insufficient redundancy. LSA cannot be performed.")
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

    X = np.linalg.inv(A.T @ A) @ A.T @ L
    V = A @ X - L
    sigma0_sq = float((V.T @ V) / r)
    std_dev = np.sqrt(np.diag(sigma0_sq * np.linalg.inv(A.T @ A)))
    z_score = stats.norm.ppf(0.995)

    st.write("### Adjusted Elevation (99% CI)")
    results = []
    for i, pt in enumerate(unknown_points):
        elev = X[i, 0]
        ci = z_score * std_dev[i]
        results.append([pt, elev, std_dev[i], elev - ci, elev + ci])
    df_result = pd.DataFrame(results, columns=["Point", "Elevation", "Std Dev", "CI Lower", "CI Upper"])
    st.dataframe(df_result.style.format({'Elevation': "{:.3f}", 'Std Dev': "{:.4f}", 'CI Lower': "{:.3f}", 'CI Upper': "{:.3f}"}))

    # ================================
    # VISUALIZATION
    # ================================
    st.subheader("📈 Graphical Output")

    fig1, ax1 = plt.subplots(figsize=(6, 3))
    residuals = V.flatten()
    threshold = 3 * np.sqrt(sigma0_sq)
    ax1.bar(range(len(residuals)), residuals, color='orange', label='Residual')
    ax1.axhline(threshold, color='red', linestyle='--', label='±3σ Threshold')
    ax1.axhline(-threshold, color='red', linestyle='--')
    ax1.set_title("Residual Plot")
    ax1.legend()
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    all_points = unknown_points + list(known_points.keys())
    all_elev = list(X.flatten()) + [known_points[k] for k in known_points]
    all_ci = list(z_score * std_dev) + [0] * len(known_points)
    colors = ['blue'] * len(unknown_points) + ['green'] * len(known_points)
    ax2.errorbar(range(len(all_points)), all_elev, yerr=all_ci, fmt='o', color='gray', ecolor='gray', capsize=5)
    for i, pt in enumerate(all_points):
        ax2.text(i, all_elev[i] + 0.1, f"{pt}\n{all_elev[i]:.2f}", ha='center')
    ax2.set_xticks(range(len(all_points)))
    ax2.set_xticklabels(all_points)
    ax2.set_title("Elevation Profile")
    st.pyplot(fig2)

    # ================================
    # DOWNLOAD RESULT
    # ================================
    st.subheader("📦 Download Results")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_bytes = df_result.to_csv(index=False).encode()
    st.download_button("📥 Download CSV", data=csv_bytes, file_name=f"lsa_results_{timestamp}.csv", mime="text/csv")

    st.success("LSA completed successfully 🎉")
