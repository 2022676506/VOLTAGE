# app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import datetime
from io import StringIO, BytesIO

# --- PAGE SETUP ---
st.set_page_config(page_title="LSA Elevation Adjuster", layout="centered", page_icon="🗺️")
st.image("assets/logo.png", width=100)
st.title("🎯 Least Squares Adjustment (Elevation Computation)")
st.markdown("### 📏 Built for Land Surveying Professionals")
st.caption(f"📅 {datetime.datetime.now().strftime('%A, %d %B %Y %H:%M:%S')}")

st.divider()

# --- INPUT MODE ---
input_mode = st.radio("📥 Choose Input Method", ["Manual Input", "Upload CSV", "Upload TXT"])

known_points = {}
unknown_points = []
observations = []

if input_mode == "Upload CSV":
    uploaded_csv = st.file_uploader("Upload your CSV file", type="csv")
    if uploaded_csv:
        df = pd.read_csv(uploaded_csv)
        known_points = dict(zip(df[df['Type'] == 'BM']['Point'], df[df['Type'] == 'BM']['Elevation']))
        unknown_points = df[df['Type'] == 'Unknown']['Point'].tolist()
        observations = list(zip(df['From'], df['To'], df['HeightDiff']))

elif input_mode == "Upload TXT":
    uploaded_txt = st.file_uploader("Upload your TXT file", type="txt")
    if uploaded_txt:
        content = uploaded_txt.read().decode("utf-8").splitlines()
        mode = None
        for line in content:
            line = line.strip()
            if line.lower() == "[benchmarks]":
                mode = 'bm'
            elif line.lower() == "[unknowns]":
                mode = 'unknown'
            elif line.lower() == "[observations]":
                mode = 'obs'
            elif not line or line.startswith("#"):
                continue
            elif mode == 'bm':
                label, val = line.split(',')
                known_points[label.strip()] = float(val)
            elif mode == 'unknown':
                unknown_points.append(line.strip())
            elif mode == 'obs':
                frm, to, diff = line.split(',')
                observations.append((frm.strip(), to.strip(), float(diff)))
else:
    bm_count = st.number_input("🔢 Number of Benchmark Points", min_value=2, step=1)
    st.markdown("### 📌 Enter Benchmark Points")
    for i in range(bm_count):
        col1, col2 = st.columns(2)
        with col1:
            label = st.text_input(f"Benchmark {i+1} Name", key=f"bm_name_{i}")
        with col2:
            value = st.number_input(f"Elevation for {label}", format="%.3f", key=f"bm_val_{i}")
        if label:
            known_points[label] = value

    unknown_raw = st.text_input("📍 Unknown Points (comma-separated)", value="A,B,C")
    unknown_points = [pt.strip() for pt in unknown_raw.split(",") if pt.strip()]

    n_obs = st.number_input("🔁 Number of Observations", min_value=1, step=1)
    st.markdown("### ⬆️ Observations")
    for i in range(int(n_obs)):
        col1, col2, col3 = st.columns(3)
        with col1:
            frm = st.text_input(f"From (Obs {i+1})", key=f"from_{i}")
        with col2:
            to = st.text_input(f"To (Obs {i+1})", key=f"to_{i}")
        with col3:
            diff = st.number_input(f"Height Diff (m) {i+1}", format="%.3f", key=f"diff_{i}")
        if frm and to:
            observations.append((frm, to, diff))

# --- PROCESSING ---
if st.button("🚀 Compute LSA"):
    if not known_points or not unknown_points or not observations:
        st.error("❌ Please complete all input sections.")
    else:
        # Create matrix
        u = len(unknown_points)
        n = len(observations)
        point_index = {pt: i for i, pt in enumerate(unknown_points)}
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

        AT = A.T
        N = AT @ A
        U = AT @ L
        X = np.linalg.inv(N) @ U
        V = A @ X - L
        r = n - u
        sigma0_squared = (V.T @ V)[0, 0] / r
        Cov = sigma0_squared * np.linalg.inv(N)
        std_dev = np.sqrt(np.diag(Cov))
        z_score = stats.norm.ppf(1 - 0.005)

        # --- Output ---
        st.success("✅ Adjustment Completed!")
        result_df = pd.DataFrame({
            "Point": unknown_points,
            "Elevation (m)": X.flatten(),
            "Std Dev (m)": std_dev,
            "CI Lower (99%)": X.flatten() - z_score * std_dev,
            "CI Upper (99%)": X.flatten() + z_score * std_dev
        })

        st.dataframe(result_df.style.format({"Elevation (m)": "{:.3f}", "Std Dev (m)": "{:.3f}"}))

        # Download button
        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Result CSV", csv, f"lsa_result_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv")

        # --- Plotting ---
        st.markdown("### 📉 Residual Plot")
        fig1, ax1 = plt.subplots()
        residuals = V.flatten()
        threshold = 3 * np.sqrt(sigma0_squared)
        outliers = np.abs(residuals) > threshold
        ax1.bar(np.arange(len(residuals)), residuals, color=['red' if o else 'orange' for o in outliers])
        ax1.axhline(0, color='black', linestyle='--')
        ax1.set_title("Residuals")
        ax1.set_xlabel("Observation Index")
        ax1.set_ylabel("Residual (m)")
        st.pyplot(fig1)

        st.markdown("### 🗺️ Adjusted Elevation Profile")
        all_pts = unknown_points + list(known_points.keys())
        all_vals = list(X.flatten()) + [known_points[k] for k in known_points]
        ci_vals = [z_score * e for e in std_dev] + [0 for _ in known_points]
        fig2, ax2 = plt.subplots()
        x_pos = list(range(len(all_pts)))
        ax2.errorbar(x_pos, all_vals, yerr=ci_vals, fmt='o', capsize=5)
        for i, txt in enumerate(all_pts):
            ax2.text(x_pos[i], all_vals[i] + 0.05, f"{txt}\n{all_vals[i]:.3f}", ha='center')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(all_pts)
        ax2.set_title("Elevation Profile with 99% CI")
        ax2.set_ylabel("Elevation (m)")
        ax2.grid(True)
        st.pyplot(fig2)
