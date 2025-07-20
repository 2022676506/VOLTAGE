import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import datetime
from io import StringIO, BytesIO

st.set_page_config(page_title="LSA Elevation Adjustment", layout="centered")
st.title("📐 Least Squares Adjustment for Elevation")
st.markdown("This app estimates unknown point elevations using LSA.")

st.markdown("---")
st.markdown("### ⏱️ Current Time:")
st.info(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Upload or manual entry
st.markdown("### 📥 Choose Input Method")
input_method = st.radio("Select:", ["Manual Input", "Upload CSV", "Upload TXT"])

def round_3(x): return round(x, 3)

if input_method == "Upload CSV":
    file = st.file_uploader("Upload CSV File", type=["csv"])
    if file:
        df = pd.read_csv(file)
        known_points = dict(zip(df[df['Type'] == 'BM']['Point'], df[df['Type'] == 'BM']['Elevation']))
        unknown_points = df[df['Type'] == 'Unknown']['Point'].tolist()
        observations = list(zip(df['From'], df['To'], df['HeightDiff']))
elif input_method == "Upload TXT":
    file = st.file_uploader("Upload TXT File", type=["txt"])
    if file:
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
            elif not line or line.startswith("#"):
                continue
            if mode == "bm":
                label, val = line.split(',')
                known_points[label.strip()] = float(val)
            elif mode == "unknown":
                unknown_points.append(line.strip())
            elif mode == "obs":
                frm, to, diff = line.split(',')
                observations.append((frm.strip(), to.strip(), float(diff)))
else:
    st.markdown("### 📌 Benchmark Points (BM)")
    bm_count = st.number_input("Number of BM:", min_value=1, value=2)
    known_points = {}
    for i in range(bm_count):
        col1, col2 = st.columns(2)
        with col1:
            label = st.text_input(f"BM {i+1} Name", value=f"BM{i+1}")
        with col2:
            elev = st.number_input(f"BM {i+1} Elevation (m)", format="%.3f")
        known_points[label] = elev

    unknown_input = st.text_input("Enter unknown point names (comma separated)", value="A,B,C")
    unknown_points = [pt.strip() for pt in unknown_input.split(",") if pt.strip()]
    
    n_obs = st.number_input("Number of Observations:", min_value=1, value=3)
    observations = []
    st.markdown("### 🧾 Observations")
    for i in range(n_obs):
        col1, col2, col3 = st.columns(3)
        with col1:
            frm = st.text_input(f"From {i+1}", key=f"frm_{i}")
        with col2:
            to = st.text_input(f"To {i+1}", key=f"to_{i}")
        with col3:
            diff = st.number_input(f"Height Diff {i+1} (m)", key=f"diff_{i}", format="%.3f")
        if frm and to:
            observations.append((frm, to, diff))

if st.button("🚀 Run LSA Calculation"):
    if not known_points or not unknown_points or not observations:
        st.error("Please complete all input data.")
    else:
        point_index = {pt: i for i, pt in enumerate(unknown_points)}
        u = len(unknown_points)
        n = len(observations)
        r = n - u
        if r <= 0:
            st.error("Redundancy (r) must be > 0. LSA cannot proceed.")
        else:
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
            sigma0_squared = (V.T @ V)[0, 0] / r
            Cov = sigma0_squared * np.linalg.inv(N)
            std_dev = np.sqrt(np.diag(Cov))
            z = stats.norm.ppf(0.995)  # 99% CI

            results = []
            for i, pt in enumerate(unknown_points):
                elev = round_3(X[i, 0])
                unc = round_3(std_dev[i])
                ci = round_3(z * std_dev[i])
                results.append([pt, elev, unc, elev - ci, elev + ci])

            df_output = pd.DataFrame(results, columns=["Point", "Adjusted Elevation (m)", "Std Dev (m)", "CI Lower", "CI Upper"])
            st.success("✅ LSA Calculation Complete!")

            st.dataframe(df_output)

            csv_buffer = BytesIO()
            df_output.to_csv(csv_buffer, index=False)
            st.download_button("📥 Download CSV", data=csv_buffer.getvalue(), file_name="lsa_results.csv", mime="text/csv")

            # Residual Plot
            fig1, ax1 = plt.subplots()
            residuals = V.flatten()
            threshold = 3 * np.sqrt(sigma0_squared)
            outliers = np.abs(residuals) > threshold
            ax1.bar(np.arange(len(residuals))[~outliers], residuals[~outliers], color='orange', label='Residuals')
            ax1.bar(np.arange(len(residuals))[outliers], residuals[outliers], color='red', label='Outliers')
            ax1.axhline(threshold, color='red', linestyle='--')
            ax1.axhline(-threshold, color='red', linestyle='--')
            ax1.set_title("Residual Plot")
            ax1.set_xlabel("Observation Index")
            ax1.set_ylabel("Residual (m)")
            ax1.legend()
            st.pyplot(fig1)

            # Elevation Profile Plot
            fig2, ax2 = plt.subplots()
            elevation_points = unknown_points + list(known_points.keys())
            elevation_values = list(X.flatten()) + [known_points[k] for k in known_points]
            confidence_intervals = [z * s for s in std_dev] + [0 for _ in known_points]
            x_pos = np.arange(len(elevation_points))
            colors = ['blue'] * len(unknown_points) + ['green'] * len(known_points)

            for i in range(len(elevation_points)):
                ax2.errorbar(x_pos[i], elevation_values[i], yerr=confidence_intervals[i],
                             fmt='o', color=colors[i], capsize=5)
                ax2.text(x_pos[i], elevation_values[i] + 0.1, f"{elevation_points[i]}\n{elevation_values[i]:.3f}",
                         ha='center', fontsize=8)

            ax2.plot(x_pos, elevation_values, linestyle='--', color='gray', alpha=0.6)
            ax2.set_title("Elevation Profile with 99% CI")
            ax2.set_xlabel("Point Index")
            ax2.set_ylabel("Elevation (m)")
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(elevation_points)
            st.pyplot(fig2)
