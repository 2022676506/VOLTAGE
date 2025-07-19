import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from io import StringIO
from datetime import datetime
import base64

# === Page Setup ===
st.set_page_config(page_title="Noah's Ark LSA App", layout="wide")
st.title("🗺️ Noah's Ark - Elevation Adjustment using LSA")
st.markdown("#### Least Squares Adjustment for Elevation with Benchmarks (BM) and Observations")

# === Logo (Optional) ===
logo_path = "logo.png"
try:
    with open(logo_path, "rb") as f:
        logo_base64 = base64.b64encode(f.read()).decode("utf-8")
    st.markdown(f'<img src="data:image/png;base64,{logo_base64}" width="150"/>', unsafe_allow_html=True)
except:
    st.info("No logo found or not loaded.")

# === Timestamp ===
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.caption(f"📅 Generated on: {timestamp}")

# === Input Method Selection ===
method = st.radio("Select input method:", ["Upload CSV", "Upload TXT", "Manual Input"])

known_points = {}
unknown_points = []
observations = []

# === Load CSV ===
if method == "Upload CSV":
    csv_file = st.file_uploader("Upload CSV file", type=["csv"])
    if csv_file:
        try:
            df = pd.read_csv(csv_file)
            required_cols = {'Type', 'Point', 'From', 'To', 'HeightDiff'}
            if not required_cols.issubset(df.columns):
                st.error("CSV must contain columns: Type, Point, From, To, HeightDiff.")
                st.stop()
            known_df = df[df['Type'] == 'BM']
            unknown_df = df[df['Type'] == 'Unknown']
            known_points = dict(zip(known_df['Point'], known_df['Elevation']))
            unknown_points = unknown_df['Point'].tolist()
            observations = list(zip(df['From'], df['To'], df['HeightDiff']))
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            st.stop()

# === Load TXT ===
elif method == "Upload TXT":
    txt_file = st.file_uploader("Upload TXT file", type=["txt"])
    if txt_file:
        try:
            lines = txt_file.read().decode().splitlines()
            mode = None
            for line in lines:
                line = line.strip()
                if line.lower() == "[benchmarks]":
                    mode = 'bm'
                    continue
                elif line.lower() == "[unknowns]":
                    mode = 'unknown'
                    continue
                elif line.lower() == "[observations]":
                    mode = 'obs'
                    continue
                if not line or line.startswith("#"):
                    continue
                if mode == 'bm':
                    label, val = line.split(',')
                    known_points[label.strip()] = float(val)
                elif mode == 'unknown':
                    unknown_points.append(line.strip())
                elif mode == 'obs':
                    frm, to, diff = line.split(',')
                    observations.append((frm.strip(), to.strip(), float(diff)))
        except Exception as e:
            st.error(f"Error reading TXT: {e}")
            st.stop()

# === Manual Input ===
elif method == "Manual Input":
    bm_count = st.number_input("Number of benchmark points?", min_value=1, max_value=10, value=2)
    for i in range(bm_count):
        col1, col2 = st.columns(2)
        with col1:
            label = st.text_input(f"Benchmark name {i+1}", key=f"bm_label_{i}")
        with col2:
            value = st.number_input(f"Elevation of {label}", key=f"bm_val_{i}")
        if label:
            known_points[label] = value

    unknown_str = st.text_input("Enter unknown point names (comma-separated):", "A,B,C")
    unknown_points = [pt.strip() for pt in unknown_str.split(",") if pt.strip()]

    n_obs = st.number_input("Number of observations", min_value=1, value=3)
    for i in range(n_obs):
        st.markdown(f"**Observation {i+1}**")
        frm = st.text_input("From", key=f"obs_from_{i}")
        to = st.text_input("To", key=f"obs_to_{i}")
        dh = st.number_input("Height Difference (m)", key=f"obs_dh_{i}")
        if frm and to:
            observations.append((frm, to, dh))

# === Proceed if data valid ===
if known_points and unknown_points and observations:
    st.subheader("📌 Input Overview")
    st.write(f"Known Points: {known_points}")
    st.write(f"Unknown Points: {unknown_points}")
    obs_df = pd.DataFrame(observations, columns=["From", "To", "Height Difference (m)"])
    st.dataframe(obs_df)

    # === Matrix Setup ===
    point_index = {pt: i for i, pt in enumerate(unknown_points)}
    u = len(unknown_points)
    n = len(observations)
    r = n - u

    if r <= 0:
        st.error("Redundancy (r = n - u) must be > 0. Not enough observations.")
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

    # === Least Squares Adjustment ===
    AT = A.T
    N = AT @ A
    U = AT @ L
    try:
        X = np.linalg.solve(N, U)
    except np.linalg.LinAlgError:
        st.error("Matrix N is singular. The network may not be connected.")
        st.stop()

    V = A @ X - L
    sigma0_squared = (V.T @ V)[0, 0] / r
    Cov = sigma0_squared * np.linalg.inv(N)
    std_dev = np.sqrt(np.diag(Cov))

    # === Result Table ===
    confidence_level = 0.99
    z_score = stats.norm.ppf(1 - (1 - confidence_level) / 2)
    results = []
    for i, pt in enumerate(unknown_points):
        elev = X[i, 0]
        unc = std_dev[i]
        ci = z_score * unc
        results.append((pt, elev, unc, elev - ci, elev + ci))

    df_result = pd.DataFrame(results, columns=["Point", "Adjusted Elevation (m)", "Std Deviation (m)", "CI Lower", "CI Upper"])
    st.subheader("✅ Adjusted Elevation Results")
    st.dataframe(df_result.style.format(precision=4))

    # === Download Button ===
    csv = df_result.to_csv(index=False).encode()
    st.download_button("📥 Download Results as CSV", data=csv, file_name="adjusted_results.csv", mime='text/csv')

    # === Residual Plot ===
    st.subheader("📊 Residual Plot with Outliers")
    threshold = 3 * np.sqrt(sigma0_squared)
    res = V.flatten()
    outliers = np.abs(res) > threshold
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(np.arange(len(res)), res, color=['red' if o else 'orange' for o in outliers])
    ax.axhline(0, color='black', linestyle='--')
    ax.axhline(threshold, color='red', linestyle=':')
    ax.axhline(-threshold, color='red', linestyle=':')
    ax.set_xticks(np.arange(len(res)))
    ax.set_xticklabels([f"Obs {i+1}" for i in range(len(res))], rotation=45)
    ax.set_ylabel("Residual (m)")
    ax.set_title("Residuals with Outlier Detection")
    st.pyplot(fig)

    # === Elevation Profile Plot ===
    st.subheader("📈 Adjusted Elevation Profile")
    elevation_points = unknown_points + list(known_points.keys())
    elevation_values = list(X.flatten()) + [known_points[k] for k in known_points]
    confidence_intervals = [z_score * e for e in std_dev] + [0 for _ in known_points]
    colors = ['blue' if pt in unknown_points else 'green' for pt in elevation_points]
    markers = ['o' if pt in unknown_points else 's' for pt in elevation_points]
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    x_pos = list(range(len(elevation_points)))
    ax2.plot(x_pos, elevation_values, linestyle='-', color='gray', alpha=0.5)
    for i, pt in enumerate(elevation_points):
        ax2.errorbar(x_pos[i], elevation_values[i], yerr=confidence_intervals[i], fmt=markers[i],
                     color=colors[i], ecolor='gray', capsize=4, markersize=7)
        ax2.text(x_pos[i], elevation_values[i]+0.05, f"{pt}\n{elevation_values[i]:.2f}m", ha='center', fontsize=8)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(elevation_points)
    ax2.set_ylabel("Elevation (m)")
    ax2.set_title("Elevation Profile with 99% Confidence Intervals")
    st.pyplot(fig2)

else:
    st.warning("Please provide valid data using one of the input methods above.")
