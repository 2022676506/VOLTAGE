
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
import base64

st.set_page_config(page_title="Noah's Ark - Elevation Adjustment", layout="wide")

st.title("ðŸ“ Noah's Ark - Elevation Adjustment using Least Squares")

st.markdown("This app calculates adjusted elevations for unknown points using least squares adjustment based on benchmark points and height difference observations.")

# --- Input method ---
input_method = st.radio("Choose input method:", ("Upload CSV", "Upload TXT", "Manual Input"))

# --- Function to process CSV ---
def process_csv(file):
    df = pd.read_csv(file)
    benchmarks = df[df['Type'] == 'BM'][['Point', 'Elevation']].dropna().set_index('Point').to_dict()['Elevation']
    unknowns = df[df['Type'] == 'Unknown']['Point'].unique().tolist()
    observations = df[['From', 'To', 'HeightDiff']].dropna()
    return benchmarks, unknowns, observations

# --- Function to process TXT ---
def process_txt(file):
    content = file.read().decode('utf-8')
    sections = {'[Benchmarks]': {}, '[Unknowns]': [], '[Observations]': []}
    current_section = None
    for line in content.splitlines():
        line = line.strip()
        if not line: continue
        if line in sections:
            current_section = line
        elif current_section == '[Benchmarks]':
            pt, elev = line.split(',')
            sections[current_section][pt.strip()] = float(elev)
        elif current_section == '[Unknowns]':
            sections[current_section].append(line.strip())
        elif current_section == '[Observations]':
            f, t, hd = line.split(',')
            sections[current_section].append((f.strip(), t.strip(), float(hd)))
    return sections['[Benchmarks]'], sections['[Unknowns]'], pd.DataFrame(sections['[Observations]'], columns=['From', 'To', 'HeightDiff'])

# --- Manual input ---
if input_method == "Upload CSV":
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")
    if uploaded_file:
        try:
            benchmarks, unknowns, observations = process_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

elif input_method == "Upload TXT":
    uploaded_file = st.file_uploader("Upload TXT file", type="txt")
    if uploaded_file:
        try:
            benchmarks, unknowns, observations = process_txt(uploaded_file)
        except Exception as e:
            st.error(f"Error reading TXT: {e}")

else:  # Manual input
    st.subheader("Enter Benchmarks")
    num_bm = st.number_input("Number of Benchmarks", min_value=1, max_value=10, step=1)
    benchmarks = {}
    for i in range(int(num_bm)):
        col1, col2 = st.columns(2)
        with col1:
            pt = st.text_input(f"BM Point {i+1} Name", key=f"bm{i}")
        with col2:
            elev = st.number_input(f"Elevation of {pt}", key=f"elev{i}")
        if pt:
            benchmarks[pt] = elev

    st.subheader("Enter Unknown Points")
    unknown_input = st.text_input("Unknown Points (comma-separated)", "A,B,C")
    unknowns = [pt.strip() for pt in unknown_input.split(',') if pt.strip()]

    st.subheader("Enter Observations")
    num_obs = st.number_input("Number of Observations", min_value=1, max_value=50, step=1)
    obs_list = []
    for i in range(int(num_obs)):
        col1, col2, col3 = st.columns(3)
        with col1:
            f = st.text_input(f"From Point {i+1}", key=f"from{i}")
        with col2:
            t = st.text_input(f"To Point {i+1}", key=f"to{i}")
        with col3:
            hd = st.number_input(f"Height Diff {i+1} (m)", key=f"hd{i}")
        if f and t:
            obs_list.append((f, t, hd))
    observations = pd.DataFrame(obs_list, columns=['From', 'To', 'HeightDiff'])

# --- LSA Computation ---
if 'benchmarks' in locals() and 'unknowns' in locals() and 'observations' in locals():
    st.subheader("ðŸ” LSA Results")
    all_points = list(benchmarks.keys()) + unknowns
    n_obs = len(observations)
    u = len(unknowns)
    redundancy = n_obs - u

    if redundancy <= 0:
        st.error("Not enough observations to adjust unknowns (redundancy â‰¤ 0).")
    else:
        unknown_idx = {pt: i for i, pt in enumerate(unknowns)}
        A = np.zeros((n_obs, u))
        L = np.zeros((n_obs, 1))

        for i, row in observations.iterrows():
            f, t, dh = row['From'], row['To'], row['HeightDiff']
            if f in unknowns:
                A[i, unknown_idx[f]] = -1
            elif f in benchmarks:
                L[i, 0] += benchmarks[f]

            if t in unknowns:
                A[i, unknown_idx[t]] = 1
            elif t in benchmarks:
                L[i, 0] -= benchmarks[t]

            L[i, 0] += dh

        N = A.T @ A
        if np.linalg.det(N) == 0:
            st.error("Matrix is singular. Check if your network is connected.")
        else:
            X = np.linalg.inv(N) @ A.T @ L
            V = A @ X - L
            sigma0_sq = (V.T @ V)[0, 0] / redundancy
            std_devs = np.sqrt(np.diag(np.linalg.inv(N)) * sigma0_sq)
            ci_99 = 2.58 * std_devs

            results = pd.DataFrame({
                'Point': unknowns,
                'Adjusted Elevation (m)': X.flatten(),
                'Std Deviation (m)': std_devs,
                'CI Lower': X.flatten() - ci_99,
                'CI Upper': X.flatten() + ci_99
            })

            st.dataframe(results)

            # Export to CSV
            csv = results.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="adjusted_elevations.csv">ðŸ“¥ Download Results as CSV</a>'
            st.markdown(href, unsafe_allow_html=True)

            # Plot residuals
            st.subheader("ðŸ“‰ Residuals")
            fig, ax = plt.subplots()
            ax.bar(range(len(V)), V.flatten())
            ax.axhline(y=2*np.sqrt(sigma0_sq), color='r', linestyle='--', label='Â±2Ïƒâ‚€')
            ax.axhline(y=-2*np.sqrt(sigma0_sq), color='r', linestyle='--')
            ax.set_title("Observation Residuals")
            ax.set_xlabel("Observation Index")
            ax.set_ylabel("Residual (m)")
            ax.legend()
            st.pyplot(fig)

            # Elevation profile plot
            st.subheader("ðŸ“Š Elevation Profile with 99% CI")
            fig2, ax2 = plt.subplots()
            for pt, elev in benchmarks.items():
                ax2.errorbar(pt, elev, yerr=0, fmt='o', label=f"{pt} (BM)")
            for i, row in results.iterrows():
                ax2.errorbar(row['Point'], row['Adjusted Elevation (m)'],
                             yerr=ci_99[i], fmt='o', label=row['Point'])
            ax2.set_ylabel("Elevation (m)")
            ax2.set_title("Elevation Profile")
            ax2.grid(True)
            ax2.legend()
            st.pyplot(fig2)
