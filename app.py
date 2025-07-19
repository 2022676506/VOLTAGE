
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

st.set_page_config(page_title="Noah's Ark LSA", layout="centered")

st.title("🏔️ Noah's Ark Least Squares Adjustment (LSA) App")
st.markdown("Upload your elevation observation file (CSV) or enter manually below.")

input_method = st.radio("Select input method:", ["Upload CSV", "Manual Input"])

if input_method == "Upload CSV":
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df)
        st.success("CSV uploaded successfully.")

else:
    st.markdown("### Manual Input")
    bm_count = st.selectbox("Number of Benchmark (BM) Points:", [1, 2, 3, "More than 3"])
    if bm_count == "More than 3":
        bm_count = st.number_input("Enter exact number of BM points:", min_value=4, step=1)
    known_points = {}
    st.markdown("#### Enter Benchmark Points")
    for i in range(int(bm_count)):
        bm_name = st.text_input(f"BM{i+1} Name", key=f"bm_name_{i}")
        bm_value = st.number_input(f"BM{i+1} Elevation", key=f"bm_value_{i}")
        if bm_name:
            known_points[bm_name] = bm_value

    raw_unknown = st.text_input("Unknown Points (comma separated)", "A,B,C")
    unknown_points = [pt.strip() for pt in raw_unknown.split(",") if pt.strip()]
    u = len(unknown_points)
    point_index = {pt: i for i, pt in enumerate(unknown_points)}
    st.write("Unknown Points:", unknown_points)

    n_obs = st.number_input("Number of observations", min_value=1, step=1)
    observations = []
    st.markdown("#### Enter Observations")
    for i in range(int(n_obs)):
        frm = st.text_input(f"From (Obs {i+1})", key=f"from_{i}")
        to = st.text_input(f"To (Obs {i+1})", key=f"to_{i}")
        dh = st.number_input(f"Height Diff (Obs {i+1})", key=f"dh_{i}", format="%.4f")
        observations.append((frm, to, dh))

    if st.button("Run LSA"):
        n = len(observations)
        r = n - u
        st.write("Redundancy:", r)
        if r <= 0:
            st.error("Redundancy must be > 0.")
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
            confidence_level = 0.99
            z_score = stats.norm.ppf(1 - (1 - confidence_level) / 2)

            results = []
            for i, pt in enumerate(unknown_points):
                elevation = X[i, 0]
                uncertainty = std_dev[i]
                ci = z_score * uncertainty
                results.append({
                    "Point": pt,
                    "Elevation (m)": round(elevation, 4),
                    "± Std Dev (m)": round(uncertainty, 4),
                    "99% CI ± (m)": round(ci, 4)
                })
            st.markdown("### 📊 Adjustment Results")
            st.dataframe(pd.DataFrame(results))

            # Residual plot
            fig1, ax1 = plt.subplots()
            residuals = V.flatten()
            outliers = np.abs(residuals) > 3 * np.sqrt(sigma0_squared)
            ax1.bar(np.arange(n), residuals, color=["red" if outliers[i] else "orange" for i in range(n)])
            ax1.axhline(0, color="black", linestyle="--")
            ax1.set_title("Residual Plot")
            ax1.set_xlabel("Observation Index")
            ax1.set_ylabel("Residual (m)")
            st.pyplot(fig1)

            # Elevation plot
            fig2, ax2 = plt.subplots()
            all_points = unknown_points + list(known_points.keys())
            all_values = list(X.flatten()) + [known_points[k] for k in known_points]
            x_pos = range(len(all_points))
            ax2.plot(x_pos, all_values, marker="o", linestyle="-")
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(all_points)
            ax2.set_title("Elevation Profile")
            ax2.set_ylabel("Elevation (m)")
            st.pyplot(fig2)
