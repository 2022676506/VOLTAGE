import streamlit as st
import pandas as pd, numpy as np, datetime, io
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

# ————— Streamlit Page Setup —————
st.set_page_config(page_title="LSA Elevation Adjuster", page_icon="📏", layout="centered")
logo_path = "logo.png"
if os.path.exists(logo_path):
    st.image(logo_path, width=100)
st.title("📐 LSA Elevation Adjustment App")
st.markdown(f"🕒 **{datetime.datetime.now():%A, %d %B %Y ‑ %I:%M %p}**")

st.markdown("""
This tool performs a Least Squares Adjustment (LSA) for elevation data.  
**Upload** CSV or TXT files, or **enter manually**, to compute adjusted heights and confidence intervals.
""")

# ————— Data Input Section —————
method = st.radio("Select Input Method:", ["Manual", "Upload CSV", "Upload TXT"])

known_points, unknown_points, observations = {}, [], []
if method == "Upload CSV":
    ufile = st.file_uploader("CSV file", type=["csv"])
    if ufile:
        df = pd.read_csv(ufile)
        known_points = dict(zip(df[df.Type=="BM"].Point, df[df.Type=="BM"].Elevation))
        unknown_points = df[df.Type=="Unknown"].Point.tolist()
        observations = list(df[["From","To","HeightDiff"]].itertuples(index=False, name=None))
elif method == "Upload TXT":
    ufile = st.file_uploader("TXT file", type=["txt"])
    if ufile:
        lines = ufile.read().decode().splitlines()
        mode = None
        for L in lines:
            L = L.strip()
            if L.lower()=="[benchmarks]": mode="bm"; continue
            if L.lower()=="[unknowns]": mode="un"; continue
            if L.lower()=="[observations]": mode="obs"; continue
            if not L or L.startswith("#"): continue
            if mode=="bm":
                k, v = L.split(","); known_points[k.strip()] = float(v)
            elif mode=="un":
                unknown_points.append(L)
            elif mode=="obs":
                f,t,d = L.split(","); observations.append((f.strip(), t.strip(), float(d)))
else:
    n_bm = st.number_input("Number of benchmarks", min_value=1, max_value=10, value=2, key="bm_input")
    for i in range(n_bm):
        k = st.text_input(f"BM{i+1} name", key=f"bm_nm_{i}")
        v = st.number_input(f"{k} elevation (m)", key=f"bm_val_{i}")
        if k: known_points[k] = v
    up = st.text_input("Unknown point names (comma-separated)", value="P1,P2")
    unknown_points = [x.strip() for x in up.split(",") if x.strip()]
    n_obs = st.number_input("Number of observations", min_value=1, value=3)
    for i in range(n_obs):
        f = st.text_input(f"Obs{i+1} From", key=f"frm_{i}")
        t = st.text_input(f"Obs{i+1} To", key=f"to_{i}")
        d = st.number_input(f"Obs{i+1} Height diff", value=0.0, key=f"diff_{i}")
        if f and t: observations.append((f, t, d))

# ————— Perform LSA when ready —————
if (method=="Manual" and known_points and unknown_points and observations) or observations:
    if st.button("🔎 Compute LSA"):
        u = len(unknown_points)
        n = len(observations)
        r = n - u
        if r <= 0:
            st.error("Insufficient redundancy (r ≤ u). Add more observations.")
        else:
            # Build A, L
            A = np.zeros((n,u)); L = np.zeros((n,1))
            idx = {pt:i for i, pt in enumerate(unknown_points)}
            for i,(f,t,d) in enumerate(observations):
                if f in idx: A[i,idx[f]] = -1
                else: L[i]+=known_points.get(f,0)
                if t in idx: A[i,idx[t]] = 1
                else: L[i]-=known_points.get(t,0)
                L[i]+=d

            # Compute X, residuals, variance
            AT,N = A.T, None
            N = AT@A
            X = np.linalg.inv(N)@(AT@L)
            V = A@X - L
            s0_sq = float((V.T@V)/(r))
            Cov = s0_sq * np.linalg.inv(N)
            std = np.sqrt(np.diag(Cov))
            z = norm.ppf(0.995)

            # Result dataframe
            df_out = pd.DataFrame({
                "Point": unknown_points,
                "Adjusted": X.flatten(),
                "StdDev": std,
                "CI Low": X.flatten()-z*std,
                "CI High":X.flatten()+z*std
            }).round(4)
            st.subheader("📊 Adjusted Elevations")
            st.dataframe(df_out)

            st.success(f"Variance factor σ₀² = {s0_sq:.6f}")

            # CSV download
            buf = io.StringIO(); df_out.to_csv(buf, index=False)
            st.download_button("⬇️ Download Results CSV", buf.getvalue(), "results.csv", "text/csv")

            # Residual plot
            st.subheader("📉 Residual Plot")
            fig,ax = plt.subplots()
            res = V.flatten()
            thresh = 3*np.sqrt(s0_sq)
            ax.bar(range(n), res, color=['red' if abs(x)>thresh else 'orange' for x in res])
            ax.axhline(0, color='black', linestyle='--')
            ax.set_xlabel("Observation #"); ax.set_ylabel("Residual (m)")
            ax.legend(handles=[
                plt.Line2D([0],[0],color='orange',label='Normal'),
                plt.Line2D([0],[0],color='red',label='Outlier')
            ])
            st.pyplot(fig)

            # Elevation plot
            st.subheader("📈 Elevation Profile")
            pts = unknown_points + list(known_points.keys())
            vals = list(X.flatten()) + [known_points[k] for k in known_points]
            cis = list(z*std) + [0]*len(known_points)
            fig2,ax2 = plt.subplots()
            x = np.arange(len(pts))
            ax2.errorbar(x, vals, yerr=cis, fmt='o', ecolor='gray', capsize=4, color='blue')
            ax2.plot(x, vals, linestyle='--', color='lightgray')
            ax2.set_xticks(x); ax2.set_xticklabels(pts)
            ax2.set_ylabel("Elevation (m)")
            st.pyplot(fig2)
