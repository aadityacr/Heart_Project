# app.py
import streamlit as st
import numpy as np
import tensorflow as tf
import wfdb
import os, base64, re, zipfile, shutil
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer

# -----------------------------------
# PAGE SETUP
# -----------------------------------
st.set_page_config(page_title="Cardiovascular Diagnostic System",
                   page_icon="🫀", layout="wide")

# -----------------------------------
# GLOBAL BACKGROUND (ALL TABS)
# -----------------------------------
def set_full_background(image_file: str):
    """Apply full-viewport background image to entire app."""
    if not os.path.exists(image_file):
        return
    with open(image_file, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    st.markdown(f"""
    <style>
      /* Background image everywhere */
      [data-testid="stAppViewContainer"] {{
        background: url("data:image/jpg;base64,{b64}") center center / cover no-repeat fixed;
      }}
      /* Transparent chrome so image shows */
      [data-testid="stHeader"], .block-container {{ background: transparent !important; }}

      /* Subtle dark overlay for readability */
      [data-testid="stAppViewContainer"]::before {{
        content:""; position: fixed; inset: 0; background: rgba(0,0,0,.45); pointer-events: none;
      }}

      /* Readable default text */
      h1,h2,h3,h4,h5,h6, p, label, span, div, li, th, td {{ color: #fff; }}

      /* Light blur on widgets/tables/alerts/metrics */
      .stDataFrame, .stAlert, .stMetric {{ backdrop-filter: blur(1px); }}

      /* ---- NAV buttons (single row, no bar) ---- */
      .nav-row {{
        display:flex; justify-content:center; gap:16px; margin: 8px 0 18px 0;
      }}
      .nav-row .stButton > button {{
        background: rgba(0,0,0,.65);
        color: #fff;
        border: 1px solid rgba(255,255,255,.25);
        padding: 8px 16px;
        border-radius: 12px;
        font-weight: 700;
      }}
      .nav-row .stButton > button:hover {{ filter: brightness(1.15); }}
      .active .stButton > button {{
        background: #22c55e !important; border-color: #22c55e !important; color:#111 !important;
      }}

      /* Nice table hover */
      .dataframe tbody tr:hover {{ background: rgba(255,255,255,.06); }}
    </style>
    """, unsafe_allow_html=True)

set_full_background("background.jpg")  # <- ensure this exists

# -----------------------------------
# NAV STATE (TABS)
# -----------------------------------
if "page" not in st.session_state:
    st.session_state.page = "Home"

tabs = ["Home", "Single Prediction", "Bulk Prediction", "Performance"]

st.markdown('<div class="nav-row">', unsafe_allow_html=True)
cols = st.columns(len(tabs))
for i, name in enumerate(tabs):
    with cols[i]:
        # wrap active for green style
        if st.session_state.page == name:
            st.markdown('<div class="active">', unsafe_allow_html=True)
        else:
            st.markdown('<div>', unsafe_allow_html=True)
        if st.button(name, key=f"tab-{name.replace(' ', '-')}"):
            st.session_state.page = name
        st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------
# MODEL + HELPERS
# -----------------------------------
DEFAULT_CLASSES = [
    "Normal Sinus Rhythm",
    "Myocardial Infarction",
    "Atrial Fibrillation",
    "Arrhythmia",
    "ST Depression"
]

@st.cache_resource
def load_system_resources():
    model, classes = None, DEFAULT_CLASSES
    if os.path.exists("Five_Class_Model.h5"):
        try:
            model = tf.keras.models.load_model("Five_Class_Model.h5")
        except Exception as e:
            st.error(f"Model load failed: {e}")
    if os.path.exists("five_classes.npy"):
        try:
            raw = np.load("five_classes.npy", allow_pickle=True).tolist()
            if isinstance(raw, (list, np.ndarray)) and len(raw) == 5:
                classes = [str(x) for x in raw]
        except Exception:
            pass
    return model, classes

model, CLASS_NAMES = load_system_resources()

def get_risk_assessment(dx: str):
    high = {"Myocardial Infarction","Atrial Fibrillation","ST Depression"}
    if dx in high: return "HIGH RISK","error"
    if dx == "Arrhythmia": return "MODERATE RISK","warning"
    if "Normal" in dx: return "LOW RISK","success"
    return "UNCERTAIN","info"

# --- Robust Age/Sex extractor ---
def extract_metadata(text: str):
    m1 = re.search(r"Age[: ]+(\d{1,3})", text or "", re.IGNORECASE)
    m2 = re.search(r"#\s*Age[: ]+(\d{1,3})", text or "", re.IGNORECASE)
    m3 = re.search(r"\bage\s*=\s*(\d{1,3})\b", text or "", re.IGNORECASE)
    m4 = re.search(r"(\d{1,3})\s*yrs?", text or "", re.IGNORECASE)

    if m1: age = int(m1.group(1))
    elif m2: age = int(m2.group(1))
    elif m3: age = int(m3.group(1))
    elif m4: age = int(m4.group(1))
    else: age = 55

    s1 = re.search(r"Sex[: ]+(Male|Female|M|F)\b", text or "", re.IGNORECASE)
    s2 = re.search(r"#\s*Sex[: ]+(Male|Female|M|F)\b", text or "", re.IGNORECASE)
    sex_raw = (s1 or s2).group(1) if (s1 or s2) else "M"
    sex = 1 if sex_raw.strip().upper() in ["F", "FEMALE"] else 0

    return age, sex

def _safe_standardize(x):
    mu, sd = np.mean(x), np.std(x)
    return (x-mu)/sd if (sd and np.isfinite(sd)) else x*0

def process_signal(record_path: str):
    try:
        sig, _ = wfdb.rdsamp(record_path)
        if sig.ndim != 2: return None
        if sig.shape[0] > 2000: sig = sig[::5, :]
        sig = _safe_standardize(sig)
        T, L = sig.shape
        if L < 12: sig = np.hstack([sig, np.zeros((T, 12-L))])
        if L > 12: sig = sig[:, :12]
        target = 1000
        if T > target: sig = sig[:target, :]
        if T < target: sig = np.vstack([sig, np.zeros((target-T, 12))])
        return sig.reshape(1, target, 12).astype(np.float32)
    except Exception:
        return None

def _softmax_if_needed(x):
    return (np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))) if (np.any(x < 0) or np.any(x > 1)) else x

# -----------------------------------
# PAGES
# -----------------------------------
TITLE = "Cardiovascular disease prediction using deep learning techniques"

if st.session_state.page == "Home":
    st.markdown(f"<h1 style='text-align:center;font-size:48px;font-weight:900;'>{TITLE}</h1>", unsafe_allow_html=True)
    a,b,c = st.columns(3)
    with a: st.info("**Multi-Modal Fusion**  \nECG + Age/Sex.")
    with b: st.info("**5-Class Model**  \nNormal • MI • AF • Arrhythmia • ST-Depression")
    with c: st.info("**Fast Screening**  \n< 0.5s per patient (CPU).")

elif st.session_state.page == "Single Prediction":
    st.header("Single Prediction")
    uploaded = st.file_uploader("Upload patient record (.zip with .hea + .dat)", type="zip")
    if uploaded:
        if not model:
            st.error("Model file 'Five_Class_Model.h5' not found.")
        else:
            tmp = "temp_single"
            if os.path.exists(tmp): shutil.rmtree(tmp)
            os.makedirs(tmp)
            with zipfile.ZipFile(uploaded, "r") as z: z.extractall(tmp)
            hea_files = [f for f in os.listdir(tmp) if f.endswith(".hea")]
            if not hea_files:
                st.error("No .hea file found in the ZIP.")
            else:
                selected_file = st.selectbox("Select Patient Record", hea_files)
                header_path = os.path.join(tmp, selected_file)
                try:
                    with open(header_path, "r") as f: header_text = f.read()
                except: header_text = ""
                age, sex = extract_metadata(header_text)
                st.subheader("Patient Data")
                st.write(f"**Age:** {age} | **Sex:** {'Female' if sex == 1 else 'Male'}")
                
                record_path = os.path.join(tmp, selected_file.replace(".hea", ""))
                sig = process_signal(record_path)
                if sig is None:
                    st.error("ECG signal could not be processed.")
                else:
                    clin = np.array([[age/100.0, sex]], dtype=np.float32)
                    try:
                        probs = _softmax_if_needed(model.predict([sig, clin], verbose=0)[0])
                        idx = int(np.argmax(probs))
                        dx = CLASS_NAMES[idx]
                        conf = float(probs[idx])
                    except Exception as e:
                        st.error(f"Inference error: {e}")
                        probs = None
                    if probs is not None:
                        risk, tone = get_risk_assessment(dx)
                        (st.error if tone=="error" else st.warning if tone=="warning" else st.success if tone=="success" else st.info)(risk)
                        c1, c2 = st.columns(2)
                        c1.metric("Primary Diagnosis", dx)
                        c2.metric("AI Confidence", f"{conf*100:.1f}%")
                        st.subheader("Class Probabilities")
                        df = pd.DataFrame({"Class": CLASS_NAMES, "Probability": [float(p) for p in probs]}).sort_values("Probability", ascending=False, ignore_index=True)
                        st.dataframe(df.style.format({"Probability":"{:.2%}"}), use_container_width=True)
                        st.subheader("ECG Lead I")
                        fig, ax = plt.subplots(figsize=(10,3))
                        ax.plot(sig[0,:800,0], linewidth=1)
                        ax.grid(True, alpha=.3, linestyle="--")
                        st.pyplot(fig)

elif st.session_state.page == "Bulk Prediction":
    st.header("Bulk Prediction")
    up = st.file_uploader("Upload dataset (.zip with multiple .hea/.dat pairs)", type="zip")
    if up:
        if not model:
            st.error("Model file 'Five_Class_Model.h5' not found.")
        elif st.button("Start Batch Analysis"):
            tmp = "temp_batch"
            if os.path.exists(tmp): shutil.rmtree(tmp)
            os.makedirs(tmp)
            with zipfile.ZipFile(up, "r") as z: z.extractall(tmp)
            heas = []
            for r,_,files in os.walk(tmp):
                heas += [os.path.join(r,f) for f in files if f.endswith(".hea")]
            st.info(f"Queue: {len(heas)} records.")
            results, bar = [], st.progress(0)
            for i, hp in enumerate(heas):
                rec = hp.replace(".hea","")
                try:
                    with open(hp,"r") as f: content = f.read()
                except: content = ""
                age, sex = extract_metadata(content)
                sig = process_signal(rec)
                if sig is not None:
                    clin = np.array([[age/100.0, sex]], dtype=np.float32)
                    try:
                        probs = _softmax_if_needed(model.predict([sig, clin], verbose=0)[0])
                        idx = int(np.argmax(probs))
                        dx = CLASS_NAMES[idx]
                        conf = float(probs[idx])
                        risk,_ = get_risk_assessment(dx)
                        results.append({"Patient ID": os.path.basename(hp).split(".")[0], "Age": int(age), "Sex": "Female" if sex==1 else "Male", "Diagnosis": dx, "Risk Level": risk, "Confidence": f"{conf*100:.1f}%"})
                    except:
                        results.append({"Patient ID": os.path.basename(hp).split(".")[0], "Age": int(age), "Sex": "Female" if sex==1 else "Male", "Diagnosis": "ERROR", "Risk Level": "UNCERTAIN", "Confidence": "—"})
                bar.progress(min((i+1)/max(1,len(heas)),1.0))
            df = pd.DataFrame(results)
            st.divider(); st.subheader("Batch Summary")
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Total", len(df))
            c2.metric("High", len(df[df["Risk Level"].str.contains("HIGH", na=False)]))
            c3.metric("Moderate", len(df[df["Risk Level"].str.contains("MODERATE", na=False)]))
            c4.metric("Low", len(df[df["Risk Level"].str.contains("LOW", na=False)]))
            st.dataframe(df, use_container_width=True)
            st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"), "Cardio_Batch_Report.csv", "text/csv")

# -----------------------------------
# PERFORMANCE TAB (UPDATED CODE)
# -----------------------------------
elif st.session_state.page == "Performance":
    st.header("Performance Metrics")
    st.info("Visualizations generated from the Test Set (Unseen Data).")

    t1, t2 = st.tabs(["Confusion Matrices", "Training Analytics"])

    with t1:
        st.subheader("Confusion Matrices (Per Class)")
        # FIX: Filter out bad files and ensure valid images exist
        image_files = []
        for f in os.listdir('.'):
            if f.endswith('.png') and "Matrix" in f:
                if "Matrix_.png" in f: continue # Skip corrupt/empty matrix file
                image_files.append(f)
        
        if image_files:
            cols = st.columns(2)
            for i, img_file in enumerate(sorted(image_files)):
                # Clean filename for display
                caption = img_file.replace("Matrix_", "").replace(".png", "").replace("_", " ")
                with cols[i % 2]:
                    st.image(img_file, caption=caption, use_container_width=True)
        else:
            st.warning("⚠️ No Confusion Matrix images found. Please run 'generate_assets.py' to create them.")

    with t2:
        st.subheader("Model Accuracy")
        if os.path.exists("Performance_Graph.png"):
            st.image("Performance_Graph.png", caption="Model Accuracy per Disease", use_container_width=True)
        else:
            st.warning("⚠️ 'Performance_Graph.png' not found. Please run 'generate_assets.py'.")

# Footer status
st.write("")
st.caption("System Status: Online" if model else "System Status: Model Missing (place Five_Class_Model.h5)")