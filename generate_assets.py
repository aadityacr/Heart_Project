import os
import glob
import zipfile
import numpy as np
import tensorflow as tf
import wfdb
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import multilabel_confusion_matrix

# ==========================================
# 1. CONFIGURATION
# ==========================================
MODEL_FILE = 'Five_Class_Model.h5'
CLASSES_FILE = 'five_classes.npy'

# Auto-detect zip file
zips = glob.glob("*.zip")
if not zips:
    print("❌ Error: No Zip file found. Please put 'Demo_Diverse_1000.zip' in this folder.")
    exit()
ZIP_FILE = zips[0]
print(f"📂 Using Data Source: {ZIP_FILE}")

# ==========================================
# 2. LOAD MODEL & CLASSES
# ==========================================
print("🧠 Loading AI Brain...")
model = tf.keras.models.load_model(MODEL_FILE)
classes = np.load(CLASSES_FILE, allow_pickle=True)
mlb = MultiLabelBinarizer()
mlb.fit([classes])

# Clinical Dictionary
DIAGNOSIS_MAP = {
    '426783006': 'Normal Sinus Rhythm', '164889003': 'Atrial Fibrillation',
    '164865005': 'Myocardial Infarction', '426177001': 'Arrhythmia',
    '427084000': 'Arrhythmia', '164909002': 'Arrhythmia',
    '251146004': 'Arrhythmia', '164931005': 'ST Depression',
    '284470004': 'Arrhythmia', '429622005': 'ST Depression',
    '164873001': 'Arrhythmia', '39732003': 'Arrhythmia',
    '47665007': 'Arrhythmia', '59118001': 'Arrhythmia',
    '270492004': 'Arrhythmia', '164884008': 'Arrhythmia',
    '164934002': 'Arrhythmia', '427393009': 'Arrhythmia'
}

# ==========================================
# 3. DATA EXTRACTION (FIXED FOR 2 INPUTS)
# ==========================================
print("extracting data... (This might take 1 minute)")
temp_dir = "temp_gen_assets"
if not os.path.exists(temp_dir): os.makedirs(temp_dir)

with zipfile.ZipFile(ZIP_FILE, 'r') as z:
    z.extractall(temp_dir)

all_headers = [os.path.join(root, f) for root, _, files in os.walk(temp_dir) for f in files if f.endswith('.hea')]

X_signal = []
X_clinical = [] # <--- ADDED THIS
y_true = []

# Regex for parsing
dx_pattern = re.compile(r"Dx\D*([\d,]+)", re.IGNORECASE)
age_pattern = re.compile(r"Age\D*(\d+)", re.IGNORECASE)
sex_pattern = re.compile(r"Sex\D*([MF])", re.IGNORECASE)

print(f"🔍 Scanning {len(all_headers)} files...")

for i, hea_path in enumerate(all_headers):
    if i >= 1000: break # Process max 1000 files
    
    try:
        with open(hea_path, 'r') as f:
            content = f.read()
            
            # 1. Get Ground Truth Labels
            dx_match = dx_pattern.search(content)
            if dx_match:
                codes = dx_match.group(1).split(",")
                labels = []
                for c in codes:
                    clean = c.strip()
                    if clean in DIAGNOSIS_MAP:
                        target = DIAGNOSIS_MAP[clean]
                        if target in classes:
                            labels.append(target)
                
                if not labels: continue # Skip if no relevant disease
                
                # 2. Get Signal (Input 1)
                rec_path = hea_path.replace(".hea", "")
                sig, _ = wfdb.rdsamp(rec_path)
                
                if sig.shape[0] > 2000: sig = sig[::5, :] # Downsample
                sig = (sig - np.mean(sig)) / np.std(sig) # Normalize
                
                target_len = 1000
                if sig.shape[0] > target_len: sig = sig[:target_len, :]
                elif sig.shape[0] < target_len:
                    pad = np.zeros((target_len - sig.shape[0], 12))
                    sig = np.vstack([sig, pad])
                
                # 3. Get Clinical Data (Input 2)
                age = 55
                age_match = age_pattern.search(content)
                if age_match: age = int(age_match.group(1))
                
                sex = 0 # Male default
                sex_match = sex_pattern.search(content)
                if sex_match and 'F' in sex_match.group(1).upper():
                    sex = 1
                
                # Append to lists
                X_signal.append(sig)
                X_clinical.append([age, sex])
                y_true.append(labels)
                
    except Exception as e: continue

# Convert to Arrays
X_signal = np.array(X_signal)
X_clinical = np.array(X_clinical, dtype=np.float32)
X_clinical[:, 0] = X_clinical[:, 0] / 100.0 # Normalize Age
y_true_bin = mlb.transform(y_true)

print(f"✅ Data Ready: {len(X_signal)} samples")

# ==========================================
# 4. PREDICTION & GENERATION
# ==========================================
print("🤖 Running AI Predictions (Dual Input)...")
# FIX: Pass BOTH inputs [Signal, Clinical]
probs = model.predict([X_signal, X_clinical], verbose=0)
y_pred_bin = (probs > 0.2).astype(int)

print("🎨 Generating Charts...")
mcm = multilabel_confusion_matrix(y_true_bin, y_pred_bin)

# Generate Confusion Matrices
for idx, class_name in enumerate(mlb.classes_):
    tn, fp, fn, tp = mcm[idx].ravel()
    
    plt.figure(figsize=(6, 5))
    data = np.array([[tn, fp], [fn, tp]])
    labels = np.array([
        [f"True Neg\n{tn}", f"False Pos\n{fp}"],
        [f"False Neg\n{fn}", f"True Pos\n{tp}"]
    ])
    
    sns.heatmap(data, annot=labels, fmt='', cmap='Blues', cbar=False,
                xticklabels=['Pred NO', 'Pred YES'],
                yticklabels=['Actual NO', 'Actual YES'])
    
    clean_name = class_name.replace(" ", "_")
    plt.title(f"Confusion Matrix: {class_name}")
    plt.tight_layout()
    
    filename = f"Matrix_{clean_name}.png"
    plt.savefig(filename, dpi=100)
    plt.close()
    print(f" -> Saved {filename}")

# Generate Accuracy Bar Chart
acc_scores = []
for idx in range(len(mlb.classes_)):
    tn, fp, fn, tp = mcm[idx].ravel()
    acc = (tp + tn) / (tp + tn + fp + fn) * 100
    acc_scores.append(acc)

plt.figure(figsize=(10, 6))
plt.bar(mlb.classes_, acc_scores, color=['#34495e', '#e74c3c', '#3498db', '#27ae60', '#f1c40f'])
plt.ylabel("Accuracy (%)")
plt.title("Model Accuracy per Disease Condition")
plt.ylim(0, 115)
plt.xticks(rotation=15)
plt.grid(axis='y', linestyle='--', alpha=0.7)

for i, v in enumerate(acc_scores):
    plt.text(i, v + 2, f"{v:.1f}%", ha='center')

plt.tight_layout()
plt.savefig("Performance_Graph.png", dpi=100)
plt.close()
print(" -> Saved Performance_Graph.png")

print("\n✅ DONE! All images are in your folder.")