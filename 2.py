import os
import ast
import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm
import neurokit2 as nk

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = r"C:\Users\Kimia\thesis related\thesis_coding\test\ptb-xl"
RECORDS_DIR = os.path.join(BASE_DIR, "records100")
PTB_CSV = os.path.join(BASE_DIR, "ptbxl_database.csv")

# -----------------------------
# Load PTB-XL metadata
# -----------------------------
df = pd.read_csv(PTB_CSV)

def label_afib(scp_codes_str):
    scp = ast.literal_eval(scp_codes_str)
    return 'AFIB' if 'AFIB' in scp else 'NON-AFIB'

df['label'] = df['scp_codes'].apply(label_afib)
df = df[df['label'].isin(['AFIB', 'NON-AFIB'])]

# -----------------------------
# Feature extraction
# -----------------------------
features = []

print("🔍 Extracting features...")
for _, row in tqdm(df.iterrows(), total=len(df)):
    path = os.path.join(BASE_DIR, row['filename_lr'].replace('.dat', ''))
    try:
        record = wfdb.rdrecord(path)
        signal = record.p_signal
        if signal.shape[0] != 1000:
            continue

        # Use lead II for rhythm analysis
        ecg = signal[:, 1]
        cleaned = nk.ecg_clean(ecg, sampling_rate=100)
        peaks, _ = nk.ecg_peaks(cleaned, sampling_rate=100)
        rate = nk.ecg_rate(peaks, sampling_rate=100)

        rr_intervals = np.diff(np.where(peaks["ECG_R_Peaks"])[0]) / 100.0

        if len(rr_intervals) < 2:
            continue

        # Features
        feat = {
            "rr_mean": np.mean(rr_intervals),
            "rr_std": np.std(rr_intervals),
            "rr_min": np.min(rr_intervals),
            "rr_max": np.max(rr_intervals),
            "rr_range": np.max(rr_intervals) - np.min(rr_intervals),
            "label": row["label"]
        }

        features.append(feat)

    except Exception as e:
        continue

# -----------------------------
# Save to CSV
# -----------------------------
features_df = pd.DataFrame(features)
features_df.to_csv("feature_data_rr_fwave.csv", index=False)
print(f"✅ Saved {len(features_df)} records to feature_data_rr_fwave.csv")
