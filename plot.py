import os
import ast
import wfdb
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# Base Paths
# -------------------------------
BASE_DIR = r"C:\Users\Kimia\thesis related\thesis_coding\test\ptb-xl"
RECORDS_DIR = os.path.join(BASE_DIR, "records100")

# -------------------------------
# Load PTB-XL Metadata
# -------------------------------
df = pd.read_csv(os.path.join(BASE_DIR, "ptbxl_database.csv"))

# -------------------------------
# Filter for AFIB Records
# -------------------------------
def has_afib(code_dict_str):
    try:
        codes = ast.literal_eval(code_dict_str)
        return 'AFIB' in codes
    except:
        return False

afib_df = df[df['scp_codes'].apply(has_afib)].copy()
print(f"✅ Found {len(afib_df)} AFIB records")
print(afib_df[['ecg_id', 'patient_id', 'filename_lr']].head())

# -------------------------------
# Plot First AFIB Signal
# -------------------------------
if len(afib_df) > 0:
    first_row = afib_df.iloc[0]
    filename = first_row['filename_lr'].replace('.dat', '')
    full_path = os.path.join(BASE_DIR, filename)
    
    print(f"\n📁 Loading: {full_path}")
    
    try:
        record = wfdb.rdrecord(full_path)
        signal = record.p_signal
        leads = record.sig_name
        sampling_rate = record.fs

        print(f"🧠 Sampling rate: {sampling_rate} Hz")
        print(f"📊 Signal shape: {signal.shape} (samples x leads)")

        # Plot all 12 leads
        plt.figure(figsize=(15, 10))
        for i in range(signal.shape[1]):
            plt.subplot(6, 2, i + 1)
            plt.plot(signal[:, i])
            plt.title(f"Lead {leads[i]}")
            plt.tight_layout()
        plt.suptitle("AFIB ECG Sample (All 12 Leads)", fontsize=16, y=1.02)
        plt.show()
    except Exception as e:
        print(f"❌ Failed to read signal: {e}")
else:
    print("❌ No AFIB samples found.")
