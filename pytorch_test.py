#This is the scrpt for training a CNN-LSTM model on the PTB-XL dataset for AFIB detection.the simple one 
import os
import ast
import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Define base path for dataset
# -------------------------------
BASE_DIR = r"C:\Users\Kimia\thesis related\thesis_coding\test\ptb-xl"
RECORDS_DIR = os.path.join(BASE_DIR, "records100")

# -------------------------------
# Load metadata
# -------------------------------
df = pd.read_csv(os.path.join(BASE_DIR, "ptbxl_database.csv"))

# Identify AFIB via scp_codes in the df itself
df['label'] = df['scp_codes'].apply(
    lambda x: 'AFIB' if 'AFIB' in ast.literal_eval(x) else 'NON-AFIB'
)

# Filter AFIB and NON-AFIB
df = df[df['label'].isin(['AFIB', 'NON-AFIB'])]

# Optional: Balance the classes
min_count = df['label'].value_counts().min()
df = df.groupby('label', group_keys=False).apply(lambda x: x.sample(min_count, random_state=42)).reset_index(drop=True)

# -------------------------------
# Load ECG signal data
# -------------------------------
print("📥 Loading ECG signals:")
signals, labels, missing = [], [], []

for _, row in tqdm(df.iterrows(), total=len(df)):
    filename = row['filename_lr'].replace('.dat', '')
    full_path = os.path.join(BASE_DIR, filename)
    try:
        record = wfdb.rdrecord(full_path)
        signal = record.p_signal
        if signal.shape[0] == 1000:
            signals.append(signal.astype(np.float32))
            labels.append(row['label'])
    except:
        missing.append(full_path)

print(f"✅ Loaded signals: {len(signals)}")
print(f"❌ Missing/unreadable files: {len(missing)}")

signals = np.array(signals)
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# -------------------------------
# Class weights
# -------------------------------
class_weights = compute_class_weight(class_weight='balanced',
                                     classes=np.unique(labels_encoded),
                                     y=labels_encoded)
class_weights = torch.tensor(class_weights, dtype=torch.float32)

# -------------------------------
# Dataset and Dataloader
# -------------------------------
class ECGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(np.array(X)).permute(0, 2, 1)
        self.y = torch.tensor(y).long()
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

X_train, X_test, y_train, y_test = train_test_split(
    signals, labels_encoded, test_size=0.2, stratify=labels_encoded, random_state=42)
train_dataset = ECGDataset(X_train, y_train)
test_dataset = ECGDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

print("Train label distribution: ", np.bincount(y_train), label_encoder.classes_)
print("Test label distribution:  ", np.bincount(y_test), label_encoder.classes_)

# -------------------------------
# CNN-LSTM Model
# -------------------------------
class CNNLSTM(nn.Module):
    def __init__(self):
        super(CNNLSTM, self).__init__()
        self.conv1 = nn.Conv1d(12, 64, kernel_size=7)
        self.pool1 = nn.MaxPool1d(2)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5)
        self.pool2 = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(128, 64, batch_first=True)
        self.fc1 = nn.Linear(64, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, len(label_encoder.classes_))

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.bn1(x)
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.permute(0, 2, 1)
        _, (hn, _) = self.lstm(x)
        x = torch.relu(self.fc1(hn[-1]))
        x = self.dropout(x)
        return self.fc2(x)

# -------------------------------
# Train Model
# -------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNNLSTM().to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    model.train()
    total_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for X_batch, y_batch in loop:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        out = model(X_batch)
        loss = criterion(out, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    print(f"Epoch {epoch+1} average loss: {total_loss / len(train_loader):.4f}")

# -------------------------------
# Evaluate
# -------------------------------
model.eval()
all_preds, all_targets = [], []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(y_batch.numpy())

print(classification_report(all_targets, all_preds, target_names=label_encoder.classes_))
cm = confusion_matrix(all_targets, all_preds)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
