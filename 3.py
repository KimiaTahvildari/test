import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- Load Feature Data ----------------
df = pd.read_csv("feature_data_rr_fwave.csv")
df.dropna(inplace=True)

drop_cols = [col for col in ["ecg_id", "label"] if col in df.columns]
X = df.drop(columns=drop_cols)
y = df["label"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

results = {}
accuracies = {}

# ---------------- Random Forest ----------------
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_std, y_train)
y_pred_rf = rf.predict(X_test_std)
results["Random Forest"] = classification_report(y_test, y_pred_rf, output_dict=True)
accuracies["Random Forest"] = accuracy_score(y_test, y_pred_rf)

# ---------------- SVM ----------------
svm = SVC(kernel='rbf', probability=True)
svm.fit(X_train_std, y_train)
y_pred_svm = svm.predict(X_test_std)
results["SVM"] = classification_report(y_test, y_pred_svm, output_dict=True)
accuracies["SVM"] = accuracy_score(y_test, y_pred_svm)

# ---------------- Feedforward NN ----------------
fnn = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
fnn.fit(X_train_std, y_train)
y_pred_fnn = fnn.predict(X_test_std)
results["FNN"] = classification_report(y_test, y_pred_fnn, output_dict=True)
accuracies["FNN"] = accuracy_score(y_test, y_pred_fnn)

# ---------------- Dataset Class ----------------
class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

train_dataset = FeatureDataset(X_train_std, y_train)
test_dataset = FeatureDataset(X_test_std, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# ---------------- CNN ----------------
class SimpleCNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SimpleCNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

device = torch.device("cpu")
model = SimpleCNN(X_train_std.shape[1], len(le.classes_)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(5):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        out = model(X_batch)
        loss = criterion(out, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"[CNN] Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

# Evaluate CNN
model.eval()
cnn_preds, cnn_targets = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        out = model(X_batch)
        preds = torch.argmax(out, dim=1)
        cnn_preds.extend(preds.cpu().tolist())
        cnn_targets.extend(y_batch.tolist())

results["CNN"] = classification_report(cnn_targets, cnn_preds, output_dict=True)
accuracies["CNN"] = accuracy_score(cnn_targets, cnn_preds)

# ---------------- CNN-LSTM ----------------
class CNNLSTMTabular(nn.Module):
    def __init__(self, input_dim):
        super(CNNLSTMTabular, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.lstm = nn.LSTM(input_size=32, hidden_size=64, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(64, 32)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(32, len(le.classes_))

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        _, (hn, _) = self.lstm(x)
        x = self.relu(self.fc1(hn[-1]))
        x = self.dropout(x)
        return self.fc2(x)

cnn_lstm_model = CNNLSTMTabular(X_train_std.shape[1]).to(device)
optimizer = torch.optim.Adam(cnn_lstm_model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(5):
    cnn_lstm_model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        out = cnn_lstm_model(X_batch)
        loss = criterion(out, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"[CNN-LSTM] Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

# Evaluate CNN-LSTM
cnn_lstm_model.eval()
lstm_preds, lstm_targets = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        out = cnn_lstm_model(X_batch)
        preds = torch.argmax(out, dim=1)
        lstm_preds.extend(preds.cpu().tolist())
        lstm_targets.extend(y_batch.tolist())

results["CNN-LSTM"] = classification_report(lstm_targets, lstm_preds, output_dict=True)
accuracies["CNN-LSTM"] = accuracy_score(lstm_targets, lstm_preds)


# ---------------- Summary with Bold Max (Tabular Print) ----------------
summary_df = pd.DataFrame({k: v["weighted avg"] for k, v in results.items()}).T
summary_df["accuracy"] = [accuracies[k] for k in summary_df.index]
summary_df = summary_df[["precision", "recall", "f1-score", "accuracy"]]

# ANSI codes for bold
def bold(text):
    return f"\033[1m{text}\033[0m"

# Identify max values for each column
max_vals = summary_df.max()

# Print table header
header = f"{'Model':<20}" + "".join([f"{col:^15}" for col in summary_df.columns])
print("\n📊 Model Comparison Summary:\n")
print(header)
print("-" * len(header))

# Print each row with bold for max values
for model, row in summary_df.iterrows():
    row_str = f"{model:<20}"
    for col in summary_df.columns:
        val = f"{row[col]:.4f}"
        if row[col] == max_vals[col]:
            val = bold(val)
        row_str += f"{val:^15}"
    print(row_str)

print()


