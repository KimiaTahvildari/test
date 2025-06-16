import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

# ---------------- Load and Prepare Data ----------------
df = pd.read_csv("feature_data_rr_fwave.csv")
df.dropna(inplace=True)

drop_cols = [col for col in ["ecg_id", "label"] if col in df.columns]
X = df.drop(columns=drop_cols).values
y = df["label"].values

le = LabelEncoder()
y_encoded = le.fit_transform(y)
num_classes = 2  # 🔁 Force binary
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Paper-Based FNN ----------------
class PaperFNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(PaperFNN, self).__init__()
        self.hidden = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.output = nn.Linear(128, num_classes)
    def forward(self, x):
        x = self.relu(self.hidden(x))
        return self.output(x)

# ---------------- Training Settings ----------------
EPOCHS = 75
BATCH_SIZE = 32
LR = 0.7
MOMENTUM = 0.0

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
fold = 1
metrics_summary = []

# ---------------- Cross-Validation Loop ----------------
for train_idx, test_idx in skf.split(X, y_encoded):
    print(f"\n🔁 Fold {fold}/10")
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Torch datasets
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                             torch.tensor(y_train, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                            torch.tensor(y_test, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # Initialize model
    model = PaperFNN(X.shape[1], num_classes).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in tqdm(range(EPOCHS), desc=f"Training Fold {fold}", leave=False):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    y_true, y_pred, y_probs = [], [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            out = model(xb.to(device))
            probs = torch.softmax(out, dim=1).cpu().numpy()
            pred = np.argmax(probs, axis=1)
            y_true.extend(yb.numpy())
            y_pred.extend(pred)
            y_probs.extend(probs[:, 1])  # Use prob for positive class

    # Metrics
    auc = roc_auc_score(y_true, y_probs)
    report = classification_report(y_true, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_true, y_pred)

    TP = conf_matrix[1, 1]
    FN = conf_matrix[1, 0]
    FP = conf_matrix[0, 1]
    TN = conf_matrix[0, 0]

    sensitivity = TP / (TP + FN + 1e-8)
    specificity = TN / (TN + FP + 1e-8)
    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-8)

    print(f"✅ Accuracy: {accuracy:.4f} | Sensitivity: {sensitivity:.4f} | Specificity: {specificity:.4f} | AUC: {auc:.4f}")
    metrics_summary.append([accuracy, sensitivity, specificity, auc])
    fold += 1

# ---------------- Summary ----------------
metrics_summary = np.array(metrics_summary)
print("\n📊 Final 10-Fold Summary:")
print(f"Mean Accuracy:     {metrics_summary[:, 0].mean():.4f}")
print(f"Mean Sensitivity:  {metrics_summary[:, 1].mean():.4f}")
print(f"Mean Specificity:  {metrics_summary[:, 2].mean():.4f}")
print(f"Mean AUC:          {metrics_summary[:, 3].mean():.4f}")
