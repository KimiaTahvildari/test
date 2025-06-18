import pandas as pd
import numpy as np
import itertools
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Load and preprocess the data
df = pd.read_csv("feature_data_rr_fwave.csv")
df.dropna(inplace=True)
drop_cols = [col for col in ["ecg_id", "label"] if col in df.columns]
X = df.drop(columns=drop_cols).values
y = df["label"].values
le = LabelEncoder()
y_encoded = le.fit_transform(y)
num_classes = len(np.unique(y_encoded))

# FNN model definition
class PaperFNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(PaperFNN, self).__init__()
        self.hidden = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.output = nn.Linear(128, num_classes)
    def forward(self, x):
        x = self.relu(self.hidden(x))
        return self.output(x)

# Grid Search over learning rate and momentum
learning_rates = [0.1, 0.3, 0.5, 0.7]
momentums = [0.0, 0.5, 0.9]
EPOCHS = 10  # Keep low for grid search
BATCH_SIZE = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

results = []
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
grid = list(itertools.product(learning_rates, momentums))

for lr, momentum in tqdm(grid, desc="Grid Search"):
    fold_aucs = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

        scaler = StandardScaler()
        X_train_std = scaler.fit_transform(X_train)
        X_val_std = scaler.transform(X_val)

        train_ds = TensorDataset(torch.tensor(X_train_std, dtype=torch.float32),
                                 torch.tensor(y_train, dtype=torch.long))
        val_ds = TensorDataset(torch.tensor(X_val_std, dtype=torch.float32),
                               torch.tensor(y_val, dtype=torch.long))
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

        model = PaperFNN(input_dim=X.shape[1], num_classes=num_classes).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(EPOCHS):
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

        model.eval()
        y_true, y_probs = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch.to(device))
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                y_true.extend(y_batch.numpy())
                y_probs.extend(probs)

        y_true_bin = np.eye(num_classes)[y_true]
        auc = roc_auc_score(y_true_bin, y_probs, average='macro', multi_class='ovr')
        fold_aucs.append(auc)

    avg_auc = np.mean(fold_aucs)
    results.append({"lr": lr, "momentum": momentum, "avg_auc": avg_auc})

# Show best parameters
df_results = pd.DataFrame(results)
print("\n🔍 Grid Search Results:\n", df_results)
print("\n✅ Best Params:", df_results.sort_values(by="avg_auc", ascending=False).iloc[0])
