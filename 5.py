import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
# import train_test_split  # Uncomment if you want to use train_test_split instead of StratifiedKFold
from sklearn.model_selection import train_test_split

# Load and preprocess the data
df = pd.read_csv("feature_data_rr_fwave.csv")
df.dropna(inplace=True)
print(f"🔍 Data shape: {df.shape}")
# 10 percent for testing
test_size = 0.1

df, test_df = train_test_split(df, test_size=test_size, stratify=df["label"], random_state=42)

# delete all pth files in the current directory
import os
for file in os.listdir("."):
    if file.endswith(".pth"):
        os.remove(file)
        print(f"🗑️ Deleted {file}")

drop_cols = [col for col in ["ecg_id", "label"] if col in df.columns]
X = df.drop(columns=drop_cols).values
y = df["label"].values

le = LabelEncoder()
y_encoded = le.fit_transform(y)
num_classes = 2  # 🔁 Force binary
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def visualize_history(history):
    import matplotlib.pyplot as plt

    metrics = ["accuracy", "sensitivity", "specificity", "auc", "f1_score"]
    # 3x2 grid that shows all metrics for all folds and
    # each metric is a separate plot

    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    axes = axes.flatten()
    for i, metric in enumerate(metrics):
        ax = axes[i]
        for fold, values in enumerate(history[metric]):
            ax.plot(values, label=f"Fold {fold+1}")
        ax.set_title(metric.capitalize())
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.capitalize())
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    plt.savefig("training_history.png")


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
fold = 0
history = {
    # history is list of lists for each fold and each epoch
    "accuracy": [],
    "sensitivity": [],
    "specificity": [],
    "auc": [],
    "f1_score": []
}

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


    # optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


    class_counts = np.bincount(y_train)
    loss_weight = torch.tensor(class_counts.max() / class_counts, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=loss_weight)
    # criterion = nn.CrossEntropyLoss()  # Uncomment if you don't want to use class weights

    best_f1_score = 0.0

    accuracy_summary = []
    auc_summary = []
    f1_summary = []
    sensitivity_summary = []
    specificity_summary = []





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
        print(conf_matrix)
        f1_score = conf_matrix[1, 1] / (conf_matrix[1, 1] + 0.5 * (conf_matrix[0, 1] + conf_matrix[1, 0]) + 1e-8)

        TP = conf_matrix[1, 1]
        FN = conf_matrix[1, 0]
        FP = conf_matrix[0, 1]
        TN = conf_matrix[0, 0]

        sensitivity = TP / (TP + FN + 1e-8)
        specificity = TN / (TN + FP + 1e-8)
        accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-8)

        if f1_score > best_f1_score:
            best_f1_score = f1_score
            print(f"🏆 Best F1 Score for Fold {fold}: {best_f1_score:.4f}")

            # save the model
            torch.save(model.state_dict(), f"best_model_fold_{fold}.pth")

        print(f"✅ Fold {fold} - Epoch {epoch+1}/{EPOCHS} | "
               f"Accuracy: {accuracy:.4f} | Sensitivity: {sensitivity:.4f} | Specificity: {specificity:.4f} | AUC: {auc:.4f}")

        accuracy_summary.append(accuracy)
        auc_summary.append(auc)
        f1_summary.append(f1_score)
        sensitivity_summary.append(sensitivity)
        specificity_summary.append(specificity)

    # Store metrics for this fold
    history["accuracy"].append(accuracy_summary)
    history["sensitivity"].append(sensitivity_summary)
    history["specificity"].append(specificity_summary)
    history["auc"].append(auc_summary)
    history["f1_score"].append(f1_summary)

    visualize_history(history)
    fold += 1


# ---------------- Summary ----------------
# print mean score for each metric across all folds
print("\n📊 Summary of Metrics Across Folds:")

for metric in history:
    mean_values = np.mean(history[metric], axis=0)
    print(f"{metric.capitalize()}: {np.mean(mean_values):.4f} ± {np.std(mean_values):.4f}")
print("✅ Training complete. Check the saved models and training history.")


#  load the best model for each fold and evaluate on the test set by averaging the predictions

best_models = []
for fold in range(10):
    model = PaperFNN(X.shape[1], num_classes).to(device)
    model.load_state_dict(torch.load(f"best_model_fold_{fold}.pth"))
    model.eval()
    best_models.append(model)

# Evaluate on the test set
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
y_true, y_pred, y_probs = [], [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        preds = [model(xb).cpu().numpy() for model in best_models]
        avg_preds = np.mean(preds, axis=0)
        pred = np.argmax(avg_preds, axis=1)
        probs = np.max(avg_preds, axis=1)  # Probability of the predicted class
        y_true.extend(yb.numpy())
        y_pred.extend(pred)
        y_probs.extend(probs)

# Final evaluation
auc = roc_auc_score(y_true, y_probs)
report = classification_report(y_true, y_pred, output_dict=True)
conf_matrix = confusion_matrix(y_true, y_pred)
print("\nFinal Confusion Matrix:")
print(conf_matrix)
f1_score = conf_matrix[1, 1] / (conf_matrix[1, 1] + 0.5 * (conf_matrix[0, 1] + conf_matrix[1, 0]) + 1e-8)
TP = conf_matrix[1, 1]
FN = conf_matrix[1, 0]
FP = conf_matrix[0, 1]
TN = conf_matrix[0, 0]
sensitivity = TP / (TP + FN + 1e-8)
specificity = TN / (TN + FP + 1e-8)
accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-8)
print(f"\nFinal Evaluation on Test Set:")
print(f"Accuracy: {accuracy:.4f} | Sensitivity: {sensitivity:.4f} | Specificity: {specificity:.4f} | AUC: {auc:.4f} | F1 Score: {f1_score:.4f}")
