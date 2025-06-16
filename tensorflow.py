#the tensoeflow version
import pandas as pd
import numpy as np
import ast
import os
import wfdb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization, LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

# ----------------------------- #
#        Load Metadata          #
# ----------------------------- #
df = pd.read_csv('ptbxl_database.csv')
scp_df = pd.read_csv('ptb-xl\scp_statements.csv')
scp_df = scp_df[scp_df['diagnostic_class'].notnull()]

afib_codes = scp_df[scp_df['description'].str.contains('fibrillation', case=False, na=False)].index.tolist()
normal_codes = ['NORM']

def classify_label(scp_codes_str):
    codes = ast.literal_eval(scp_codes_str)
    keys = list(codes.keys())
    if any(code in afib_codes for code in keys):
        return 'AFIB'
    elif any(code in normal_codes for code in keys):
        return 'NORM'
    else:
        return 'OTHER'

df['label'] = df['scp_codes'].apply(classify_label)
df = df[df['label'].isin(['NORM', 'AFIB', 'OTHER'])]

# ----------------------------- #
#       Balance Dataset         #
# ----------------------------- #
min_count = df['label'].value_counts().min()
df = df.groupby('label').apply(lambda x: x.sample(min_count, random_state=42)).reset_index(drop=True)

# ----------------------------- #
#       Load ECG Signals        #
# ----------------------------- #
def load_signal(filepath):
    try:
        record = wfdb.rdrecord(filepath)
        signal = record.p_signal
        if signal.shape[0] != 1000:
            return None
        return signal
    except:
        return None

X, y = [], []
for _, row in df.iterrows():
    path = os.path.join('records100', row['filename_lr']).replace('.dat', '')
    sig = load_signal(path)
    if sig is not None:
        X.append(sig)
        y.append(row['label'])

X = np.array(X)
y = np.array(y)

# ----------------------------- #
#       Encode Labels           #
# ----------------------------- #
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_onehot = to_categorical(y_encoded)

# ----------------------------- #
#       Train-Test Split        #
# ----------------------------- #
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, stratify=y_encoded, random_state=42)

# ----------------------------- #
#         Build Model           #
# ----------------------------- #
model = Sequential([
    Conv1D(64, kernel_size=7, activation='relu', input_shape=(1000, 12)),
    MaxPooling1D(pool_size=2),
    BatchNormalization(),
    Conv1D(128, kernel_size=5, activation='relu'),
    MaxPooling1D(pool_size=2),
    LSTM(64),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ----------------------------- #
#       Train the Model         #
# ----------------------------- #
callbacks = [
    EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True),
    ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True)
]

history = model.fit(X_train, y_train, validation_split=0.2, epochs=20, batch_size=32, callbacks=callbacks)

# ----------------------------- #
#      Evaluate the Model       #
# ----------------------------- #
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

print(classification_report(y_true_labels, y_pred_labels, target_names=label_encoder.classes_))

cm = confusion_matrix(y_true_labels, y_pred_labels)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
