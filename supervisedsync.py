import pandas as pd
import numpy as np
import dask.dataframe as dd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import warnings
warnings.filterwarnings("ignore")  # Suppress warnings for cleaner output

# Load Data with Dask and specify dtype for problematic columns
df = dd.read_csv("SYNc.csv", assume_missing=True, dtype={'SimillarHTTP': 'object'})

# Strip spaces from column names
df.columns = df.columns.str.strip()

# Print available columns after stripping spaces
print("✅ Available Columns:", list(df.columns))

# Define important columns (after cleaning names)
columns_to_check = ['Source Port', 'Destination Port', 'Total Fwd Packets', 'Total Backward Packets', 'Protocol']

# Convert only if the column exists
for col in columns_to_check:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    else:
        print(f"⚠️ Warning: '{col}' column not found. Skipping conversion.")

# Drop rows with NaN values after conversion
df = df.dropna()

# Compute dataset before further processing
df = df.compute()

# Feature Selection (after cleaning column names)
features_to_use = [
    'Total Fwd Packets', 'Total Backward Packets', 'Flow Bytes/s', 'Flow Packets/s',
    'Fwd Packet Length Max', 'Bwd Packet Length Max', 'Fwd Packet Length Mean',
    'Bwd Packet Length Mean', 'Fwd Packet Length Std', 'Bwd Packet Length Std'
]

# Ensure only existing features are selected
features_to_use = [col for col in features_to_use if col in df.columns]
X = df[features_to_use]
y = df['Label'].astype('category').cat.codes  # Convert labels to numeric

# **Fix: Replace Inf values with NaN and then replace NaN with column mean**
X.replace([np.inf, -np.inf], np.nan, inplace=True)  # Convert Inf to NaN
X.fillna(X.mean(), inplace=True)  # Replace NaN with column mean

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # **Fixed Scaling Error**
X_test = scaler.transform(X_test)

# **Supervised Model: Random Forest**
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Model Evaluation
print("\n📊 Model Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-Score:", f1_score(y_test, y_pred))

# Confusion Matrix Visualization
plt.figure(figsize=(5, 5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Save the Model for Future Use
joblib.dump(model, "SYNcddos_rf_model.pkl")
print("\n✅ Model saved as 'SYNcddos_rf_model.pkl'")