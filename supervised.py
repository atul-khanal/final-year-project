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
import subprocess
from datetime import datetime
warnings.filterwarnings("ignore")  # Suppress warnings for cleaner output

def block_ip(ip_address):
    """Block an IP address using iptables"""
    try:
        # Check if the IP is already blocked
        check_cmd = f"sudo iptables -L INPUT -v -n | grep {ip_address}"
        result = subprocess.run(check_cmd, shell=True, capture_output=True, text=True)
        
        if ip_address not in result.stdout:
            # Block the IP using iptables
            block_cmd = f"sudo iptables -A INPUT -s {ip_address} -j DROP"
            subprocess.run(block_cmd, shell=True, check=True)
            
            # Log the blocked IP
            with open('blocked_ips.log', 'a') as f:
                f.write(f"{datetime.now()} - Blocked IP: {ip_address}\n")
            
            print(f"🚫 Blocked malicious IP: {ip_address}")
        else:
            print(f"ℹ️ IP {ip_address} is already blocked")
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Error blocking IP {ip_address}: {str(e)}")

def process_data(csv_path):
    # Load Data with Dask and specify dtype for problematic columns
    df = dd.read_csv(csv_path, assume_missing=True, dtype={'SimillarHTTP': 'object'})
    
    # Strip spaces from column names
    df.columns = df.columns.str.strip()
    
    # Print available columns after stripping spaces
    print("✅ Available Columns:", list(df.columns))
    
    # Define important columns (after cleaning names)
    columns_to_check = ['Source Port', 'Destination Port', 'Total Fwd Packets', 'Total Backward Packets', 'Protocol', 'Source IP', 'Destination IP']
    
    # Convert only if the column exists
    for col in columns_to_check:
        if col in df.columns:
            if col not in ['Source IP', 'Destination IP']:  # Skip IP address columns for numeric conversion
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
    
    # Store IP addresses for blocking if attack is detected
    ip_addresses = None
    if 'Source IP' in df.columns:
        ip_addresses = df['Source IP'].copy()
    
    # Ensure only existing features are selected
    features_to_use = [col for col in features_to_use if col in df.columns]
    X = df[features_to_use]
    y = df['Label'].astype('category').cat.codes  # Convert labels to numeric
    
    # Fix: Replace Inf values with NaN and then replace NaN with column mean
    X.replace([np.inf, -np.inf], np.nan, inplace=True)  # Convert Inf to NaN
    X.fillna(X.mean(), inplace=True)  # Replace NaN with column mean
    
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # If we have IP addresses, split them accordingly
    if ip_addresses is not None:
        _, ip_test = train_test_split(ip_addresses, test_size=0.2, random_state=42, stratify=y)
    
    # Feature Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Save the scaler for future use
    scaler_path = "static/ddos_detection_scaler.pkl"
    joblib.dump(scaler, scaler_path)
    
    # Supervised Model: Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Block IPs for detected attacks
    if ip_addresses is not None:
        attack_indices = np.where(y_pred == 1)[0]
        malicious_ips = ip_test.iloc[attack_indices].unique()
        
        print("\n🔍 Analyzing detected attacks...")
        for ip in malicious_ips:
            block_ip(ip)
    
    # Model Evaluation
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    }
    
    # Confusion Matrix Visualization
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", 
                xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Supervised Learning")
    
    # Save the plot
    plt.savefig('static/supervised_confusion_matrix.png')
    plt.close()
    
    # Save the Model for Future Use
    model_path = "static/ddos_rf_model.pkl"
    joblib.dump(model, model_path)
    
    return metrics
