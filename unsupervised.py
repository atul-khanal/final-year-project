import pandas as pd
import numpy as np
import dask.dataframe as dd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import joblib
import warnings
import sys
warnings.filterwarnings("ignore")  # Suppress warnings for cleaner output

# Function to safely convert columns to numeric
def safe_convert(col):
    return pd.to_numeric(col, errors='coerce')

def process_data(csv_path):
    # Load Data with Dask and specify dtype for problematic columns
    df = dd.read_csv(csv_path, assume_missing=True, dtype={'SimillarHTTP': 'object'})
    
    # Strip spaces from column names
    df.columns = df.columns.str.strip()
    
    # Print available columns after stripping spaces
    print("✅ Available Columns:", list(df.columns))
    
    # Define important columns (after cleaning names)
    columns_to_check = ['Source Port', 'Destination Port', 'Total Fwd Packets', 'Total Backward Packets', 'Protocol']
    
    # Convert only if the column exists
    for col in columns_to_check:
        if col in df.columns:
            df[col] = safe_convert(df[col])
        else:
            print(f"⚠️ Warning: '{col}' column not found. Skipping conversion.")
    
    # Drop rows with NaN values after conversion
    df = df.dropna()
    
    # Compute dataset before further processing
    df = df.compute()
    
    # Enhanced Feature Selection
    features_to_use = [
        'Total Fwd Packets', 'Total Backward Packets', 'Flow Bytes/s', 'Flow Packets/s',
        'Fwd Packet Length Max', 'Bwd Packet Length Max', 'Fwd Packet Length Mean',
        'Bwd Packet Length Mean', 'Fwd Packet Length Std', 'Bwd Packet Length Std',
        'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
        'Fwd IAT Mean', 'Fwd IAT Std', 'Bwd IAT Mean', 'Bwd IAT Std',
        'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags',
        'Active Mean', 'Active Std', 'Idle Mean', 'Idle Std'
    ]
    
    # Ensure only existing features are selected
    features_to_use = [col for col in features_to_use if col in df.columns]
    X = df[features_to_use]
    y = df['Label'].astype('category').cat.codes  # Convert labels to numeric
    
    # Advanced Data Preprocessing
    # Replace Inf values with NaN
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Calculate more robust statistics for NaN replacement
    column_medians = X.median()
    column_iqr = X.quantile(0.75) - X.quantile(0.25)
    lower_bound = X.quantile(0.25) - (1.5 * column_iqr)
    upper_bound = X.quantile(0.75) + (1.5 * column_iqr)
    
    # Replace outliers and NaN values
    for column in X.columns:
        mask = (X[column] < lower_bound[column]) | (X[column] > upper_bound[column])
        X.loc[mask, column] = column_medians[column]
    X.fillna(column_medians, inplace=True)
    
    # Feature Selection using Mutual Information
    k = min(15, len(features_to_use))  # Select top 15 features or all if less
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()].tolist()
    X = X[selected_features]
    
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Use RobustScaler for better handling of outliers
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Calculate contamination based on actual ratio of anomalies in the dataset
    contamination = np.mean(y == 1)  # Assuming 1 represents anomalies
    contamination = max(min(contamination, 0.5), 0.01)  # Keep between 1% and 50%
    
    # Enhanced Isolation Forest with tuned parameters
    model = IsolationForest(
        n_estimators=200,  # Increased number of trees
        max_samples='auto',
        contamination=contamination,
        max_features=1.0,
        bootstrap=True,
        n_jobs=-1,  # Use all CPU cores
        random_state=42,
        verbose=0
    )
    
    model.fit(X_train)
    
    # Predict Anomalies
    y_pred = model.predict(X_test)
    y_pred = np.where(y_pred == -1, 1, 0)
    
    # Model Evaluation
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    }
    
    # Print Feature Importance Information
    print("\n🔍 Selected Features for Detection:")
    for feature in selected_features:
        print(f"- {feature}")
    
    # Enhanced Confusion Matrix Visualization
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Enhanced Unsupervised Learning")
    
    # Save the plot
    plt.savefig('static/unsupervised_confusion_matrix.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Save the Model and Scaler
    joblib.dump(model, "static/ddos_detection_model.pkl")
    joblib.dump(scaler, "static/ddos_detection_scaler.pkl")
    
    return metrics

if __name__ == "__main__":
    try:
        if len(sys.argv) != 2:
            print("❌ Error: Please provide the path to the CSV file")
            print("Usage: python unsupervised.py <path_to_csv>")
            sys.exit(1)
            
        csv_path = sys.argv[1]
        print(f"🔄 Processing data from: {csv_path}")
        
        metrics = process_data(csv_path)
        
        print("\n📊 Model Performance Metrics:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        
        print("\n✅ Model and visualizations have been saved successfully!")
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find the CSV file at {csv_path}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: An unexpected error occurred: {str(e)}")
        sys.exit(1)
