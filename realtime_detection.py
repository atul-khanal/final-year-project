import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

class RealtimeDDoSDetector:
    def __init__(self):
        # Load both models
        try:
            self.supervised_model = joblib.load("static/ddos_rf_model.pkl")
            self.unsupervised_model = joblib.load("static/ddos_detection_model.pkl")
            self.scaler = joblib.load("static/ddos_detection_scaler.pkl")
            print("✅ Models loaded successfully")
        except Exception as e:
            print(f"❌ Error loading models: {str(e)}")
            raise

        # Define features needed for detection
        self.required_features = [
            'Total Fwd Packets', 'Total Backward Packets', 'Flow Bytes/s', 'Flow Packets/s',
            'Fwd Packet Length Max', 'Bwd Packet Length Max', 'Fwd Packet Length Mean',
            'Bwd Packet Length Mean', 'Fwd Packet Length Std', 'Bwd Packet Length Std'
        ]

    def preprocess_data(self, data):
        """Preprocess a single data point or batch for prediction"""
        try:
            # Convert data to DataFrame if it's a dictionary
            if isinstance(data, dict):
                data = pd.DataFrame([data])
            
            # Ensure all required features are present
            missing_features = set(self.required_features) - set(data.columns)
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")

            # Select only required features
            X = data[self.required_features]

            # Handle infinite values
            X.replace([np.inf, -np.inf], np.nan, inplace=True)
            
            # Fill NaN values with mean of the feature
            X.fillna(X.mean(), inplace=True)

            # Scale the features
            X_scaled = self.scaler.transform(X)
            
            return X_scaled

        except Exception as e:
            print(f"❌ Error in preprocessing: {str(e)}")
            raise

    def detect(self, data, threshold=0.7):
        """
        Perform real-time DDoS detection using both models
        Returns: dict with detection results and confidence scores
        """
        try:
            # Preprocess the data
            X_scaled = self.preprocess_data(data)

            # Get predictions from both models
            supervised_pred = self.supervised_model.predict_proba(X_scaled)
            unsupervised_pred = self.unsupervised_model.predict(X_scaled)

            # Calculate confidence scores
            supervised_confidence = supervised_pred[:, 1]  # Probability of DDoS
            unsupervised_confidence = np.where(unsupervised_pred == -1, 1, 0)  # Convert to binary (1 for anomaly)

            # Combine predictions (ensemble approach)
            combined_confidence = (supervised_confidence + unsupervised_confidence) / 2

            # Make final decision
            is_ddos = combined_confidence.mean() > threshold

            return {
                'is_ddos': bool(is_ddos),
                'confidence': float(combined_confidence.mean()),
                'supervised_confidence': float(supervised_confidence.mean()),
                'unsupervised_confidence': float(unsupervised_confidence.mean()),
                'alert_level': 'high' if combined_confidence.mean() > 0.8 else 'medium' if combined_confidence.mean() > 0.6 else 'low'
            }

        except Exception as e:
            print(f"❌ Error in detection: {str(e)}")
            raise

    def start_monitoring(self, data_stream, callback=None):
        """
        Monitor a stream of network data in real-time
        data_stream: iterator of network data points
        callback: function to call with detection results
        """
        try:
            for data_point in data_stream:
                # Perform detection
                result = self.detect(data_point)
                
                # If callback is provided, send results
                if callback and result['is_ddos']:
                    callback(result)
                
                yield result

        except Exception as e:
            print(f"❌ Error in monitoring: {str(e)}")
            raise

# Example usage:
if __name__ == "__main__":
    # Initialize detector
    detector = RealtimeDDoSDetector()
    
    # Example data point (replace with real network data)
    example_data = {
        'Total Fwd Packets': 100,
        'Total Backward Packets': 80,
        'Flow Bytes/s': 1000,
        'Flow Packets/s': 50,
        'Fwd Packet Length Max': 1500,
        'Bwd Packet Length Max': 1500,
        'Fwd Packet Length Mean': 800,
        'Bwd Packet Length Mean': 700,
        'Fwd Packet Length Std': 200,
        'Bwd Packet Length Std': 180
    }
    
    # Test detection
    result = detector.detect(example_data)
    print("Detection Result:", result)