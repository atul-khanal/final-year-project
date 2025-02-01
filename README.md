# DDoS Detection Web Application

A web application for analyzing network traffic data using both supervised and unsupervised machine learning approaches to detect DDoS attacks.

## Features

- File upload interface for CSV data
- Support for both supervised and unsupervised learning models
- Real-time processing and visualization
- Interactive results display with metrics
- Confusion matrix visualization
- Modern UI with HTMX and Tailwind CSS

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python app.py
   ```

4. Open your browser and navigate to `http://localhost:5000`

## Usage

1. Select the model type (Supervised or Unsupervised)
2. Upload your CSV file containing network traffic data
3. Click "Process File" to start the analysis
4. View the results, including:
   - Model performance metrics
   - Confusion matrix visualization

## CSV File Format

The input CSV file should contain the following columns:
- Total Fwd Packets
- Total Backward Packets
- Flow Bytes/s
- Flow Packets/s
- Fwd Packet Length Max
- Bwd Packet Length Max
- Fwd Packet Length Mean
- Bwd Packet Length Mean
- Fwd Packet Length Std
- Bwd Packet Length Std
- Label (for ground truth) 