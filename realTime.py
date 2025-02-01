# import joblib
# import numpy as np
# import pandas as pd
# from scapy.all import sniff, IP, TCP, UDP
# from sklearn.preprocessing import StandardScaler

# # Load the trained model and scaler
# model = joblib.load("final_ddos_model.pkl")
# scaler = joblib.load("scaler.pkl")  # Make sure you saved the scaler during training

# # Define the features to extract from network packets
# def extract_features(packet):
#     try:
#         if IP in packet:
#             src_ip = packet[IP].src
#             dst_ip = packet[IP].dst
#             proto = packet[IP].proto
#             pkt_size = len(packet)
#             flow_packets = 1  # Each packet is a new flow
#             flow_bytes = pkt_size
            
#             # TCP-specific features
#             if TCP in packet:
#                 flags = packet[TCP].flags
#                 flow_packets += 1  # Increment packet count for flow
#                 flow_bytes += len(packet)
#             else:
#                 flags = 0

#             return [flow_packets, flow_bytes, proto, pkt_size, flags]
#         else:
#             return None  # Skip non-IP packets
#     except Exception as e:
#         print(f"Error extracting features: {e}")
#         return None

# # Function to process and classify real-time packets
# def detect_ddos(packet):
#     features = extract_features(packet)
    
#     if features:
#         # Convert to NumPy array and reshape for model input
#         input_data = np.array(features).reshape(1, -1)
        
#         # Apply the same scaling as during training
#         input_data = scaler.transform(input_data)

#         # Predict using the trained model
#         prediction = model.predict(input_data)

#         # Interpret the result
#         if prediction[0] == 1:
#             print(f"🚨 DDoS Attack Detected! Packet Info: {packet.summary()}")
#         else:
#             print(f"✅ Normal Traffic: {packet.summary()}")

# # Start real-time packet capture (sniff on interface 'eth0' or 'wlan0' for Wi-Fi)
# print("🔍 Monitoring network traffic for potential DDoS attacks...")
# sniff(filter="ip", prn=detect_ddos, store=0, iface="eth0")  # Change iface if needed


import joblib
import numpy as np
import pandas as pd
from scapy.all import sniff, IP, TCP, UDP, ICMP
from sklearn.preprocessing import StandardScaler

# 1. Load Model and Scaler
try:
    model = joblib.load("final_ddos_model.pkl")
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError:
    print("Error: Model or scaler file not found. Make sure they are in the correct directory.")
    exit()  # Exit if files are not found
except Exception as e:
    print(f"Error loading model/scaler: {e}")
    exit()

# 2. Feature Extraction Function
def extract_features(packet):
    try:
        features = []

        if IP in packet:
            ip_layer = packet[IP]
            features.extend([1, len(packet), ip_layer.proto])  # Packet count (initialized to 1), size, protocol

            if TCP in packet:
                tcp_layer = packet[TCP]
                features.extend([tcp_layer.flags, tcp_layer.sport, tcp_layer.dport])  # TCP flags, source/dest ports
            elif UDP in packet:
                udp_layer = packet[UDP]
                features.extend([0, udp_layer.sport, udp_layer.dport])  # 0 for TCP flags, UDP ports
            elif ICMP in packet:  # Handle ICMP packets
                features.extend([0, 0, 0])  # No ports or TCP flags for ICMP
            else:
                features.extend([0, 0, 0])  # Default values if no TCP/UDP/ICMP

            # Add more features (IP, TCP, UDP, ICMP specific) as needed.
            # Example: IP TTL, TCP window size, etc.
            features.extend([ip_layer.ttl]) #IP ttl

            return features
        else:
            return None  # Skip non-IP packets

    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

# 3. DDoS Detection Function
def detect_ddos(packet):
    features = extract_features(packet)

    if features:
        try:
            input_data = np.array(features).reshape(1, -1)
            input_data = scaler.transform(input_data)  # Scale the input data

            prediction = model.predict(input_data)

            if prediction[0] == 1:  # Assuming 1 is the attack class
                print(f"🚨 DDoS Attack Detected! Packet Info: {packet.summary()}")
            else:
                print(f"✅ Normal Traffic: {packet.summary()}") #Comment out normal traffic messages for less verbose output

        except ValueError as ve:
            print(f"Value Error during prediction: {ve}. Features: {features}") #Print feature values causing the error
        except Exception as e:
            print(f"Error during prediction: {e}")


# 4. Start Packet Capture
print("🔍 Monitoring network traffic for potential DDoS attacks...")

try:
    sniff(filter="ip", prn=detect_ddos, store=0, iface="eth0")  # Change interface if needed
except Exception as e:
    print(f"Error during sniffing: {e}")