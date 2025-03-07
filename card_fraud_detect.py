import hashlib
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from torch_geometric.nn import GCNConv
import tensorflow as tf
import geopy.distance
import speech_recognition as sr
import warnings

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')


# Load and preprocess dataset
def load_and_preprocess_data(data):
    data['Transaction_Date'] = pd.to_datetime(data['Transaction_Date'])
    data['Hour'] = data['Transaction_Date'].dt.hour
    data['Day'] = data['Transaction_Date'].dt.day
    data['Month'] = data['Transaction_Date'].dt.month
    
    categorical_cols = ['Gender', 'Age_Group', 'Transaction_City', 'Merchant_Type', 'Card_Type', 'Device_Os']
    label_encoders = {col: LabelEncoder().fit(data[col]) for col in categorical_cols}
    
    for col, le in label_encoders.items():
        data[col] = le.transform(data[col])

    scaler = StandardScaler()
    numerical_cols = ['Amount', 'Hour', 'Day', 'Month']
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
    
    return data, label_encoders, scaler

# Graph Neural Network (GNN)
class GNNFraudDetector(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNFraudDetector, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return torch.sigmoid(x)

# Reinforcement Learning Agent
class RLAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = self.build_model()
    
    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_dim=self.state_dim),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.action_dim, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        return model
    
    def train(self, states, actions, rewards):
        self.model.fit(states, actions, sample_weight=rewards, epochs=10, verbose=1)

# Train models
def train_models(data):
    X = data.drop(columns=['Fraud'])
    y = data['Fraud']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create edge index
    edge_index = torch.tensor([[i, j] for i in range(len(X_train)) for j in range(len(X_train)) if i != j], dtype=torch.long).t()
    x = torch.tensor(X_train.values, dtype=torch.float)
    y_gnn = torch.tensor(y_train.values, dtype=torch.float).view(-1, 1)
    
    # Train GNN Model
    gnn_model = GNNFraudDetector(input_dim=X_train.shape[1], hidden_dim=64, output_dim=1)
    optimizer = optim.Adam(gnn_model.parameters(), lr=0.01)
    criterion = nn.BCELoss()
    
    for epoch in range(10):
        optimizer.zero_grad()
        out = gnn_model(x, edge_index)
        loss = criterion(out, y_gnn)
        loss.backward()
        optimizer.step()
    
    # Train Reinforcement Learning Model
    rl_agent = RLAgent(state_dim=X_train.shape[1], action_dim=2)
    rl_agent.train(X_train.values, y_train.values, np.ones(len(y_train)))
    
    return gnn_model, rl_agent, X_test, y_test

# Blockchain Implementation
class Block:
    def __init__(self, index, previous_hash, timestamp, data, hash):
        self.index = index
        self.previous_hash = previous_hash
        self.timestamp = timestamp
        self.data = data
        self.hash = hash

def calculate_hash(index, previous_hash, timestamp, data):
    value = str(index) + str(previous_hash) + str(timestamp) + str(data)
    return hashlib.sha256(value.encode('utf-8')).hexdigest()

def create_genesis_block():
    return Block(0, "0", datetime.now(), "Genesis Block", calculate_hash(0, "0", datetime.now(), "Genesis Block"))

def create_new_block(previous_block, data):
    index = previous_block.index + 1
    timestamp = datetime.now()
    hash = calculate_hash(index, previous_block.hash, timestamp, data)
    return Block(index, previous_block.hash, timestamp, data, hash)

blockchain = [create_genesis_block()]
previous_block = blockchain[0]

# AI Fraud Detection Model
class GNNFraudDetector(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNFraudDetector, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return torch.sigmoid(x)

def train_models(data):
    """Train the AI model for fraud detection."""
    X = data.drop(columns=['Is_Fraudulent'])
    y = data['Is_Fraudulent']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    edge_index = torch.tensor([[i, j] for i in range(len(X_train)) for j in range(len(X_train)) if i != j], dtype=torch.long).t()
    x = torch.tensor(X_train.values, dtype=torch.float)
    y_gnn = torch.tensor(y_train.values, dtype=torch.float).view(-1, 1)
    
    gnn_model = GNNFraudDetector(input_dim=X_train.shape[1], hidden_dim=64, output_dim=1)
    optimizer = optim.Adam(gnn_model.parameters(), lr=0.01)
    criterion = nn.BCELoss()
    
    for epoch in range(10):
        optimizer.zero_grad()
        out = gnn_model(x, edge_index)
        loss = criterion(out, y_gnn)
        loss.backward()
        optimizer.step()
    
    return gnn_model, X_test, y_test

# Geolocation Check
def check_geolocation(transaction, user_history):
    """Validate transaction location."""
    if user_history.empty:
        return True  
    last_location = (user_history['Latitude'].iloc[-1], user_history['Longitude'].iloc[-1])
    current_location = (transaction['Latitude'], transaction['Longitude'])
    return geopy.distance.geodesic(last_location, current_location).km < 500  # 500km threshold

# AI-Powered Voice Authentication
def voice_authentication():
    """Use AI-powered voice authentication for approvals."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        try:
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio)
            return "approve" in text.lower()
        except:
            return False

# Reverse Transaction
def reverse_transaction(transaction_id):
    """Reverse a fraudulent transaction."""
    print(f"🚨 Reversing transaction: {transaction_id}")

# Send Alert
def send_alert(message):
    """Send an alert for suspicious activity."""
    print(f"⚠️ ALERT: {message}")