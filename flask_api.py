from flask import Flask, request, jsonify
import pandas as pd
import os
import torch
from flask_cors import CORS
from card_fraud_detect import check_geolocation, reverse_transaction, send_alert, train_models, voice_authentication

app = Flask(__name__)
CORS(app)

DATA_FILE = "Pakistan_Credit_Card_Fraud_2021_2025.csv"
df = pd.read_csv(DATA_FILE) if os.path.exists(DATA_FILE) else pd.DataFrame()

@app.route('/get_city_transactions', methods=['POST'])
def get_city_transactions():
    """Fetch transactions for a given city."""
    try:
        city_name = request.json.get('city', "").strip().title()
        if not city_name:
            return jsonify({"error": "Missing 'city' parameter."}), 400

        if df.empty:
            return jsonify({"error": "Dataset not available."}), 500

        city_transactions = df[df["Transaction_City"].str.title() == city_name]

        if city_transactions.empty:
            return jsonify({"message": f"No transactions found for city: {city_name}"}), 404

        return jsonify({
            "city": city_name,
            "total_transactions": len(city_transactions),
            "transactions": city_transactions.to_dict(orient="records")
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/process_transaction', methods=['POST'])
def process_transaction():
    """Process a new transaction and detect fraud using AI."""
    try:
        transaction_data = request.json
        user_history = df[df['Card_Type'] == transaction_data['Card_Type']]

        # Train AI fraud detection model
        gnn_model, X_test, y_test = train_models(df)
        prediction = gnn_model(torch.tensor(X_test.values, dtype=torch.float), 
                               torch.tensor([[i, j] for i in range(len(X_test)) for j in range(len(X_test)) if i != j], dtype=torch.long).t())

        if prediction[-1] > 0.5:
            send_alert("⚠️ Unusual transaction detected!")
            if not voice_authentication():
                reverse_transaction(transaction_data['Transaction_ID'])
                return jsonify({"message": "Transaction reversed due to fraud."}), 400

        # Geolocation fraud check
        if not check_geolocation(transaction_data, user_history):
            send_alert("📍 Unusual transaction location detected!")
            if not voice_authentication():
                reverse_transaction(transaction_data['Transaction_ID'])
                return jsonify({"message": "Transaction reversed due to location mismatch."}), 400

        return jsonify({"message": "Transaction processed successfully."}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
