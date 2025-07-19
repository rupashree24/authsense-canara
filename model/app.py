from flask import Flask, request, jsonify
import torch
import numpy as np
from joblib import load
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load model and scaler
class Autoencoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(6, 4),
            torch.nn.ReLU(),
            torch.nn.Linear(4, 2)
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(2, 4),
            torch.nn.ReLU(),
            torch.nn.Linear(4, 6)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

model = Autoencoder()
model.load_state_dict(torch.load("autoencoder.pth"))
model.eval()
scaler = load("scaler.save")

anomaly_scores = []

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        print("Received:", data)

        vector = np.array([
            data["avgKeystrokeInterval"],
            data["mouseVelocity"],
            data["clickFrequency"],
            data["scrollPattern"],
            data["navigationFlow"],
            data["sessionDuration"]
        ]).reshape(1, -1)

        print("Vector:", vector)

        scaled = scaler.transform(vector)
        print("Scaled:", scaled)

        tensor = torch.tensor(scaled, dtype=torch.float32)
        with torch.no_grad():
            output = model(tensor)
            loss = torch.mean((tensor - output) ** 2).item()

        anomaly_scores.append(loss)
        if len(anomaly_scores) > 10:
            anomaly_scores.pop(0)

        mean = np.mean(anomaly_scores)
        std = np.std(anomaly_scores)
        hijack = loss > (mean + 2 * std)

        result = {
            "anomaly_score": float(loss),
            "risk_score": float(round(loss * 100, 2)),
            "confidence": float(max(20, round(100 - loss * 100, 2))),
            "hijack_detected": bool(hijack)
        }

        print("✅ Returning:", result)
        return jsonify(result)

    except Exception as e:
        print("❌ Error:", e)
        return jsonify({"error": str(e)}), 500


@app.route('/')
def home():
    return "Flask is running"



if __name__ == '__main__':
    app.run(port=5000, debug=True)
