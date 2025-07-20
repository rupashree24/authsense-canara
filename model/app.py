from flask import Flask, request, jsonify
import torch
import numpy as np
from joblib import load
from flask_cors import CORS
import logging
from datetime import datetime
import os

app = Flask(__name__)
CORS(app)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Initialize model and scaler
try:
    model = Autoencoder()
    if os.path.exists("autoencoder.pth"):
        model.load_state_dict(torch.load("autoencoder.pth", map_location='cpu'))
        model.eval()
        logger.info("✅ Model loaded successfully")
    else:
        logger.warning("⚠️ Model file not found, using untrained model")
    
    if os.path.exists("scaler.save"):
        scaler = load("scaler.save")
        logger.info("✅ Scaler loaded successfully")
    else:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        # Initialize with dummy data if scaler doesn't exist
        dummy_data = np.random.rand(100, 6)
        scaler.fit(dummy_data)
        logger.warning("⚠️ Using dummy scaler")
        
except Exception as e:
    logger.error(f"❌ Error loading model/scaler: {e}")
    model = None
    scaler = None

# Session tracking
active_sessions = {}
anomaly_scores = []

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        session_id = data.get('session_id', 'default')
        
        # Log received data
        logger.info(f"📊 Analysis request from session: {session_id}")
        
        # Extract behavior vector
        vector = np.array([
            float(data.get("avgKeystrokeInterval", 150)),
            float(data.get("mouseVelocity", 0.5)),
            float(data.get("clickFrequency", 0.1)),
            float(data.get("scrollPattern", 0.5)),
            float(data.get("navigationFlow", 0.8)),
            float(data.get("sessionDuration", 60))
        ]).reshape(1, -1)

        # Validate input ranges
        vector = np.clip(vector, 0, 10000)  # Reasonable bounds
        
        if model is None or scaler is None:
            # Fallback analysis without ML model
            return jsonify({
                "anomaly_score": 0.1,
                "risk_score": 10.0,
                "confidence": 85.0,
                "hijack_detected": False,
                "status": "model_unavailable"
            })

        # Scale and predict
        scaled = scaler.transform(vector)
        tensor = torch.tensor(scaled, dtype=torch.float32)
        
        with torch.no_grad():
            output = model(tensor)
            loss = torch.mean((tensor - output) ** 2).item()

        # Update session tracking
        if session_id not in active_sessions:
            active_sessions[session_id] = {
                'scores': [],
                'start_time': datetime.now(),
                'alerts': 0
            }
        
        session = active_sessions[session_id]
        session['scores'].append(loss)
        
        # Keep last 20 scores per session
        if len(session['scores']) > 20:
            session['scores'].pop(0)

        # Calculate adaptive threshold
        if len(session['scores']) >= 5:
            mean = np.mean(session['scores'])
            std = np.std(session['scores'])
            threshold = mean + (2 * std) + 0.1  # Adaptive threshold
        else:
            threshold = 0.3  # Default threshold

        # Determine hijack status
        hijack_detected = loss > threshold
        if hijack_detected:
            session['alerts'] += 1
            logger.warning(f"🚨 Potential hijack detected in session {session_id}")

        # Calculate risk and confidence
        risk_score = min(100, loss * 200)  # Scale to 0-100
        confidence = max(20, 100 - risk_score)

        result = {
            "anomaly_score": float(loss),
            "risk_score": float(risk_score),
            "confidence": float(confidence),
            "hijack_detected": bool(hijack_detected),
            "threshold": float(threshold),
            "session_alerts": session['alerts'],
            "status": "active"
        }

        logger.info(f"✅ Analysis complete - Risk: {risk_score:.1f}%, Hijack: {hijack_detected}")
        return jsonify(result)

    except Exception as e:
        logger.error(f"❌ Analysis error: {e}")
        return jsonify({
            "error": str(e),
            "anomaly_score": 0.5,
            "risk_score": 50.0,
            "confidence": 50.0,
            "hijack_detected": True,
            "status": "error"
        }), 500

@app.route('/session/start', methods=['POST'])
def start_session():
    try:
        data = request.json
        session_id = data.get('session_id') or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        active_sessions[session_id] = {
            'scores': [],
            'start_time': datetime.now(),
            'alerts': 0,
            'user_profile': data.get('user_profile', {})
        }
        
        logger.info(f"🚀 New session started: {session_id}")
        return jsonify({"session_id": session_id, "status": "started"})
        
    except Exception as e:
        logger.error(f"❌ Session start error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/session/end', methods=['POST'])
def end_session():
    try:
        data = request.json
        session_id = data.get('session_id')
        
        if session_id in active_sessions:
            session = active_sessions[session_id]
            duration = (datetime.now() - session['start_time']).total_seconds()
            
            summary = {
                "session_id": session_id,
                "duration": duration,
                "total_analyses": len(session['scores']),
                "alerts_count": session['alerts'],
                "avg_anomaly": np.mean(session['scores']) if session['scores'] else 0,
                "status": "completed"
            }
            
            del active_sessions[session_id]
            logger.info(f"🏁 Session ended: {session_id}")
            return jsonify(summary)
        else:
            return jsonify({"error": "Session not found"}), 404
            
    except Exception as e:
        logger.error(f"❌ Session end error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "active_sessions": len(active_sessions),
        "timestamp": datetime.now().isoformat()
    })

@app.route('/')
def home():
    return jsonify({
        "service": "AuthSense ML Backend",
        "version": "1.0",
        "endpoints": ["/analyze", "/session/start", "/session/end", "/health"]
    })

if __name__ == '__main__':
    logger.info("🚀 Starting AuthSense ML Backend...")
    app.run(host='0.0.0.0', port=5000, debug=True)