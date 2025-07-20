from flask import Flask, request, jsonify
import torch
import numpy as np
from joblib import load
from flask_cors import CORS
import logging
from datetime import datetime
import os
import json
from collections import deque
import hashlib

# Import our training modules
from behavioral_auth import UserAdaptiveBehaviorModel, AdaptiveAutoencoder, PersonalizedBehaviorClassifier

app = Flask(__name__)
CORS(app)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instance
behavior_model = None
active_sessions = {}

def initialize_model():
    """Initialize the behavioral authentication model"""
    global behavior_model
    
    try:
        behavior_model = UserAdaptiveBehaviorModel(model_dir="model")
        
        # Try to load existing models
        if os.path.exists("model/global_autoencoder.pth"):
            behavior_model.load_models()
            logger.info("✅ Pre-trained models loaded successfully")
        else:
            logger.warning("⚠️ No pre-trained models found. Training new models...")
            # Generate and train on synthetic data
            df = behavior_model.create_training_data(n_legitimate=1500, n_attack=300, user_profiles=8)
            X, y, user_ids = behavior_model.preprocess_data(df)
            behavior_model.train_global_model(X, y, epochs=100)
            behavior_model.save_models()
            logger.info("✅ New models trained and saved")
            
    except Exception as e:
        logger.error(f"❌ Error initializing model: {e}")
        behavior_model = None

# Initialize model on startup
initialize_model()

def get_user_id_from_session(session_data):
    """Generate a consistent user ID from session characteristics"""
    # Use session metadata to identify user (could be user_agent, IP, etc.)
    user_string = f"{session_data.get('user_agent', '')}{session_data.get('user_profile', {})}"
    return hashlib.md5(user_string.encode()).hexdigest()[:8]

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        session_id = data.get('session_id', 'default')
        user_agent = request.headers.get('User-Agent', '')
        
        logger.info(f"📊 Analysis request from session: {session_id}")
        
        # Extract behavior vector
        behavior_data = {
            "avgKeystrokeInterval": float(data.get("avgKeystrokeInterval", 150)),
            "mouseVelocity": float(data.get("mouseVelocity", 0.5)),
            "clickFrequency": float(data.get("clickFrequency", 0.1)),
            "scrollPattern": float(data.get("scrollPattern", 0.5)),
            "navigationFlow": float(data.get("navigationFlow", 0.8)),
            "sessionDuration": float(data.get("sessionDuration", 60))
        }

        # Validate input ranges
        for key, value in behavior_data.items():
            behavior_data[key] = max(0, min(value, 10000))  # Reasonable bounds
        
        if behavior_model is None:
            return jsonify({
                "anomaly_score": 0.1,
                "risk_score": 10.0,
                "confidence": 85.0,
                "hijack_detected": False,
                "status": "model_unavailable",
                "user_adapted": False
            })

        # Get or create user ID
        user_id = data.get('user_id') or get_user_id_from_session({
            'user_agent': user_agent,
            'user_profile': data.get('user_profile', {})
        })

        # Make prediction using the advanced model
        prediction = behavior_model.predict(user_id, behavior_data)
        
        # Update session tracking
        if session_id not in active_sessions:
            active_sessions[session_id] = {
                'user_id': user_id,
                'scores': [],
                'start_time': datetime.now(),
                'alerts': 0,
                'legitimate_count': 0,
                'total_predictions': 0
            }
        
        session = active_sessions[session_id]
        session['user_id'] = user_id
        session['scores'].append(prediction['reconstruction_error'])
        session['total_predictions'] += 1
        
        # Keep last 50 scores per session
        if len(session['scores']) > 50:
            session['scores'].pop(0)

        # Determine if behavior is legitimate based on advanced prediction
        hijack_detected = prediction['anomaly_detected']
        risk_score = min(100, (1 - prediction['legitimate_probability']) * 100)
        confidence = prediction['legitimate_probability'] * 100
        
        if hijack_detected:
            session['alerts'] += 1
            logger.warning(f"🚨 Potential hijack detected for user {user_id} in session {session_id}")
        else:
            session['legitimate_count'] += 1
            
        # Adapt model if we have confident legitimate behavior
        if not hijack_detected and prediction['legitimate_probability'] > 0.7:
            behavior_data['label'] = 1  # Mark as legitimate for adaptation
            behavior_model.adapt_user_model(user_id, behavior_data)
            user_adapted = True
        else:
            user_adapted = False

        # Calculate session-based metrics
        session_risk = (session['alerts'] / session['total_predictions']) * 100 if session['total_predictions'] > 0 else 0
        adaptation_score = (session['legitimate_count'] / session['total_predictions']) * 100 if session['total_predictions'] > 0 else 0

        result = {
            "anomaly_score": float(prediction['reconstruction_error']),
            "risk_score": float(risk_score),
            "confidence": float(confidence),
            "legitimate_probability": float(prediction['legitimate_probability']),
            "hijack_detected": bool(hijack_detected),
            "session_risk": float(session_risk),
            "adaptation_score": float(adaptation_score),
            "session_alerts": session['alerts'],
            "total_predictions": session['total_predictions'],
            "user_id": user_id,
            "user_adapted": user_adapted,
            "has_user_model": user_id in behavior_model.user_models,
            "status": "active"
        }

        logger.info(f"✅ Analysis complete - User: {user_id}, Risk: {risk_score:.1f}%, Confidence: {confidence:.1f}%, Hijack: {hijack_detected}")
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
        user_agent = request.headers.get('User-Agent', '')
        
        # Get user ID
        user_id = data.get('user_id') or get_user_id_from_session({
            'user_agent': user_agent,
            'user_profile': data.get('user_profile', {})
        })
        
        active_sessions[session_id] = {
            'user_id': user_id,
            'scores': [],
            'start_time': datetime.now(),
            'alerts': 0,
            'legitimate_count': 0,
            'total_predictions': 0,
            'user_profile': data.get('user_profile', {}),
            'initial_behavior': data.get('initial_behavior', {})
        }
        
        # Create user model if it doesn't exist and we have initial behavior data
        if (behavior_model and user_id not in behavior_model.user_models 
            and data.get('initial_behavior')):
            try:
                behavior_model.create_user_model(user_id, [data['initial_behavior']])
                model_created = True
            except Exception as e:
                logger.error(f"Error creating user model: {e}")
                model_created = False
        else:
            model_created = user_id in behavior_model.user_models if behavior_model else False
        
        logger.info(f"🚀 New session started: {session_id} for user: {user_id}")
        return jsonify({
            "session_id": session_id, 
            "user_id": user_id,
            "status": "started",
            "user_model_created": model_created,
            "has_existing_model": user_id in behavior_model.user_models if behavior_model else False
        })
        
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
                "user_id": session.get('user_id', 'unknown'),
                "duration": duration,
                "total_analyses": len(session['scores']),
                "total_predictions": session['total_predictions'],
                "alerts_count": session['alerts'],
                "legitimate_count": session['legitimate_count'],
                "avg_anomaly": np.mean(session['scores']) if session['scores'] else 0,
                "session_risk": (session['alerts'] / session['total_predictions']) * 100 if session['total_predictions'] > 0 else 0,
                "adaptation_rate": (session['legitimate_count'] / session['total_predictions']) * 100 if session['total_predictions'] > 0 else 0,
                "status": "completed"
            }
            
            # Save models after session ends (user adaptation completed)
            if behavior_model:
                try:
                    behavior_model.save_models()
                    logger.info("💾 Models saved after session completion")
                except Exception as e:
                    logger.error(f"Error saving models: {e}")
            
            del active_sessions[session_id]
            logger.info(f"🏁 Session ended: {session_id}")
            return jsonify(summary)
        else:
            return jsonify({"error": "Session not found"}), 404
            
    except Exception as e:
        logger.error(f"❌ Session end error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/user/profile', methods=['GET', 'POST'])
def user_profile():
    """Get or update user behavioral profile"""
    try:
        if request.method == 'GET':
            user_id = request.args.get('user_id')
            if not user_id or not behavior_model:
                return jsonify({"error": "User ID required or model not available"}), 400
            
            has_model = user_id in behavior_model.user_models
            buffer_size = len(behavior_model.user_buffers.get(user_id, [])) if has_model else 0
            
            return jsonify({
                "user_id": user_id,
                "has_personalized_model": has_model,
                "training_samples": buffer_size,
                "model_status": "active" if has_model else "using_global"
            })
            
        elif request.method == 'POST':
            data = request.json
            user_id = data.get('user_id')
            training_data = data.get('training_data', [])
            
            if not user_id or not behavior_model:
                return jsonify({"error": "User ID and training data required"}), 400
            
            # Create or update user model with provided training data
            if user_id not in behavior_model.user_models:
                behavior_model.create_user_model(user_id, training_data)
                created = True
            else:
                # Add training data to existing model
                for sample in training_data:
                    behavior_model.adapt_user_model(user_id, sample)
                created = False
            
            return jsonify({
                "user_id": user_id,
                "model_created": created,
                "training_samples_added": len(training_data),
                "status": "success"
            })
            
    except Exception as e:
        logger.error(f"❌ User profile error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/model/retrain', methods=['POST'])
def retrain_model():
    """Retrain the global model with new data"""
    try:
        data = request.json
        epochs = data.get('epochs', 50)
        
        if not behavior_model:
            return jsonify({"error": "Model not initialized"}), 500
        
        # Generate new training data (in production, this would come from real data)
        logger.info("🔄 Retraining global model...")
        df = behavior_model.create_training_data(
            n_legitimate=data.get('n_legitimate', 1000), 
            n_attack=data.get('n_attack', 200),
            user_profiles=data.get('user_profiles', 5)
        )
        
        X, y, user_ids = behavior_model.preprocess_data(df)
        behavior_model.train_global_model(X, y, epochs=epochs)
        behavior_model.save_models()
        
        return jsonify({
            "status": "success",
            "message": "Global model retrained successfully",
            "training_samples": len(X),
            "epochs": epochs
        })
        
    except Exception as e:
        logger.error(f"❌ Retrain error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/analytics/dashboard', methods=['GET'])
def analytics_dashboard():
    """Get analytics data for dashboard"""
    try:
        if not behavior_model:
            return jsonify({"error": "Model not available"}), 500
        
        # Aggregate session statistics
        total_sessions = len(active_sessions)
        total_users = len(set(session.get('user_id', 'unknown') for session in active_sessions.values()))
        total_predictions = sum(session['total_predictions'] for session in active_sessions.values())
        total_alerts = sum(session['alerts'] for session in active_sessions.values())
        
        # User model statistics
        personalized_users = len(behavior_model.user_models)
        
        # Recent activity (last hour)
        current_time = datetime.now()
        recent_sessions = [
            session for session in active_sessions.values() 
            if (current_time - session['start_time']).total_seconds() < 3600
        ]
        
        analytics = {
            "overview": {
                "active_sessions": total_sessions,
                "unique_users": total_users,
                "personalized_users": personalized_users,
                "total_predictions": total_predictions,
                "total_alerts": total_alerts,
                "global_alert_rate": (total_alerts / total_predictions * 100) if total_predictions > 0 else 0
            },
            "recent_activity": {
                "sessions_last_hour": len(recent_sessions),
                "predictions_last_hour": sum(session['total_predictions'] for session in recent_sessions),
                "alerts_last_hour": sum(session['alerts'] for session in recent_sessions)
            },
            "model_status": {
                "global_model_loaded": True,
                "user_models_count": personalized_users,
                "model_version": "1.0_adaptive"
            }
        }
        
        return jsonify(analytics)
        
    except Exception as e:
        logger.error(f"❌ Analytics error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    model_status = "healthy" if behavior_model else "unavailable"
    user_models_loaded = len(behavior_model.user_models) if behavior_model else 0
    
    return jsonify({
        "status": "healthy",
        "model_status": model_status,
        "global_model_loaded": behavior_model is not None,
        "user_models_loaded": user_models_loaded,
        "active_sessions": len(active_sessions),
        "timestamp": datetime.now().isoformat(),
        "version": "2.0_adaptive"
    })

@app.route('/')
def home():
    return jsonify({
        "service": "AuthSense ML Backend - Adaptive Behavioral Authentication",
        "version": "2.0",
        "features": [
            "User-adaptive behavioral modeling",
            "Real-time anomaly detection", 
            "Personalized authentication",
            "Session-based learning"
        ],
        "endpoints": [
            "/analyze", "/session/start", "/session/end", 
            "/user/profile", "/model/retrain", "/analytics/dashboard", "/health"
        ]
    })

@app.route('/test/demo', methods=['GET'])
def demo():
    """Demo endpoint to test the system"""
    try:
        if not behavior_model:
            return jsonify({"error": "Model not available"}), 500
        
        # Simulate different user behaviors
        demo_users = {
            "legitimate_user": {
                "avgKeystrokeInterval": 140.0,
                "mouseVelocity": 0.85,
                "clickFrequency": 0.16,
                "scrollPattern": 0.65,
                "navigationFlow": 0.87,
                "sessionDuration": 320.0
            },
            "suspicious_user": {
                "avgKeystrokeInterval": 80.0,  # Much faster typing
                "mouseVelocity": 2.5,          # Erratic mouse movement
                "clickFrequency": 0.8,         # Excessive clicking
                "scrollPattern": 0.2,          # Unusual scroll pattern
                "navigationFlow": 0.3,         # Poor navigation
                "sessionDuration": 30.0        # Very short session
            },
            "bot_like": {
                "avgKeystrokeInterval": 50.0,  # Extremely fast, consistent
                "mouseVelocity": 0.1,          # Very low mouse activity
                "clickFrequency": 2.0,         # Mechanical clicking
                "scrollPattern": 1.0,          # Perfect scroll pattern
                "navigationFlow": 1.0,         # Perfect navigation
                "sessionDuration": 600.0       # Long session
            }
        }
        
        results = {}
        for user_type, behavior in demo_users.items():
            prediction = behavior_model.predict(f"demo_{user_type}", behavior)
            results[user_type] = {
                "behavior": behavior,
                "prediction": prediction,
                "interpretation": {
                    "likely_legitimate": prediction['legitimate_probability'] > 0.5,
                    "risk_level": "low" if prediction['legitimate_probability'] > 0.7 else "medium" if prediction['legitimate_probability'] > 0.3 else "high"
                }
            }
        
        return jsonify({
            "demo_results": results,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Demo error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    logger.info("🚀 Starting Enhanced AuthSense ML Backend...")
    logger.info("🧠 Features: User-adaptive behavioral authentication")
    app.run(host='0.0.0.0', port=5000, debug=True)