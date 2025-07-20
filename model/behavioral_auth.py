import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump, load
import logging
from datetime import datetime
import os
import json
from collections import deque

# Enhanced Autoencoder for Behavioral Authentication
class AdaptiveAutoencoder(nn.Module):
    def __init__(self, input_dim=6, hidden_dims=[8, 4], latent_dim=2):
        super(AdaptiveAutoencoder, self).__init__()
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        
        self.decoder = nn.Sequential(*decoder_layers)
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)

class PersonalizedBehaviorClassifier(nn.Module):
    """Classification model for behavioral authentication"""
    def __init__(self, input_dim=6, hidden_dims=[16, 8], num_classes=2):
        super(PersonalizedBehaviorClassifier, self).__init__()
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        self.classifier = nn.Sequential(*layers)
        
    def forward(self, x):
        return torch.softmax(self.classifier(x), dim=1)

class UserAdaptiveBehaviorModel:
    """Main class for user-adaptive behavioral authentication"""
    
    def __init__(self, model_dir="model"):
        self.model_dir = model_dir
        self.feature_names = [
            'avgKeystrokeInterval', 'mouseVelocity', 'clickFrequency', 
            'scrollPattern', 'navigationFlow', 'sessionDuration'
        ]
        
        # Models
        self.global_autoencoder = AdaptiveAutoencoder()
        self.global_classifier = PersonalizedBehaviorClassifier()
        self.user_models = {}  # Store per-user adaptive models
        
        # Scalers
        self.global_scaler = StandardScaler()
        self.user_scalers = {}
        
        # User behavior buffers for online learning
        self.user_buffers = {}
        self.buffer_size = 100
        
        # Training parameters
        self.learning_rate = 0.001
        self.adaptation_rate = 0.0001  # Lower rate for user adaptation
        
        self.setup_logging()
        self.ensure_model_dir()
        
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def ensure_model_dir(self):
        os.makedirs(self.model_dir, exist_ok=True)
        
    def create_training_data(self, n_legitimate=1000, n_attack=200, user_profiles=5):
        """Generate synthetic training data with realistic behavioral patterns"""
        
        np.random.seed(42)
        data = []
        
        # Generate legitimate user profiles
        for user_id in range(user_profiles):
            # Each user has their own behavioral baseline
            user_baseline = {
                'avgKeystrokeInterval': np.random.normal(120 + user_id * 20, 15),
                'mouseVelocity': np.random.normal(0.8 + user_id * 0.1, 0.1),
                'clickFrequency': np.random.normal(0.15 + user_id * 0.02, 0.02),
                'scrollPattern': np.random.normal(0.6 + user_id * 0.05, 0.08),
                'navigationFlow': np.random.normal(0.85 + user_id * 0.02, 0.05),
                'sessionDuration': np.random.normal(300 + user_id * 50, 30)
            }
            
            # Generate legitimate samples for this user
            n_user_samples = n_legitimate // user_profiles
            for _ in range(n_user_samples):
                sample = {}
                for feature, baseline in user_baseline.items():
                    # Add some natural variation
                    sample[feature] = max(0, np.random.normal(baseline, baseline * 0.1))
                
                sample['label'] = 1  # Legitimate
                sample['user_id'] = user_id
                data.append(sample)
                
            # Generate attack samples (different behavioral patterns)
            n_attack_user = n_attack // user_profiles
            for _ in range(n_attack_user):
                sample = {}
                for feature, baseline in user_baseline.items():
                    if feature == 'avgKeystrokeInterval':
                        # Attackers might type faster/slower
                        sample[feature] = max(0, np.random.normal(baseline * np.random.choice([0.5, 1.8]), baseline * 0.2))
                    elif feature == 'mouseVelocity':
                        # Different mouse patterns
                        sample[feature] = max(0, np.random.normal(baseline * np.random.choice([0.3, 2.0]), baseline * 0.3))
                    elif feature == 'clickFrequency':
                        # Unusual clicking patterns
                        sample[feature] = max(0, np.random.normal(baseline * np.random.choice([0.2, 3.0]), baseline * 0.4))
                    else:
                        # General behavioral differences
                        sample[feature] = max(0, np.random.normal(baseline * np.random.uniform(0.4, 2.2), baseline * 0.3))
                
                sample['label'] = 0  # Attack
                sample['user_id'] = user_id
                data.append(sample)
        
        df = pd.DataFrame(data)
        self.logger.info(f"Generated {len(df)} training samples")
        return df
        
    def preprocess_data(self, df):
        """Preprocess the training data"""
        X = df[self.feature_names].values
        y = df['label'].values
        user_ids = df['user_id'].values
        
        # Handle any infinite or NaN values
        X = np.nan_to_num(X, nan=0.0, posinf=1000.0, neginf=0.0)
        
        return X, y, user_ids
        
    def train_global_model(self, X, y, epochs=100):
        """Train the global autoencoder and classifier"""
        
        # Fit scaler
        self.global_scaler.fit(X)
        X_scaled = self.global_scaler.transform(X)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_scaled)
        y_tensor = torch.LongTensor(y)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_tensor, y_tensor, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train autoencoder
        self.logger.info("Training global autoencoder...")
        ae_optimizer = optim.Adam(self.global_autoencoder.parameters(), lr=self.learning_rate)
        ae_criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            self.global_autoencoder.train()
            ae_optimizer.zero_grad()
            
            reconstructed = self.global_autoencoder(X_train)
            loss = ae_criterion(reconstructed, X_train)
            
            loss.backward()
            ae_optimizer.step()
            
            if epoch % 20 == 0:
                self.logger.info(f"Autoencoder Epoch {epoch}, Loss: {loss.item():.4f}")
        
        # Train classifier
        self.logger.info("Training global classifier...")
        clf_optimizer = optim.Adam(self.global_classifier.parameters(), lr=self.learning_rate)
        clf_criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            self.global_classifier.train()
            clf_optimizer.zero_grad()
            
            outputs = self.global_classifier(X_train)
            loss = clf_criterion(outputs, y_train)
            
            loss.backward()
            clf_optimizer.step()
            
            if epoch % 20 == 0:
                # Validation accuracy
                self.global_classifier.eval()
                with torch.no_grad():
                    val_outputs = self.global_classifier(X_val)
                    _, predicted = torch.max(val_outputs, 1)
                    accuracy = (predicted == y_val).float().mean()
                    self.logger.info(f"Classifier Epoch {epoch}, Loss: {loss.item():.4f}, Val Acc: {accuracy:.4f}")
    
    def create_user_model(self, user_id, initial_data=None):
        """Create a personalized model for a specific user"""
        
        # Clone global model for user-specific adaptation
        user_autoencoder = AdaptiveAutoencoder()
        user_autoencoder.load_state_dict(self.global_autoencoder.state_dict())
        
        user_classifier = PersonalizedBehaviorClassifier()
        user_classifier.load_state_dict(self.global_classifier.state_dict())
        
        user_scaler = StandardScaler()
        user_scaler.mean_ = self.global_scaler.mean_.copy()
        user_scaler.scale_ = self.global_scaler.scale_.copy()
        user_scaler.var_ = self.global_scaler.var_.copy()
        user_scaler.n_samples_seen_ = self.global_scaler.n_samples_seen_
        
        self.user_models[user_id] = {
            'autoencoder': user_autoencoder,
            'classifier': user_classifier,
            'ae_optimizer': optim.Adam(user_autoencoder.parameters(), lr=self.adaptation_rate),
            'clf_optimizer': optim.Adam(user_classifier.parameters(), lr=self.adaptation_rate)
        }
        
        self.user_scalers[user_id] = user_scaler
        self.user_buffers[user_id] = deque(maxlen=self.buffer_size)
        
        # If initial data is provided, do initial adaptation
        if initial_data is not None:
            self.adapt_user_model(user_id, initial_data)
            
        self.logger.info(f"Created personalized model for user {user_id}")
        
    def adapt_user_model(self, user_id, behavior_data):
        """Adapt user model with new behavioral data"""
        
        if user_id not in self.user_models:
            self.create_user_model(user_id)
            
        # Add to user buffer
        self.user_buffers[user_id].append(behavior_data)
        
        # Only adapt if we have enough samples
        if len(self.user_buffers[user_id]) < 5:
            return
            
        # Get recent samples for adaptation
        recent_samples = list(self.user_buffers[user_id])[-20:]  # Last 20 samples
        
        X = np.array([[
            sample['avgKeystrokeInterval'],
            sample['mouseVelocity'], 
            sample['clickFrequency'],
            sample['scrollPattern'],
            sample['navigationFlow'],
            sample['sessionDuration']
        ] for sample in recent_samples])
        
        # Get labels (assume legitimate for adaptation, or use provided labels)
        y = np.array([sample.get('label', 1) for sample in recent_samples])
        
        # Scale data with user scaler
        X_scaled = self.user_scalers[user_id].transform(X)
        X_tensor = torch.FloatTensor(X_scaled)
        y_tensor = torch.LongTensor(y)
        
        user_model = self.user_models[user_id]
        
        # Adapt autoencoder (unsupervised)
        user_model['autoencoder'].train()
        user_model['ae_optimizer'].zero_grad()
        
        reconstructed = user_model['autoencoder'](X_tensor)
        ae_loss = nn.MSELoss()(reconstructed, X_tensor)
        ae_loss.backward()
        user_model['ae_optimizer'].step()
        
        # Adapt classifier if we have labels
        if len(set(y)) > 1:  # Only if we have both classes
            user_model['classifier'].train()
            user_model['clf_optimizer'].zero_grad()
            
            outputs = user_model['classifier'](X_tensor)
            clf_loss = nn.CrossEntropyLoss()(outputs, y_tensor)
            clf_loss.backward()
            user_model['clf_optimizer'].step()
        
        self.logger.info(f"Adapted model for user {user_id} - AE Loss: {ae_loss.item():.4f}")
        
    def predict(self, user_id, behavior_data):
        """Make prediction for user behavior"""
        
        # Prepare input
        X = np.array([[
            behavior_data['avgKeystrokeInterval'],
            behavior_data['mouseVelocity'],
            behavior_data['clickFrequency'],
            behavior_data['scrollPattern'],
            behavior_data['navigationFlow'],
            behavior_data['sessionDuration']
        ]])
        
        # Use user-specific model if available, otherwise global
        if user_id in self.user_models:
            autoencoder = self.user_models[user_id]['autoencoder']
            classifier = self.user_models[user_id]['classifier']
            scaler = self.user_scalers[user_id]
        else:
            autoencoder = self.global_autoencoder
            classifier = self.global_classifier
            scaler = self.global_scaler
        
        # Scale and predict
        X_scaled = scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled)
        
        with torch.no_grad():
            # Autoencoder reconstruction error
            reconstructed = autoencoder(X_tensor)
            reconstruction_error = torch.mean((X_tensor - reconstructed) ** 2).item()
            
            # Classifier prediction
            classifier.eval()
            with torch.no_grad():
                probs = classifier(X_tensor)
            legitimate_prob = probs[0][1].item()
            
        return {
            'reconstruction_error': reconstruction_error,
            'legitimate_probability': legitimate_prob,
            'anomaly_detected': reconstruction_error > 0.5 or legitimate_prob < 0.5
        }
        
    def save_models(self):
        """Save all models and scalers"""
        
        # Save global models
        torch.save(self.global_autoencoder.state_dict(), 
                  os.path.join(self.model_dir, 'global_autoencoder.pth'))
        torch.save(self.global_classifier.state_dict(), 
                  os.path.join(self.model_dir, 'global_classifier.pth'))
        dump(self.global_scaler, os.path.join(self.model_dir, 'global_scaler.save'))
        
        # Save user models
        user_models_dir = os.path.join(self.model_dir, 'user_models')
        os.makedirs(user_models_dir, exist_ok=True)
        
        for user_id, models in self.user_models.items():
            user_dir = os.path.join(user_models_dir, f'user_{user_id}')
            os.makedirs(user_dir, exist_ok=True)
            
            torch.save(models['autoencoder'].state_dict(), 
                      os.path.join(user_dir, 'autoencoder.pth'))
            torch.save(models['classifier'].state_dict(), 
                      os.path.join(user_dir, 'classifier.pth'))
            dump(self.user_scalers[user_id], 
                 os.path.join(user_dir, 'scaler.save'))
        
        self.logger.info("All models saved successfully")
        
    def load_models(self):
        """Load saved models"""
        
        try:
            # Load global models
            self.global_autoencoder.load_state_dict(
                torch.load(os.path.join(self.model_dir, 'global_autoencoder.pth'), map_location='cpu')
            )
            self.global_classifier.load_state_dict(
                torch.load(os.path.join(self.model_dir, 'global_classifier.pth'), map_location='cpu')
            )
            self.global_scaler = load(os.path.join(self.model_dir, 'global_scaler.save'))
            
            # Load user models if they exist
            user_models_dir = os.path.join(self.model_dir, 'user_models')
            if os.path.exists(user_models_dir):
                for user_folder in os.listdir(user_models_dir):
                    if user_folder.startswith('user_'):
                        user_id = user_folder.replace('user_', '')
                        user_dir = os.path.join(user_models_dir, user_folder)
                        
                        # Load user models
                        user_autoencoder = AdaptiveAutoencoder()
                        user_autoencoder.load_state_dict(
                            torch.load(os.path.join(user_dir, 'autoencoder.pth'), map_location='cpu')
                        )
                        
                        user_classifier = PersonalizedBehaviorClassifier()
                        user_classifier.load_state_dict(
                            torch.load(os.path.join(user_dir, 'classifier.pth'), map_location='cpu')
                        )
                        
                        user_scaler = load(os.path.join(user_dir, 'scaler.save'))
                        
                        self.user_models[user_id] = {
                            'autoencoder': user_autoencoder,
                            'classifier': user_classifier,
                            'ae_optimizer': optim.Adam(user_autoencoder.parameters(), lr=self.adaptation_rate),
                            'clf_optimizer': optim.Adam(user_classifier.parameters(), lr=self.adaptation_rate)
                        }
                        
                        self.user_scalers[user_id] = user_scaler
                        self.user_buffers[user_id] = deque(maxlen=self.buffer_size)
            
            self.logger.info("Models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")

def main():
    """Main training and demonstration function"""
    
    # Initialize the behavioral model
    behavior_model = UserAdaptiveBehaviorModel()
    
    # Generate training data
    print("Generating training data...")
    df = behavior_model.create_training_data(n_legitimate=2000, n_attack=400, user_profiles=10)
    
    # Preprocess data
    X, y, user_ids = behavior_model.preprocess_data(df)
    
    # Train global model
    print("Training global models...")
    behavior_model.train_global_model(X, y, epochs=150)
    
    # Create user models for demonstration
    print("Creating user-specific models...")
    for user_id in range(3):  # Create models for first 3 users
        user_data = df[df['user_id'] == user_id]
        initial_samples = user_data.head(20).to_dict('records')
        behavior_model.create_user_model(str(user_id), initial_samples)
    
    # Save models
    behavior_model.save_models()
    
    # Demonstration
    print("\nDemonstration:")
    test_behavior = {
        'avgKeystrokeInterval': 140.0,
        'mouseVelocity': 0.85,
        'clickFrequency': 0.16,
        'scrollPattern': 0.65,
        'navigationFlow': 0.87,
        'sessionDuration': 320.0
    }
    
    # Test with different users
    for user_id in ['0', '1', '2', '999']:  # 999 is unknown user
        result = behavior_model.predict(user_id, test_behavior)
        print(f"User {user_id}: {result}")
        
        # Adapt model with this new sample
        test_behavior['label'] = 1  # Assume legitimate for adaptation
        behavior_model.adapt_user_model(user_id, test_behavior)
    
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()