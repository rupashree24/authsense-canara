#!/usr/bin/env python3
"""
Behavioral Authentication Model Training Script

This script trains the adaptive behavioral authentication models with your specified parameters:
- avgKeystrokeInterval
- mouseVelocity  
- clickFrequency
- scrollPattern
- navigationFlow
- sessionDuration
- label (for supervised learning)

Usage:
    python train_models.py [options]
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from behavioral_auth import UserAdaptiveBehaviorModel
import logging
import os
from datetime import datetime
import json
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')



def setup_logging(log_level='INFO'):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_real_data(csv_path):
    """Load real behavioral data from CSV file"""
    logger = logging.getLogger(__name__)
    
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"✅ Loaded {len(df)} samples from {csv_path}")
        
        # Validate required columns
        required_columns = [
            'avgKeystrokeInterval', 'mouseVelocity', 'clickFrequency',
            'scrollPattern', 'navigationFlow', 'sessionDuration', 'label'
        ]
        
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            logger.error(f"❌ Missing required columns: {missing_columns}")
            return None
            
        # Clean data
        df = df.dropna()
        
        # Ensure label is binary (0 for attack/anomaly, 1 for legitimate)
        df['label'] = df['label'].astype(int)
        
        # Add user_id if not present (for demonstration)
        if 'user_id' not in df.columns:
            # Create synthetic user IDs based on behavioral patterns
            df['user_id'] = pd.cut(df['avgKeystrokeInterval'], 
                                 bins=5, labels=range(5)).astype(str)
        
        logger.info(f"📊 Data distribution: {df['label'].value_counts().to_dict()}")
        return df
        
    except Exception as e:
        logger.error(f"❌ Error loading data: {e}")
        return None

def create_synthetic_realistic_data(n_users=20, samples_per_user=200, attack_ratio=0.15):
    """Create more realistic synthetic data based on research patterns"""
    logger = logging.getLogger(__name__)
    logger.info(f"🔧 Generating realistic synthetic data for {n_users} users...")
    
    np.random.seed(42)
    data = []
    
    # Define realistic user archetypes
    user_archetypes = {
        'fast_typer': {
            'avgKeystrokeInterval': (80, 15),   # Fast typing
            'mouseVelocity': (1.2, 0.2),       # Active mouse user
            'clickFrequency': (0.25, 0.05),    # Frequent clicking
            'scrollPattern': (0.8, 0.1),       # Smooth scrolling
            'navigationFlow': (0.9, 0.05),     # Efficient navigation
            'sessionDuration': (450, 60)       # Long sessions
        },
        'casual_user': {
            'avgKeystrokeInterval': (150, 25),  # Moderate typing
            'mouseVelocity': (0.8, 0.15),      # Normal mouse usage
            'clickFrequency': (0.15, 0.03),    # Normal clicking
            'scrollPattern': (0.6, 0.15),      # Varied scrolling
            'navigationFlow': (0.7, 0.1),      # Standard navigation
            'sessionDuration': (280, 45)       # Medium sessions
        },
        'slow_careful': {
            'avgKeystrokeInterval': (220, 30),  # Slow, careful typing
            'mouseVelocity': (0.5, 0.1),       # Deliberate mouse movement
            'clickFrequency': (0.08, 0.02),    # Infrequent clicking
            'scrollPattern': (0.4, 0.12),      # Careful scrolling
            'navigationFlow': (0.6, 0.08),     # Methodical navigation
            'sessionDuration': (180, 30)       # Shorter sessions
        },
        'power_user': {
            'avgKeystrokeInterval': (100, 20),  # Fast, efficient typing
            'mouseVelocity': (1.5, 0.25),      # Dynamic mouse usage
            'clickFrequency': (0.3, 0.06),     # Heavy clicking
            'scrollPattern': (0.9, 0.08),      # Expert scrolling
            'navigationFlow': (0.95, 0.03),    # Very efficient navigation
            'sessionDuration': (600, 90)       # Extended sessions
        }
    }
    
    archetype_names = list(user_archetypes.keys())
    
    for user_id in range(n_users):
        # Assign archetype to user
        archetype_name = np.random.choice(archetype_names)
        archetype = user_archetypes[archetype_name]
        
        # Generate legitimate samples
        n_legitimate = int(samples_per_user * (1 - attack_ratio))
        for _ in range(n_legitimate):
            sample = {'user_id': user_id, 'label': 1}  # Legitimate
            
            for feature, (mean, std) in archetype.items():
                # Add daily variation (users change slightly day to day)
                daily_factor = np.random.normal(1.0, 0.05)
                sample[feature] = max(0, np.random.normal(mean * daily_factor, std))
            
            data.append(sample)
        
        # Generate attack samples (behavioral anomalies)
        n_attacks = samples_per_user - n_legitimate
        for _ in range(n_attacks):
            sample = {'user_id': user_id, 'label': 0}  # Attack
            
            for feature, (mean, std) in archetype.items():
                if np.random.random() < 0.7:  # 70% chance to modify this feature
                    if feature == 'avgKeystrokeInterval':
                        # Attackers might use automated tools (very fast) or be unfamiliar (slow)
                        attack_factor = np.random.choice([0.3, 0.4, 2.5, 3.0])
                    elif feature == 'mouseVelocity':
                        # Unusual mouse patterns
                        attack_factor = np.random.choice([0.1, 0.2, 3.0, 4.0])
                    elif feature == 'clickFrequency':
                        # Bot-like clicking or confusion clicking
                        attack_factor = np.random.choice([0.05, 5.0, 8.0])
                    elif feature == 'scrollPattern':
                        # Erratic or mechanical scrolling
                        attack_factor = np.random.choice([0.1, 0.95, 1.0])
                    elif feature == 'navigationFlow':
                        # Poor navigation or too perfect (bot-like)
                        attack_factor = np.random.choice([0.2, 0.3, 0.99, 1.0])
                    else:  # sessionDuration
                        # Very short (hit-and-run) or very long (persistence)
                        attack_factor = np.random.choice([0.1, 0.2, 3.0, 5.0])
                    
                    sample[feature] = max(0, np.random.normal(mean * attack_factor, std * 1.5))
                else:
                    # Keep some features normal to make detection harder
                    sample[feature] = max(0, np.random.normal(mean, std))
            
            data.append(sample)
    
    df = pd.DataFrame(data)
    logger.info(f"✅ Generated {len(df)} samples with distribution: {df['label'].value_counts().to_dict()}")
    return df

def evaluate_model(behavior_model, test_data):
    """Comprehensive model evaluation"""
    logger = logging.getLogger(__name__)
    logger.info("🔍 Evaluating model performance...")
    
    predictions = []
    true_labels = []
    
    for _, row in test_data.iterrows():
        behavior_data = {
            'avgKeystrokeInterval': row['avgKeystrokeInterval'],
            'mouseVelocity': row['mouseVelocity'],
            'clickFrequency': row['clickFrequency'],
            'scrollPattern': row['scrollPattern'],
            'navigationFlow': row['navigationFlow'],
            'sessionDuration': row['sessionDuration']
        }
        
        # Test with both global and user-specific models
        user_id = str(row['user_id'])
        prediction = behavior_model.predict(user_id, behavior_data)
        
        predictions.append(prediction['legitimate_probability'])
        true_labels.append(row['label'])
    
    # Convert to binary predictions
    binary_predictions = [1 if p > 0.5 else 0 for p in predictions]
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(true_labels, binary_predictions)
    precision = precision_score(true_labels, binary_predictions)
    recall = recall_score(true_labels, binary_predictions)
    f1 = f1_score(true_labels, binary_predictions)
    auc = roc_auc_score(true_labels, predictions)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_roc': auc
    }
    
    logger.info(f"📈 Model Performance Metrics:")
    logger.info(f"   Accuracy:  {accuracy:.3f}")
    logger.info(f"   Precision: {precision:.3f}")
    logger.info(f"   Recall:    {recall:.3f}")
    logger.info(f"   F1-Score:  {f1:.3f}")
    logger.info(f"   AUC-ROC:   {auc:.3f}")
    
    # Classification report
    logger.info("\n📊 Detailed Classification Report:")
    print(classification_report(true_labels, binary_predictions, 
                              target_names=['Attack', 'Legitimate']))
    
    return metrics

def plot_training_results(behavior_model, test_data, save_dir='plots'):
    """Create visualization plots"""
    logger = logging.getLogger(__name__)
    logger.info("📊 Creating visualization plots...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Feature distributions by class
    plt.figure(figsize=(15, 10))
    features = ['avgKeystrokeInterval', 'mouseVelocity', 'clickFrequency', 
                'scrollPattern', 'navigationFlow', 'sessionDuration']
    
    for i, feature in enumerate(features, 1):
        plt.subplot(2, 3, i)
        
        legitimate = test_data[test_data['label'] == 1][feature]
        attack = test_data[test_data['label'] == 0][feature]
        
        plt.hist(legitimate, alpha=0.7, label='Legitimate', bins=20, color='green')
        plt.hist(attack, alpha=0.7, label='Attack', bins=20, color='red')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.legend()
        plt.title(f'Distribution of {feature}')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"📈 Plots saved to {save_dir}/")

def main():
    parser = argparse.ArgumentParser(description='Train Behavioral Authentication Models')
    parser.add_argument('--data', type=str, help='Path to CSV data file (optional)')
    parser.add_argument('--epochs', type=int, default=150, help='Training epochs')
    parser.add_argument('--users', type=int, default=25, help='Number of synthetic users')
    parser.add_argument('--samples', type=int, default=300, help='Samples per user')
    parser.add_argument('--model-dir', type=str, default='model', help='Model save directory')
    parser.add_argument('--log-level', type=str, default='INFO', help='Logging level')
    parser.add_argument('--evaluate', action='store_true', help='Run evaluation after training')
    parser.add_argument('--plots', action='store_true', help='Generate visualization plots')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    logger.info("🚀 Starting Behavioral Authentication Model Training")
    
    # Initialize model
    behavior_model = UserAdaptiveBehaviorModel(model_dir=args.model_dir)
    
    # Load or generate data
    if args.data:
        df = load_real_data(args.data)
        if df is None:
            logger.error("❌ Failed to load data, exiting")
            return
    else:
        logger.info("📝 No data file provided, generating realistic synthetic data...")
        df = create_synthetic_realistic_data(
            n_users=args.users, 
            samples_per_user=args.samples,
            attack_ratio=0.2
        )
    
    # Preprocess data
    X, y, user_ids = behavior_model.preprocess_data(df)
    logger.info(f"📊 Training data shape: {X.shape}")
    
    # Split data for evaluation
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    
    # Train global model
    logger.info("🧠 Training global models...")
    X_train, y_train, _ = behavior_model.preprocess_data(train_df)
    behavior_model.train_global_model(X_train, y_train, epochs=args.epochs)
    
    # Create user models for some users
    logger.info("👥 Creating personalized user models...")
    unique_users = df['user_id'].unique()[:10]  # First 10 users
    
    for user_id in unique_users:
        user_data = train_df[train_df['user_id'] == user_id]
        if len(user_data) >= 10:  # Need minimum samples
            initial_samples = user_data.head(20).to_dict('records')
            behavior_model.create_user_model(str(user_id), initial_samples)
            logger.info(f"✅ Created model for user {user_id}")
    
    # Save models
    behavior_model.save_models()
    logger.info("💾 Models saved successfully")
    
    # Evaluation
    if args.evaluate:
        metrics = evaluate_model(behavior_model, test_df)
        
        # Save metrics
        with open(f'{args.model_dir}/evaluation_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info("✅ Evaluation completed")
    
    # Generate plots
    if args.plots:
        plot_training_results(behavior_model, test_df)
        logger.info("📊 Plots generated")
    
    # Training summary
    logger.info("🎉 Training completed successfully!")
    logger.info(f"📈 Summary:")
    logger.info(f"   - Total samples: {len(df)}")
    logger.info(f"   - Unique users: {len(df['user_id'].unique())}")
    logger.info(f"   - Training epochs: {args.epochs}")
    logger.info(f"   - Personalized models: {len(behavior_model.user_models)}")
    logger.info(f"   - Models saved to: {args.model_dir}")

if __name__ == "__main__":
    main()