#!/usr/bin/env python3
"""
Train ML models with the demo data we just created
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import logging
from datetime import datetime
import asyncio

# Add the app directory to the Python path
sys.path.append(str(Path(__file__).parent))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleMLTrainer:
    """Simple ML trainer for demonstration"""
    
    def __init__(self):
        self.models = {}
        self.performance_metrics = {}
    
    def load_demo_data(self):
        """Load the demo datasets"""
        logger.info("Loading demo datasets...")
        
        try:
            # Load datasets
            self.health_df = pd.read_csv('demo_data/health_reports_dataset.csv')
            self.water_df = pd.read_csv('demo_data/water_quality_dataset.csv')
            self.train_health = pd.read_csv('demo_data/train_health_dataset.csv')
            self.test_health = pd.read_csv('demo_data/test_health_dataset.csv')
            self.train_water = pd.read_csv('demo_data/train_water_dataset.csv')
            self.test_water = pd.read_csv('demo_data/test_water_dataset.csv')
            
            logger.info(f"✓ Loaded datasets:")
            logger.info(f"  - Health reports: {len(self.health_df)} total, {len(self.train_health)} train, {len(self.test_health)} test")
            logger.info(f"  - Water quality: {len(self.water_df)} total, {len(self.train_water)} train, {len(self.test_water)} test")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading demo data: {e}")
            return False
    
    def prepare_features(self):
        """Prepare features for ML training"""
        logger.info("Preparing features for ML training...")
        
        # Prepare health features
        health_features = []
        health_labels = []
        
        for _, row in self.train_health.iterrows():
            # Extract features
            features = [
                float(row['fever']),
                float(row['diarrhea']),
                float(row['vomiting']),
                float(row['nausea']),
                float(row['abdominal_pain']),
                float(row['dehydration']),
                float(row['symptom_severity']),
                float(row['location_lat']),
                float(row['location_lon'])
            ]
            
            # Create label (simple heuristic: outbreak if multiple severe symptoms)
            symptom_count = sum([
                row['fever'], row['diarrhea'], row['vomiting'], 
                row['nausea'], row['abdominal_pain'], row['dehydration']
            ])
            severity = row['symptom_severity']
            
            # Outbreak label: multiple symptoms + high severity
            is_outbreak = (symptom_count >= 3) or (severity >= 4)
            health_labels.append(1 if is_outbreak else 0)
            health_features.append(features)
        
        self.health_features = np.array(health_features)
        self.health_labels = np.array(health_labels)
        
        # Prepare water features
        water_features = []
        water_labels = []
        
        for _, row in self.train_water.iterrows():
            features = [
                float(row['turbidity']) if pd.notna(row['turbidity']) else 0.0,
                float(row['ph_level']) if pd.notna(row['ph_level']) else 7.0,
                float(row['temperature']) if pd.notna(row['temperature']) else 25.0,
                float(row['dissolved_oxygen']) if pd.notna(row['dissolved_oxygen']) else 8.0,
                float(row['bacterial_count']) if pd.notna(row['bacterial_count']) else 0.0,
                float(row['chlorine_residual']) if pd.notna(row['chlorine_residual']) else 0.0,
                float(row['conductivity']) if pd.notna(row['conductivity']) else 500.0,
                float(row['total_dissolved_solids']) if pd.notna(row['total_dissolved_solids']) else 250.0,
                float(row['nitrate_level']) if pd.notna(row['nitrate_level']) else 0.0,
                float(row['phosphate_level']) if pd.notna(row['phosphate_level']) else 0.0,
                float(row['latitude']),
                float(row['longitude'])
            ]
            
            # Contamination label
            is_contaminated = row['is_contaminated']
            water_labels.append(1 if is_contaminated else 0)
            water_features.append(features)
        
        self.water_features = np.array(water_features)
        self.water_labels = np.array(water_labels)
        
        logger.info(f"✓ Prepared features:")
        logger.info(f"  - Health features: {self.health_features.shape}")
        logger.info(f"  - Water features: {self.water_features.shape}")
        logger.info(f"  - Health labels: {np.sum(self.health_labels)}/{len(self.health_labels)} outbreaks")
        logger.info(f"  - Water labels: {np.sum(self.water_labels)}/{len(self.water_labels)} contaminated")
    
    def train_models(self):
        """Train ML models"""
        logger.info("Training ML models...")
        
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.svm import SVC
            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
            from sklearn.preprocessing import StandardScaler
            
            # Scale features
            scaler = StandardScaler()
            health_features_scaled = scaler.fit_transform(self.health_features)
            water_features_scaled = scaler.fit_transform(self.water_features)
            
            # Train health outbreak classifier
            logger.info("Training health outbreak classifier...")
            health_rf = RandomForestClassifier(n_estimators=100, random_state=42)
            health_rf.fit(health_features_scaled, self.health_labels)
            self.models['health_outbreak_classifier'] = health_rf
            
            # Train water contamination classifier
            logger.info("Training water contamination classifier...")
            water_rf = RandomForestClassifier(n_estimators=100, random_state=42)
            water_rf.fit(water_features_scaled, self.water_labels)
            self.models['water_contamination_classifier'] = water_rf
            
            # Train ensemble model (use only health data for simplicity)
            logger.info("Training ensemble model...")
            # For ensemble, we'll use health features and add water contamination as a feature
            water_contamination_feature = np.zeros(len(self.health_features))
            # Match water contamination to nearest health record (simplified)
            for i in range(len(self.health_features)):
                if i < len(self.water_labels):
                    water_contamination_feature[i] = self.water_labels[i]
                else:
                    water_contamination_feature[i] = 0  # Default to clean
            
            ensemble_features = np.column_stack([health_features_scaled, water_contamination_feature])
            ensemble_labels = self.health_labels  # Use health labels as primary
            ensemble_rf = RandomForestClassifier(n_estimators=100, random_state=42)
            ensemble_rf.fit(ensemble_features, ensemble_labels)
            self.models['ensemble_classifier'] = ensemble_rf
            
            logger.info("✓ All models trained successfully!")
            
            # Calculate performance metrics
            self._calculate_performance_metrics()
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
            raise
    
    def _calculate_performance_metrics(self):
        """Calculate model performance metrics"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        # Health classifier performance
        health_pred = self.models['health_outbreak_classifier'].predict(self.health_features)
        health_acc = accuracy_score(self.health_labels, health_pred)
        health_prec = precision_score(self.health_labels, health_pred, zero_division=0)
        health_rec = recall_score(self.health_labels, health_pred, zero_division=0)
        health_f1 = f1_score(self.health_labels, health_pred, zero_division=0)
        
        self.performance_metrics['health_classifier'] = {
            'accuracy': health_acc,
            'precision': health_prec,
            'recall': health_rec,
            'f1_score': health_f1
        }
        
        # Water classifier performance
        water_pred = self.models['water_contamination_classifier'].predict(self.water_features)
        water_acc = accuracy_score(self.water_labels, water_pred)
        water_prec = precision_score(self.water_labels, water_pred, zero_division=0)
        water_rec = recall_score(self.water_labels, water_pred, zero_division=0)
        water_f1 = f1_score(self.water_labels, water_pred, zero_division=0)
        
        self.performance_metrics['water_classifier'] = {
            'accuracy': water_acc,
            'precision': water_prec,
            'recall': water_rec,
            'f1_score': water_f1
        }
        
        logger.info("✓ Performance metrics calculated")
    
    def make_predictions(self):
        """Make predictions on test data"""
        logger.info("Making predictions on test data...")
        
        try:
            from sklearn.preprocessing import StandardScaler
            
            # Prepare test features
            test_health_features = []
            for _, row in self.test_health.iterrows():
                features = [
                    float(row['fever']), float(row['diarrhea']), float(row['vomiting']),
                    float(row['nausea']), float(row['abdominal_pain']), float(row['dehydration']),
                    float(row['symptom_severity']), float(row['location_lat']), float(row['location_lon'])
                ]
                test_health_features.append(features)
            
            test_water_features = []
            for _, row in self.test_water.iterrows():
                features = [
                    float(row['turbidity']) if pd.notna(row['turbidity']) else 0.0,
                    float(row['ph_level']) if pd.notna(row['ph_level']) else 7.0,
                    float(row['temperature']) if pd.notna(row['temperature']) else 25.0,
                    float(row['dissolved_oxygen']) if pd.notna(row['dissolved_oxygen']) else 8.0,
                    float(row['bacterial_count']) if pd.notna(row['bacterial_count']) else 0.0,
                    float(row['chlorine_residual']) if pd.notna(row['chlorine_residual']) else 0.0,
                    float(row['conductivity']) if pd.notna(row['conductivity']) else 500.0,
                    float(row['total_dissolved_solids']) if pd.notna(row['total_dissolved_solids']) else 250.0,
                    float(row['nitrate_level']) if pd.notna(row['nitrate_level']) else 0.0,
                    float(row['phosphate_level']) if pd.notna(row['phosphate_level']) else 0.0,
                    float(row['latitude']), float(row['longitude'])
                ]
                test_water_features.append(features)
            
            test_health_features = np.array(test_health_features)
            test_water_features = np.array(test_water_features)
            
            # Scale test features
            scaler = StandardScaler()
            test_health_scaled = scaler.fit_transform(test_health_features)
            test_water_scaled = scaler.fit_transform(test_water_features)
            
            # Make predictions
            health_predictions = self.models['health_outbreak_classifier'].predict(test_health_scaled)
            water_predictions = self.models['water_contamination_classifier'].predict(test_water_scaled)
            
            # Calculate outbreak probabilities
            health_probabilities = self.models['health_outbreak_classifier'].predict_proba(test_health_scaled)[:, 1]
            water_probabilities = self.models['water_contamination_classifier'].predict_proba(test_water_scaled)[:, 1]
            
            # Create prediction results
            predictions = []
            for i in range(len(test_health_features)):
                pred = {
                    'location': f"Test Location {i+1}",
                    'health_outbreak_probability': float(health_probabilities[i]),
                    'water_contamination_probability': float(water_predictions[i]),
                    'combined_risk_score': float((health_probabilities[i] + water_probabilities[i]) / 2),
                    'predicted_disease': self._predict_disease_type(health_probabilities[i], water_probabilities[i]),
                    'risk_level': self._get_risk_level(health_probabilities[i], water_probabilities[i])
                }
                predictions.append(pred)
            
            logger.info(f"✓ Generated {len(predictions)} predictions")
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return []
    
    def _predict_disease_type(self, health_prob, water_prob):
        """Predict disease type based on probabilities"""
        if health_prob > 0.7 and water_prob > 0.5:
            return "Cholera"
        elif health_prob > 0.5 and water_prob > 0.3:
            return "Typhoid"
        elif health_prob > 0.3:
            return "Gastroenteritis"
        else:
            return "No outbreak predicted"
    
    def _get_risk_level(self, health_prob, water_prob):
        """Get risk level based on probabilities"""
        combined_risk = (health_prob + water_prob) / 2
        
        if combined_risk > 0.7:
            return "High"
        elif combined_risk > 0.4:
            return "Medium"
        else:
            return "Low"
    
    def display_results(self, predictions):
        """Display training and prediction results"""
        print("\n" + "=" * 80)
        print("🎯 MACHINE LEARNING TRAINING RESULTS")
        print("=" * 80)
        
        # Display performance metrics
        print("\n📊 MODEL PERFORMANCE METRICS:")
        print("-" * 50)
        
        for model_name, metrics in self.performance_metrics.items():
            print(f"\n{model_name.upper()}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
        
        # Display predictions
        print(f"\n🔮 OUTBREAK PREDICTIONS:")
        print("-" * 80)
        print(f"{'Location':<20} {'Health Risk':<12} {'Water Risk':<12} {'Combined':<10} {'Disease':<20} {'Level':<8}")
        print("-" * 80)
        
        for pred in predictions:
            print(f"{pred['location']:<20} {pred['health_outbreak_probability']:<12.3f} {pred['water_contamination_probability']:<12.3f} {pred['combined_risk_score']:<10.3f} {pred['predicted_disease']:<20} {pred['risk_level']:<8}")
        
        # Summary statistics
        high_risk = len([p for p in predictions if p['risk_level'] == 'High'])
        medium_risk = len([p for p in predictions if p['risk_level'] == 'Medium'])
        low_risk = len([p for p in predictions if p['risk_level'] == 'Low'])
        
        print(f"\n📈 RISK LEVEL SUMMARY:")
        print(f"  High Risk: {high_risk} locations")
        print(f"  Medium Risk: {medium_risk} locations")
        print(f"  Low Risk: {low_risk} locations")
        
        # Dataset summary
        print(f"\n📋 DATASET SUMMARY:")
        print(f"  Training samples: {len(self.health_features)} health + {len(self.water_features)} water")
        print(f"  Test samples: {len(predictions)} predictions")
        print(f"  Features per health record: {self.health_features.shape[1]}")
        print(f"  Features per water record: {self.water_features.shape[1]}")
        
        print(f"\n✅ TRAINING COMPLETED SUCCESSFULLY!")
        print(f"The models are now ready to predict water-borne disease outbreaks!")

def main():
    """Main function"""
    print("🚀 SMART HEALTH SURVEILLANCE - ML TRAINING WITH DEMO DATA")
    print("=" * 80)
    
    # Initialize trainer
    trainer = SimpleMLTrainer()
    
    try:
        # Step 1: Load demo data
        if not trainer.load_demo_data():
            print("❌ Failed to load demo data. Please run demo_data_scraping.py first.")
            return
        
        # Step 2: Prepare features
        trainer.prepare_features()
        
        # Step 3: Train models
        trainer.train_models()
        
        # Step 4: Make predictions
        predictions = trainer.make_predictions()
        
        # Step 5: Display results
        trainer.display_results(predictions)
        
        print(f"\n🎉 SUCCESS! The system has been trained and is ready for predictions.")
        print(f"\nNext steps:")
        print(f"1. Start the API server: docker-compose up app")
        print(f"2. Test the API: python scripts/test_api.py")
        print(f"3. View API documentation: http://localhost:8000/docs")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
