"""
Anomaly Detection using Isolation Forest
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import logging

logger = logging.getLogger(__name__)

class AnomalyDetector:
    """Anomaly detection for water quality and health data"""
    
    def __init__(self):
        self.isolation_forest = IsolationForest(
            contamination=0.1,  # Expected proportion of outliers
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_columns = []
        self.performance_metrics = {}
    
    async def train(self, water_data: pd.DataFrame) -> Dict[str, Any]:
        """Train the anomaly detection model"""
        try:
            # Prepare features
            X = self._extract_water_features(water_data)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train isolation forest
            logger.info("Training Isolation Forest for anomaly detection...")
            self.isolation_forest.fit(X_scaled)
            
            # Evaluate on training data
            predictions = self.isolation_forest.predict(X_scaled)
            anomaly_scores = self.isolation_forest.decision_function(X_scaled)
            
            # Calculate performance metrics
            self.performance_metrics = {
                'anomaly_count': np.sum(predictions == -1),
                'anomaly_rate': np.mean(predictions == -1),
                'mean_anomaly_score': np.mean(anomaly_scores),
                'std_anomaly_score': np.std(anomaly_scores)
            }
            
            self.is_trained = True
            logger.info("Anomaly detection model trained successfully")
            
            return {
                'status': 'success',
                'performance_metrics': self.performance_metrics,
                'anomalies_detected': int(self.performance_metrics['anomaly_count'])
            }
            
        except Exception as e:
            logger.error(f"Error training anomaly detector: {e}")
            raise
    
    async def predict(self, water_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect anomalies in water quality data"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        try:
            # Prepare features
            X = self._extract_water_features(water_data)
            X_scaled = self.scaler.transform(X)
            
            # Make predictions
            predictions = self.isolation_forest.predict(X_scaled)
            anomaly_scores = self.isolation_forest.decision_function(X_scaled)
            
            # Convert to boolean (True = anomaly)
            is_anomaly = predictions == -1
            
            return {
                'is_anomaly': is_anomaly.tolist(),
                'anomaly_scores': anomaly_scores.tolist(),
                'anomaly_count': int(np.sum(is_anomaly)),
                'anomaly_rate': float(np.mean(is_anomaly))
            }
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            raise
    
    def _extract_water_features(self, water_data: pd.DataFrame) -> np.ndarray:
        """Extract features from water quality data"""
        features = []
        
        for _, row in water_data.iterrows():
            water_features = [
                float(row.get('turbidity', 0)) if pd.notna(row.get('turbidity')) else 0.0,
                float(row.get('ph_level', 7)) if pd.notna(row.get('ph_level')) else 7.0,
                float(row.get('temperature', 25)) if pd.notna(row.get('temperature')) else 25.0,
                float(row.get('dissolved_oxygen', 0)) if pd.notna(row.get('dissolved_oxygen')) else 0.0,
                float(row.get('bacterial_count', 0)) if pd.notna(row.get('bacterial_count')) else 0.0,
                float(row.get('chlorine_residual', 0)) if pd.notna(row.get('chlorine_residual')) else 0.0,
                float(row.get('conductivity', 0)) if pd.notna(row.get('conductivity')) else 0.0,
                float(row.get('total_dissolved_solids', 0)) if pd.notna(row.get('total_dissolved_solids')) else 0.0,
                float(row.get('nitrate_level', 0)) if pd.notna(row.get('nitrate_level')) else 0.0,
                float(row.get('phosphate_level', 0)) if pd.notna(row.get('phosphate_level')) else 0.0
            ]
            features.append(water_features)
        
        self.feature_columns = [
            'turbidity', 'ph_level', 'temperature', 'dissolved_oxygen',
            'bacterial_count', 'chlorine_residual', 'conductivity',
            'total_dissolved_solids', 'nitrate_level', 'phosphate_level'
        ]
        
        return np.array(features)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return self.performance_metrics
