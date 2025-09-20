"""
Outbreak Classification Models (Random Forest, XGBoost, AdaBoost)
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import joblib
import logging

logger = logging.getLogger(__name__)

class OutbreakClassifier:
    """Outbreak classification using multiple algorithms"""
    
    def __init__(self):
        self.random_forest = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.xgboost = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        self.adaboost = AdaBoostClassifier(
            n_estimators=100,
            learning_rate=1.0,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.feature_columns = []
        self.performance_metrics = {}
    
    async def train(self, health_data: pd.DataFrame, water_data: pd.DataFrame) -> Dict[str, Any]:
        """Train the outbreak classification models"""
        try:
            # Prepare training data
            X, y = self._prepare_training_data(health_data, water_data)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train models
            logger.info("Training Random Forest...")
            self.random_forest.fit(X_train_scaled, y_train)
            
            logger.info("Training XGBoost...")
            self.xgboost.fit(X_train_scaled, y_train)
            
            logger.info("Training AdaBoost...")
            self.adaboost.fit(X_train_scaled, y_train)
            
            # Evaluate models
            rf_pred = self.random_forest.predict(X_test_scaled)
            xgb_pred = self.xgboost.predict(X_test_scaled)
            ada_pred = self.adaboost.predict(X_test_scaled)
            
            # Calculate metrics
            self.performance_metrics = {
                'random_forest': self._calculate_metrics(y_test, rf_pred, 'Random Forest'),
                'xgboost': self._calculate_metrics(y_test, xgb_pred, 'XGBoost'),
                'adaboost': self._calculate_metrics(y_test, ada_pred, 'AdaBoost')
            }
            
            self.is_trained = True
            logger.info("Outbreak classification models trained successfully")
            
            return {
                'status': 'success',
                'performance_metrics': self.performance_metrics,
                'feature_importance': self._get_feature_importance()
            }
            
        except Exception as e:
            logger.error(f"Error training outbreak classifier: {e}")
            raise
    
    async def predict(self, health_data: pd.DataFrame, water_data: pd.DataFrame) -> Dict[str, Any]:
        """Make outbreak predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        try:
            # Prepare prediction data
            X = self._prepare_prediction_data(health_data, water_data)
            X_scaled = self.scaler.transform(X)
            
            # Get predictions from all models
            rf_pred = self.random_forest.predict(X_scaled)
            rf_proba = self.random_forest.predict_proba(X_scaled)
            
            xgb_pred = self.xgboost.predict(X_scaled)
            xgb_proba = self.xgboost.predict_proba(X_scaled)
            
            ada_pred = self.adaboost.predict(X_scaled)
            ada_proba = self.adaboost.predict_proba(X_scaled)
            
            # Ensemble prediction (majority vote)
            ensemble_pred = self._ensemble_predict(rf_pred, xgb_pred, ada_pred)
            ensemble_proba = self._ensemble_predict_proba(rf_proba, xgb_proba, ada_proba)
            
            return {
                'outbreak_prediction': ensemble_pred.tolist(),
                'outbreak_probability': ensemble_proba.tolist(),
                'individual_predictions': {
                    'random_forest': rf_pred.tolist(),
                    'xgboost': xgb_pred.tolist(),
                    'adaboost': ada_pred.tolist()
                },
                'confidence_scores': {
                    'random_forest': np.max(rf_proba, axis=1).tolist(),
                    'xgboost': np.max(xgb_proba, axis=1).tolist(),
                    'adaboost': np.max(ada_proba, axis=1).tolist()
                }
            }
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise
    
    def _prepare_training_data(self, health_data: pd.DataFrame, water_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data by combining health and water quality features"""
        # Create outbreak labels based on symptom patterns and water contamination
        outbreak_labels = []
        
        for _, health_row in health_data.iterrows():
            # Check if this is an outbreak case
            is_outbreak = self._is_outbreak_case(health_row, water_data)
            outbreak_labels.append(1 if is_outbreak else 0)
        
        # Prepare features
        features = self._extract_features(health_data, water_data)
        
        return features, np.array(outbreak_labels)
    
    def _prepare_prediction_data(self, health_data: pd.DataFrame, water_data: pd.DataFrame) -> np.ndarray:
        """Prepare prediction data"""
        return self._extract_features(health_data, water_data)
    
    def _extract_features(self, health_data: pd.DataFrame, water_data: pd.DataFrame) -> np.ndarray:
        """Extract features from health and water quality data"""
        features = []
        
        for _, health_row in health_data.iterrows():
            # Health features
            health_features = [
                float(health_row.get('fever', False)),
                float(health_row.get('diarrhea', False)),
                float(health_row.get('vomiting', False)),
                float(health_row.get('nausea', False)),
                float(health_row.get('abdominal_pain', False)),
                float(health_row.get('dehydration', False)),
                float(health_row.get('symptom_severity', 0)) if pd.notna(health_row.get('symptom_severity')) else 0.0
            ]
            
            # Water quality features (find nearest water quality data)
            water_features = self._get_nearest_water_features(
                health_row['location_lat'], 
                health_row['location_lon'], 
                water_data
            )
            
            # Combine features
            combined_features = health_features + water_features
            features.append(combined_features)
        
        self.feature_columns = [f"health_feature_{i}" for i in range(len(health_features))] + \
                              [f"water_feature_{i}" for i in range(len(water_features))]
        
        return np.array(features)
    
    def _get_nearest_water_features(self, lat: float, lon: float, water_data: pd.DataFrame) -> List[float]:
        """Get water quality features from nearest sensor"""
        if water_data.empty:
            return [0.0] * 10  # Default values
        
        # Calculate distances (simplified)
        distances = np.sqrt(
            (water_data['location_lat'] - lat)**2 + 
            (water_data['location_lon'] - lon)**2
        )
        
        nearest_idx = distances.idxmin()
        nearest_water = water_data.loc[nearest_idx]
        
        return [
            float(nearest_water.get('turbidity', 0)) if pd.notna(nearest_water.get('turbidity')) else 0.0,
            float(nearest_water.get('ph_level', 7)) if pd.notna(nearest_water.get('ph_level')) else 7.0,
            float(nearest_water.get('temperature', 25)) if pd.notna(nearest_water.get('temperature')) else 25.0,
            float(nearest_water.get('dissolved_oxygen', 0)) if pd.notna(nearest_water.get('dissolved_oxygen')) else 0.0,
            float(nearest_water.get('bacterial_count', 0)) if pd.notna(nearest_water.get('bacterial_count')) else 0.0,
            float(nearest_water.get('chlorine_residual', 0)) if pd.notna(nearest_water.get('chlorine_residual')) else 0.0,
            float(nearest_water.get('conductivity', 0)) if pd.notna(nearest_water.get('conductivity')) else 0.0,
            float(nearest_water.get('total_dissolved_solids', 0)) if pd.notna(nearest_water.get('total_dissolved_solids')) else 0.0,
            float(nearest_water.get('nitrate_level', 0)) if pd.notna(nearest_water.get('nitrate_level')) else 0.0,
            float(nearest_water.get('phosphate_level', 0)) if pd.notna(nearest_water.get('phosphate_level')) else 0.0
        ]
    
    def _is_outbreak_case(self, health_row: pd.Series, water_data: pd.DataFrame) -> bool:
        """Determine if a health case is part of an outbreak"""
        # Simple heuristic: multiple severe symptoms + contaminated water nearby
        severe_symptoms = sum([
            health_row.get('fever', False),
            health_row.get('diarrhea', False),
            health_row.get('vomiting', False)
        ])
        
        # Check for contaminated water nearby
        nearby_contaminated = self._check_nearby_contamination(
            health_row['location_lat'], 
            health_row['location_lon'], 
            water_data
        )
        
        return severe_symptoms >= 2 and nearby_contaminated
    
    def _check_nearby_contamination(self, lat: float, lon: float, water_data: pd.DataFrame) -> bool:
        """Check if there's contaminated water nearby"""
        if water_data.empty:
            return False
        
        # Find water data within 5km radius (simplified)
        nearby_water = water_data[
            (abs(water_data['location_lat'] - lat) < 0.05) & 
            (abs(water_data['location_lon'] - lon) < 0.05)
        ]
        
        if nearby_water.empty:
            return False
        
        # Check if any nearby water is contaminated
        return nearby_water['is_contaminated'].any()
    
    def _ensemble_predict(self, rf_pred: np.ndarray, xgb_pred: np.ndarray, ada_pred: np.ndarray) -> np.ndarray:
        """Combine predictions using majority vote"""
        predictions = np.column_stack([rf_pred, xgb_pred, ada_pred])
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=predictions)
    
    def _ensemble_predict_proba(self, rf_proba: np.ndarray, xgb_proba: np.ndarray, ada_proba: np.ndarray) -> np.ndarray:
        """Combine prediction probabilities using average"""
        return (rf_proba + xgb_proba + ada_proba) / 3
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> Dict[str, float]:
        """Calculate performance metrics"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted'),
            'auc': roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.0
        }
    
    def _get_feature_importance(self) -> Dict[str, Any]:
        """Get feature importance from trained models"""
        if not self.is_trained:
            return {}
        
        return {
            'random_forest': dict(zip(self.feature_columns, self.random_forest.feature_importances_)),
            'xgboost': dict(zip(self.feature_columns, self.xgboost.feature_importances_)),
            'adaboost': dict(zip(self.feature_columns, self.adaboost.feature_importances_))
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return self.performance_metrics
