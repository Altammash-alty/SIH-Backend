"""
ML Model Manager for loading, training, and managing all models
"""
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from pathlib import Path
import asyncio
from datetime import datetime
import logging

from app.ml.outbreak_classifier import OutbreakClassifier
from app.ml.anomaly_detector import AnomalyDetector
from app.ml.hotspot_clusterer import HotspotClusterer
from app.ml.time_series_forecaster import TimeSeriesForecaster
from app.ml.ensemble_predictor import EnsemblePredictor
from app.core.config import Config

logger = logging.getLogger(__name__)

class ModelManager:
    """Manages all ML models in the system"""
    
    def __init__(self):
        self.models = {}
        self.model_path = Path(Config.MODEL_PATH)
        self.model_path.mkdir(exist_ok=True)
        
        # Initialize model instances
        self.outbreak_classifier = OutbreakClassifier()
        self.anomaly_detector = AnomalyDetector()
        self.hotspot_clusterer = HotspotClusterer()
        self.time_series_forecaster = TimeSeriesForecaster()
        self.ensemble_predictor = EnsemblePredictor()
    
    async def load_all_models(self) -> Dict[str, Any]:
        """Load all trained models"""
        try:
            # Load individual models
            self.models['outbreak_classifier'] = await self._load_model('outbreak_classifier')
            self.models['anomaly_detector'] = await self._load_model('anomaly_detector')
            self.models['hotspot_clusterer'] = await self._load_model('hotspot_clusterer')
            self.models['time_series_forecaster'] = await self._load_model('time_series_forecaster')
            self.models['ensemble_predictor'] = await self._load_model('ensemble_predictor')
            
            logger.info("All models loaded successfully")
            return self.models
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            # Return empty models if loading fails
            return {}
    
    async def train_all_models(self, training_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Train all models with provided data"""
        results = {}
        
        try:
            # Train outbreak classifier
            if 'health_reports' in training_data and 'water_quality' in training_data:
                logger.info("Training outbreak classifier...")
                outbreak_result = await self.outbreak_classifier.train(
                    health_data=training_data['health_reports'],
                    water_data=training_data['water_quality']
                )
                results['outbreak_classifier'] = outbreak_result
                await self._save_model('outbreak_classifier', self.outbreak_classifier)
            
            # Train anomaly detector
            if 'water_quality' in training_data:
                logger.info("Training anomaly detector...")
                anomaly_result = await self.anomaly_detector.train(training_data['water_quality'])
                results['anomaly_detector'] = anomaly_result
                await self._save_model('anomaly_detector', self.anomaly_detector)
            
            # Train hotspot clusterer
            if 'health_reports' in training_data:
                logger.info("Training hotspot clusterer...")
                cluster_result = await self.hotspot_clusterer.train(training_data['health_reports'])
                results['hotspot_clusterer'] = cluster_result
                await self._save_model('hotspot_clusterer', self.hotspot_clusterer)
            
            # Train time series forecaster
            if 'health_reports' in training_data:
                logger.info("Training time series forecaster...")
                forecast_result = await self.time_series_forecaster.train(training_data['health_reports'])
                results['time_series_forecaster'] = forecast_result
                await self._save_model('time_series_forecaster', self.time_series_forecaster)
            
            # Train ensemble predictor
            logger.info("Training ensemble predictor...")
            ensemble_result = await self.ensemble_predictor.train(self.models)
            results['ensemble_predictor'] = ensemble_result
            await self._save_model('ensemble_predictor', self.ensemble_predictor)
            
            logger.info("All models trained successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
            raise
    
    async def predict_outbreak(self, health_data: pd.DataFrame, water_data: pd.DataFrame) -> Dict[str, Any]:
        """Make outbreak prediction using ensemble approach"""
        try:
            # Get individual model predictions
            outbreak_pred = await self.outbreak_classifier.predict(health_data, water_data)
            anomaly_pred = await self.anomaly_detector.predict(water_data)
            cluster_pred = await self.hotspot_clusterer.predict(health_data)
            forecast_pred = await self.time_series_forecaster.predict(health_data)
            
            # Combine predictions using ensemble
            ensemble_pred = await self.ensemble_predictor.predict({
                'outbreak_classifier': outbreak_pred,
                'anomaly_detector': anomaly_pred,
                'hotspot_clusterer': cluster_pred,
                'time_series_forecaster': forecast_pred
            })
            
            return ensemble_pred
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise
    
    async def _load_model(self, model_name: str) -> Any:
        """Load a specific model from disk"""
        model_file = self.model_path / f"{model_name}.joblib"
        
        if model_file.exists():
            try:
                model = joblib.load(model_file)
                logger.info(f"Loaded {model_name} from disk")
                return model
            except Exception as e:
                logger.warning(f"Failed to load {model_name}: {e}")
                return None
        else:
            logger.warning(f"Model file {model_name} not found")
            return None
    
    async def _save_model(self, model_name: str, model: Any) -> None:
        """Save a model to disk"""
        model_file = self.model_path / f"{model_name}.joblib"
        
        try:
            joblib.dump(model, model_file)
            logger.info(f"Saved {model_name} to disk")
        except Exception as e:
            logger.error(f"Failed to save {model_name}: {e}")
            raise
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get performance metrics for all models"""
        performance = {}
        
        for model_name, model in self.models.items():
            if hasattr(model, 'get_performance_metrics'):
                performance[model_name] = model.get_performance_metrics()
            else:
                performance[model_name] = {"status": "no_metrics_available"}
        
        return performance
