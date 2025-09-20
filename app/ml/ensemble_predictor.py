"""
Ensemble Predictor combining all models
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class EnsemblePredictor:
    """Ensemble predictor combining all ML models"""
    
    def __init__(self):
        self.is_trained = False
        self.weights = {
            'outbreak_classifier': 0.3,
            'anomaly_detector': 0.2,
            'hotspot_clusterer': 0.2,
            'time_series_forecaster': 0.3
        }
        self.performance_metrics = {}
    
    async def train(self, models: Dict[str, Any]) -> Dict[str, Any]:
        """Train ensemble predictor"""
        try:
            # Validate that all required models are available
            required_models = ['outbreak_classifier', 'anomaly_detector', 'hotspot_clusterer', 'time_series_forecaster']
            available_models = [name for name in required_models if name in models and models[name] is not None]
            
            if len(available_models) < 2:
                logger.warning("Insufficient models for ensemble training")
                return {'status': 'insufficient_models'}
            
            # Adjust weights based on available models
            self._adjust_weights(available_models)
            
            self.is_trained = True
            logger.info("Ensemble predictor trained successfully")
            
            return {
                'status': 'success',
                'available_models': available_models,
                'weights': self.weights
            }
            
        except Exception as e:
            logger.error(f"Error training ensemble predictor: {e}")
            raise
    
    async def predict(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Make ensemble prediction"""
        if not self.is_trained:
            raise ValueError("Ensemble predictor must be trained before making predictions")
        
        try:
            # Extract individual model predictions
            outbreak_pred = predictions.get('outbreak_classifier', {})
            anomaly_pred = predictions.get('anomaly_detector', {})
            cluster_pred = predictions.get('hotspot_clusterer', {})
            forecast_pred = predictions.get('time_series_forecaster', {})
            
            # Calculate ensemble risk score
            risk_score = self._calculate_ensemble_risk_score(
                outbreak_pred, anomaly_pred, cluster_pred, forecast_pred
            )
            
            # Determine outbreak probability
            outbreak_probability = self._calculate_outbreak_probability(risk_score)
            
            # Identify likely disease
            predicted_disease = self._predict_disease_type(outbreak_pred, anomaly_pred)
            
            # Generate hotspot information
            hotspot_info = self._extract_hotspot_info(cluster_pred)
            
            # Generate time-based insights
            time_insights = self._extract_time_insights(forecast_pred)
            
            # Create final prediction
            ensemble_prediction = {
                'outbreak_probability': outbreak_probability,
                'risk_score': risk_score,
                'predicted_disease': predicted_disease,
                'hotspot_info': hotspot_info,
                'time_insights': time_insights,
                'confidence_level': self._calculate_confidence_level(predictions),
                'contributing_factors': self._identify_contributing_factors(
                    outbreak_pred, anomaly_pred, cluster_pred, forecast_pred
                ),
                'recommendations': self._generate_recommendations(
                    outbreak_probability, predicted_disease, hotspot_info
                )
            }
            
            return ensemble_prediction
            
        except Exception as e:
            logger.error(f"Error making ensemble prediction: {e}")
            raise
    
    def _adjust_weights(self, available_models: List[str]) -> None:
        """Adjust weights based on available models"""
        if not available_models:
            return
        
        # Normalize weights for available models
        total_weight = sum(self.weights[model] for model in available_models)
        
        for model in available_models:
            self.weights[model] = self.weights[model] / total_weight
        
        # Set unavailable models to 0
        for model in self.weights:
            if model not in available_models:
                self.weights[model] = 0.0
    
    def _calculate_ensemble_risk_score(self, outbreak_pred: Dict, anomaly_pred: Dict, 
                                     cluster_pred: Dict, forecast_pred: Dict) -> float:
        """Calculate ensemble risk score"""
        risk_components = []
        
        # Outbreak classifier contribution
        if 'outbreak_probability' in outbreak_pred:
            outbreak_risk = np.mean(outbreak_pred['outbreak_probability'])
            risk_components.append(outbreak_risk * self.weights['outbreak_classifier'])
        
        # Anomaly detector contribution
        if 'anomaly_rate' in anomaly_pred:
            anomaly_risk = anomaly_pred['anomaly_rate']
            risk_components.append(anomaly_risk * self.weights['anomaly_detector'])
        
        # Hotspot clusterer contribution
        if 'kmeans_clusters' in cluster_pred:
            cluster_risk = self._calculate_cluster_risk(cluster_pred['kmeans_clusters'])
            risk_components.append(cluster_risk * self.weights['hotspot_clusterer'])
        
        # Time series forecaster contribution
        if 'combined_forecast' in forecast_pred and 'predictions' in forecast_pred['combined_forecast']:
            forecast_risk = self._calculate_forecast_risk(forecast_pred['combined_forecast']['predictions'])
            risk_components.append(forecast_risk * self.weights['time_series_forecaster'])
        
        # Calculate weighted average
        if risk_components:
            return float(np.mean(risk_components))
        else:
            return 0.0
    
    def _calculate_outbreak_probability(self, risk_score: float) -> float:
        """Convert risk score to outbreak probability"""
        # Use sigmoid function to map risk score to probability
        return float(1 / (1 + np.exp(-10 * (risk_score - 0.5))))
    
    def _predict_disease_type(self, outbreak_pred: Dict, anomaly_pred: Dict) -> str:
        """Predict the most likely disease type"""
        # Simple heuristic based on symptoms and water contamination
        if 'outbreak_prediction' in outbreak_pred:
            outbreak_cases = np.sum(outbreak_pred['outbreak_prediction'])
        else:
            outbreak_cases = 0
        
        if 'anomaly_rate' in anomaly_pred:
            contamination_level = anomaly_pred['anomaly_rate']
        else:
            contamination_level = 0
        
        # Disease prediction logic
        if contamination_level > 0.7 and outbreak_cases > 0:
            return "Cholera"
        elif contamination_level > 0.5 and outbreak_cases > 0:
            return "Typhoid"
        elif outbreak_cases > 0:
            return "Gastroenteritis"
        else:
            return "No outbreak predicted"
    
    def _extract_hotspot_info(self, cluster_pred: Dict) -> Dict[str, Any]:
        """Extract hotspot information from clustering results"""
        if 'kmeans_clusters' not in cluster_pred:
            return {'hotspots': [], 'total_hotspots': 0}
        
        clusters = cluster_pred['kmeans_clusters']
        hotspots = []
        
        for cluster in clusters:
            if cluster['case_count'] > 0:  # Only include clusters with cases
                hotspots.append({
                    'center_lat': cluster['center_lat'],
                    'center_lon': cluster['center_lon'],
                    'radius_km': cluster['radius_km'],
                    'case_count': cluster['case_count'],
                    'risk_level': cluster['risk_level'],
                    'density': cluster['density']
                })
        
        return {
            'hotspots': hotspots,
            'total_hotspots': len(hotspots),
            'highest_risk_hotspot': max(hotspots, key=lambda x: x['case_count']) if hotspots else None
        }
    
    def _extract_time_insights(self, forecast_pred: Dict) -> Dict[str, Any]:
        """Extract time-based insights from forecasting results"""
        if 'combined_forecast' not in forecast_pred:
            return {'trend': 'unknown', 'forecast_days': 0}
        
        forecast = forecast_pred['combined_forecast']
        
        if 'predictions' not in forecast:
            return {'trend': 'unknown', 'forecast_days': 0}
        
        predictions = forecast['predictions']
        
        # Calculate trend
        if len(predictions) >= 2:
            trend = 'increasing' if predictions[-1] > predictions[0] else 'decreasing'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'forecast_days': len(predictions),
            'next_week_prediction': predictions[:7] if len(predictions) >= 7 else predictions,
            'peak_prediction': max(predictions) if predictions else 0
        }
    
    def _calculate_confidence_level(self, predictions: Dict[str, Any]) -> str:
        """Calculate confidence level based on model agreement"""
        available_predictions = [pred for pred in predictions.values() if pred and 'error' not in pred]
        
        if len(available_predictions) >= 3:
            return "high"
        elif len(available_predictions) >= 2:
            return "medium"
        else:
            return "low"
    
    def _identify_contributing_factors(self, outbreak_pred: Dict, anomaly_pred: Dict, 
                                     cluster_pred: Dict, forecast_pred: Dict) -> List[str]:
        """Identify factors contributing to the prediction"""
        factors = []
        
        if 'outbreak_probability' in outbreak_pred and np.mean(outbreak_pred['outbreak_probability']) > 0.5:
            factors.append("High outbreak probability from classification models")
        
        if 'anomaly_rate' in anomaly_pred and anomaly_pred['anomaly_rate'] > 0.3:
            factors.append("Anomalous water quality patterns detected")
        
        if 'kmeans_clusters' in cluster_pred:
            high_risk_clusters = [c for c in cluster_pred['kmeans_clusters'] if c['risk_level'] in ['high', 'critical']]
            if high_risk_clusters:
                factors.append(f"High-risk disease clusters identified ({len(high_risk_clusters)} hotspots)")
        
        if 'combined_forecast' in forecast_pred and 'predictions' in forecast_pred['combined_forecast']:
            predictions = forecast_pred['combined_forecast']['predictions']
            if predictions and max(predictions) > np.mean(predictions) * 1.5:
                factors.append("Rising disease trend predicted")
        
        return factors
    
    def _generate_recommendations(self, outbreak_probability: float, predicted_disease: str, 
                                hotspot_info: Dict) -> List[str]:
        """Generate recommendations based on prediction"""
        recommendations = []
        
        if outbreak_probability > 0.8:
            recommendations.append("URGENT: High outbreak probability - activate emergency response")
            recommendations.append("Deploy mobile health teams to affected areas")
            recommendations.append("Issue public health advisory immediately")
        elif outbreak_probability > 0.6:
            recommendations.append("MODERATE: Elevated outbreak risk - increase surveillance")
            recommendations.append("Prepare emergency response resources")
        elif outbreak_probability > 0.4:
            recommendations.append("LOW: Monitor situation closely")
        
        if predicted_disease != "No outbreak predicted":
            recommendations.append(f"Focus on {predicted_disease} prevention measures")
            recommendations.append("Test water sources for contamination")
        
        if hotspot_info['total_hotspots'] > 0:
            recommendations.append(f"Investigate {hotspot_info['total_hotspots']} identified hotspots")
            recommendations.append("Implement targeted interventions in high-risk areas")
        
        return recommendations
    
    def _calculate_cluster_risk(self, clusters: List[Dict]) -> float:
        """Calculate overall risk from clustering results"""
        if not clusters:
            return 0.0
        
        # Weight by case count and risk level
        risk_scores = []
        for cluster in clusters:
            risk_level_weights = {'low': 0.2, 'medium': 0.5, 'high': 0.8, 'critical': 1.0}
            risk_weight = risk_level_weights.get(cluster['risk_level'], 0.2)
            risk_scores.append(cluster['case_count'] * risk_weight)
        
        return float(np.mean(risk_scores)) if risk_scores else 0.0
    
    def _calculate_forecast_risk(self, predictions: List[float]) -> float:
        """Calculate risk from time series forecast"""
        if not predictions:
            return 0.0
        
        # Risk based on trend and magnitude
        if len(predictions) >= 2:
            trend_risk = (predictions[-1] - predictions[0]) / max(predictions[0], 1)
        else:
            trend_risk = 0
        
        magnitude_risk = np.mean(predictions) / 10  # Normalize by expected daily cases
        
        return float((trend_risk + magnitude_risk) / 2)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return self.performance_metrics
