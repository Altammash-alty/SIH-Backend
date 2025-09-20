"""
Hotspot Clustering using KMeans and DBSCAN
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import logging

logger = logging.getLogger(__name__)

class HotspotClusterer:
    """Geospatial clustering for disease hotspots"""
    
    def __init__(self):
        self.kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        self.dbscan = DBSCAN(eps=0.01, min_samples=3)  # eps in degrees (roughly 1km)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.performance_metrics = {}
        self.optimal_clusters = 5
    
    async def train(self, health_data: pd.DataFrame) -> Dict[str, Any]:
        """Train the clustering models"""
        try:
            # Prepare location features
            X = self._extract_location_features(health_data)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Find optimal number of clusters using elbow method
            optimal_k = self._find_optimal_clusters(X_scaled)
            self.optimal_clusters = optimal_k
            
            # Train KMeans with optimal clusters
            self.kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            kmeans_labels = self.kmeans.fit_predict(X_scaled)
            
            # Train DBSCAN
            dbscan_labels = self.dbscan.fit_predict(X_scaled)
            
            # Calculate performance metrics
            self.performance_metrics = {
                'kmeans_silhouette': silhouette_score(X_scaled, kmeans_labels) if len(set(kmeans_labels)) > 1 else 0,
                'kmeans_calinski_harabasz': calinski_harabasz_score(X_scaled, kmeans_labels) if len(set(kmeans_labels)) > 1 else 0,
                'dbscan_silhouette': silhouette_score(X_scaled, dbscan_labels) if len(set(dbscan_labels)) > 1 else 0,
                'dbscan_calinski_harabasz': calinski_harabasz_score(X_scaled, dbscan_labels) if len(set(dbscan_labels)) > 1 else 0,
                'optimal_clusters': optimal_k,
                'kmeans_clusters': len(set(kmeans_labels)),
                'dbscan_clusters': len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0),
                'noise_points': np.sum(dbscan_labels == -1)
            }
            
            self.is_trained = True
            logger.info("Hotspot clustering models trained successfully")
            
            return {
                'status': 'success',
                'performance_metrics': self.performance_metrics,
                'cluster_centers': self.kmeans.cluster_centers_.tolist()
            }
            
        except Exception as e:
            logger.error(f"Error training hotspot clusterer: {e}")
            raise
    
    async def predict(self, health_data: pd.DataFrame) -> Dict[str, Any]:
        """Cluster health data into hotspots"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        try:
            # Prepare location features
            X = self._extract_location_features(health_data)
            X_scaled = self.scaler.transform(X)
            
            # Get cluster assignments
            kmeans_labels = self.kmeans.predict(X_scaled)
            dbscan_labels = self.dbscan.fit_predict(X_scaled)
            
            # Calculate cluster properties
            kmeans_clusters = self._analyze_clusters(health_data, kmeans_labels, 'kmeans')
            dbscan_clusters = self._analyze_clusters(health_data, dbscan_labels, 'dbscan')
            
            return {
                'kmeans_clusters': kmeans_clusters,
                'dbscan_clusters': dbscan_clusters,
                'kmeans_labels': kmeans_labels.tolist(),
                'dbscan_labels': dbscan_labels.tolist()
            }
            
        except Exception as e:
            logger.error(f"Error clustering data: {e}")
            raise
    
    def _extract_location_features(self, health_data: pd.DataFrame) -> np.ndarray:
        """Extract location-based features"""
        features = []
        
        for _, row in health_data.iterrows():
            # Basic location features
            location_features = [
                float(row.get('location_lat', 0)),
                float(row.get('location_lon', 0))
            ]
            
            # Add symptom density features
            symptom_count = sum([
                row.get('fever', False),
                row.get('diarrhea', False),
                row.get('vomiting', False),
                row.get('nausea', False),
                row.get('abdominal_pain', False),
                row.get('dehydration', False)
            ])
            
            location_features.append(float(symptom_count))
            features.append(location_features)
        
        return np.array(features)
    
    def _find_optimal_clusters(self, X_scaled: np.ndarray) -> int:
        """Find optimal number of clusters using elbow method"""
        if len(X_scaled) < 2:
            return 1
        
        max_clusters = min(10, len(X_scaled) // 2)
        if max_clusters < 2:
            return 2
        
        inertias = []
        K_range = range(2, max_clusters + 1)
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)
        
        # Simple elbow detection (find the point with maximum curvature)
        if len(inertias) < 3:
            return 2
        
        # Calculate second derivative
        second_derivatives = []
        for i in range(1, len(inertias) - 1):
            second_deriv = inertias[i-1] - 2*inertias[i] + inertias[i+1]
            second_derivatives.append(second_deriv)
        
        # Find the elbow point
        if second_derivatives:
            elbow_idx = np.argmax(second_derivatives)
            optimal_k = K_range[elbow_idx + 1]
        else:
            optimal_k = 2
        
        return max(2, min(optimal_k, max_clusters))
    
    def _analyze_clusters(self, health_data: pd.DataFrame, labels: np.ndarray, method: str) -> List[Dict[str, Any]]:
        """Analyze cluster properties"""
        clusters = []
        unique_labels = set(labels)
        
        # Remove noise label (-1) for DBSCAN
        if -1 in unique_labels:
            unique_labels.remove(-1)
        
        for cluster_id in unique_labels:
            cluster_mask = labels == cluster_id
            cluster_data = health_data[cluster_mask]
            
            if len(cluster_data) == 0:
                continue
            
            # Calculate cluster center
            center_lat = cluster_data['location_lat'].mean()
            center_lon = cluster_data['location_lon'].mean()
            
            # Calculate cluster radius (max distance from center)
            distances = np.sqrt(
                (cluster_data['location_lat'] - center_lat)**2 + 
                (cluster_data['location_lon'] - center_lon)**2
            )
            radius_km = distances.max() * 111  # Rough conversion to km
            
            # Calculate case density
            case_count = len(cluster_data)
            area_km2 = np.pi * (radius_km ** 2) if radius_km > 0 else 1
            density = case_count / area_km2
            
            # Calculate risk score based on symptoms
            risk_score = self._calculate_cluster_risk_score(cluster_data)
            
            clusters.append({
                'cluster_id': f"{method}_{cluster_id}",
                'center_lat': float(center_lat),
                'center_lon': float(center_lon),
                'radius_km': float(radius_km),
                'case_count': int(case_count),
                'density': float(density),
                'risk_score': float(risk_score),
                'risk_level': self._get_risk_level(risk_score)
            })
        
        return clusters
    
    def _calculate_cluster_risk_score(self, cluster_data: pd.DataFrame) -> float:
        """Calculate risk score for a cluster"""
        if cluster_data.empty:
            return 0.0
        
        # Weight different symptoms
        symptom_weights = {
            'fever': 0.2,
            'diarrhea': 0.3,
            'vomiting': 0.25,
            'nausea': 0.1,
            'abdominal_pain': 0.1,
            'dehydration': 0.05
        }
        
        risk_score = 0.0
        for symptom, weight in symptom_weights.items():
            if symptom in cluster_data.columns:
                symptom_rate = cluster_data[symptom].mean()
                risk_score += symptom_rate * weight
        
        # Normalize to 0-1 range
        return min(1.0, risk_score)
    
    def _get_risk_level(self, risk_score: float) -> str:
        """Convert risk score to risk level"""
        if risk_score >= 0.8:
            return "critical"
        elif risk_score >= 0.6:
            return "high"
        elif risk_score >= 0.4:
            return "medium"
        else:
            return "low"
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return self.performance_metrics
