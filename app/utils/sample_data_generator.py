"""
Sample data generator for testing and demonstration
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any
import random

class SampleDataGenerator:
    """Generate sample data for testing the health surveillance system"""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        random.seed(seed)
    
    def generate_health_reports(self, num_reports: int = 100, 
                              outbreak_areas: List[Dict] = None) -> pd.DataFrame:
        """Generate sample health reports"""
        if outbreak_areas is None:
            outbreak_areas = [
                {'lat': 28.6139, 'lon': 77.2090, 'radius': 0.01, 'intensity': 0.8},  # Delhi
                {'lat': 19.0760, 'lon': 72.8777, 'radius': 0.01, 'intensity': 0.6},  # Mumbai
            ]
        
        reports = []
        
        for i in range(num_reports):
            # Determine if this is an outbreak case
            is_outbreak = False
            for area in outbreak_areas:
                if random.random() < area['intensity']:
                    # Generate location near outbreak area
                    lat = area['lat'] + np.random.normal(0, area['radius'])
                    lon = area['lon'] + np.random.normal(0, area['radius'])
                    is_outbreak = True
                    break
            
            if not is_outbreak:
                # Generate random location in India
                lat = np.random.uniform(6.0, 37.0)
                lon = np.random.uniform(68.0, 97.0)
            
            # Generate symptoms based on outbreak status
            if is_outbreak:
                fever = random.random() < 0.7
                diarrhea = random.random() < 0.8
                vomiting = random.random() < 0.6
                nausea = random.random() < 0.7
                abdominal_pain = random.random() < 0.6
                dehydration = random.random() < 0.4
                severity = random.randint(3, 5)
            else:
                fever = random.random() < 0.1
                diarrhea = random.random() < 0.05
                vomiting = random.random() < 0.03
                nausea = random.random() < 0.08
                abdominal_pain = random.random() < 0.1
                dehydration = random.random() < 0.02
                severity = random.randint(1, 3)
            
            report = {
                'report_id': f"RPT_{i:06d}",
                'user_id': f"USER_{random.randint(1000, 9999)}",
                'location_lat': lat,
                'location_lon': lon,
                'location_address': f"Address {i}",
                'fever': fever,
                'diarrhea': diarrhea,
                'vomiting': vomiting,
                'nausea': nausea,
                'abdominal_pain': abdominal_pain,
                'dehydration': dehydration,
                'other_symptoms': f"Symptom details {i}" if random.random() < 0.3 else None,
                'symptom_severity': severity,
                'report_source': random.choice(['mobile_app', 'sms', 'web']),
                'report_timestamp': datetime.now() - timedelta(days=random.randint(0, 30))
            }
            reports.append(report)
        
        return pd.DataFrame(reports)
    
    def generate_water_quality_data(self, num_records: int = 200,
                                  contaminated_areas: List[Dict] = None) -> pd.DataFrame:
        """Generate sample water quality data"""
        if contaminated_areas is None:
            contaminated_areas = [
                {'lat': 28.6139, 'lon': 77.2090, 'radius': 0.02, 'contamination': 0.7},
                {'lat': 19.0760, 'lon': 72.8777, 'radius': 0.02, 'contamination': 0.5},
            ]
        
        records = []
        
        for i in range(num_records):
            # Determine if this is a contaminated area
            is_contaminated = False
            for area in contaminated_areas:
                if random.random() < area['contamination']:
                    lat = area['lat'] + np.random.normal(0, area['radius'])
                    lon = area['lon'] + np.random.normal(0, area['radius'])
                    is_contaminated = True
                    break
            
            if not is_contaminated:
                # Generate random location
                lat = np.random.uniform(6.0, 37.0)
                lon = np.random.uniform(68.0, 97.0)
            
            # Generate water quality parameters
            if is_contaminated:
                turbidity = np.random.uniform(5.0, 15.0)
                ph_level = np.random.uniform(6.0, 8.5)
                bacterial_count = np.random.uniform(100, 1000)
                chlorine_residual = np.random.uniform(0.0, 0.5)
            else:
                turbidity = np.random.uniform(0.1, 4.0)
                ph_level = np.random.uniform(6.5, 8.0)
                bacterial_count = np.random.uniform(0, 50)
                chlorine_residual = np.random.uniform(0.2, 2.0)
            
            record = {
                'sensor_id': f"SENSOR_{i:04d}",
                'location_lat': lat,
                'location_lon': lon,
                'turbidity': turbidity,
                'ph_level': ph_level,
                'temperature': np.random.uniform(20, 35),
                'dissolved_oxygen': np.random.uniform(5, 12),
                'bacterial_count': bacterial_count,
                'chlorine_residual': chlorine_residual,
                'conductivity': np.random.uniform(100, 1000),
                'total_dissolved_solids': np.random.uniform(50, 500),
                'nitrate_level': np.random.uniform(0, 10),
                'phosphate_level': np.random.uniform(0, 2),
                'measurement_timestamp': datetime.now() - timedelta(hours=random.randint(0, 168))
            }
            records.append(record)
        
        return pd.DataFrame(records)
    
    def generate_time_series_data(self, days: int = 30) -> pd.DataFrame:
        """Generate time series data for forecasting"""
        dates = pd.date_range(start=datetime.now() - timedelta(days=days), 
                             end=datetime.now(), freq='D')
        
        # Generate trend and seasonality
        trend = np.linspace(10, 25, len(dates))
        seasonal = 5 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)  # Weekly seasonality
        noise = np.random.normal(0, 2, len(dates))
        
        case_counts = np.maximum(0, trend + seasonal + noise).astype(int)
        
        return pd.DataFrame({
            'date': dates,
            'case_count': case_counts
        })
    
    def generate_outbreak_scenario(self) -> Dict[str, Any]:
        """Generate a complete outbreak scenario"""
        # Generate outbreak areas
        outbreak_areas = [
            {'lat': 28.6139, 'lon': 77.2090, 'radius': 0.01, 'intensity': 0.8},
            {'lat': 19.0760, 'lon': 72.8777, 'radius': 0.01, 'intensity': 0.6},
        ]
        
        contaminated_areas = [
            {'lat': 28.6139, 'lon': 77.2090, 'radius': 0.02, 'contamination': 0.7},
            {'lat': 19.0760, 'lon': 72.8777, 'radius': 0.02, 'contamination': 0.5},
        ]
        
        # Generate data
        health_reports = self.generate_health_reports(200, outbreak_areas)
        water_quality = self.generate_water_quality_data(100, contaminated_areas)
        time_series = self.generate_time_series_data(30)
        
        return {
            'health_reports': health_reports,
            'water_quality': water_quality,
            'time_series': time_series,
            'outbreak_areas': outbreak_areas,
            'contaminated_areas': contaminated_areas
        }
