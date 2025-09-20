"""
Data processing and preprocessing services
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from datetime import datetime, timedelta
import re
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, insert
from app.models.health_data import HealthReport, WaterQualityData

class DataProcessor:
    """Data preprocessing and cleaning service"""
    
    def __init__(self):
        self.symptom_keywords = {
            'fever': ['fever', 'temperature', 'hot', 'burning'],
            'diarrhea': ['diarrhea', 'loose', 'watery', 'stool'],
            'vomiting': ['vomit', 'throw up', 'nausea', 'sick'],
            'abdominal_pain': ['stomach', 'belly', 'abdominal', 'cramp'],
            'dehydration': ['dehydrated', 'thirsty', 'dry mouth', 'dizzy']
        }
    
    async def process_health_reports_csv(self, df: pd.DataFrame, db: AsyncSession) -> List[Dict]:
        """Process health reports from CSV upload"""
        processed_reports = []
        
        for _, row in df.iterrows():
            try:
                # Clean and validate data
                report_data = {
                    'report_id': str(row.get('report_id', f"csv_{datetime.now().timestamp()}")),
                    'user_id': str(row.get('user_id', 'unknown')),
                    'location_lat': float(row.get('latitude', 0)),
                    'location_lon': float(row.get('longitude', 0)),
                    'location_address': str(row.get('address', '')),
                    'fever': self._parse_boolean(row.get('fever', False)),
                    'diarrhea': self._parse_boolean(row.get('diarrhea', False)),
                    'vomiting': self._parse_boolean(row.get('vomiting', False)),
                    'nausea': self._parse_boolean(row.get('nausea', False)),
                    'abdominal_pain': self._parse_boolean(row.get('abdominal_pain', False)),
                    'dehydration': self._parse_boolean(row.get('dehydration', False)),
                    'other_symptoms': str(row.get('other_symptoms', '')),
                    'symptom_severity': int(row.get('severity', 1)) if pd.notna(row.get('severity')) else None,
                    'report_source': 'csv_upload',
                    'report_timestamp': self._parse_datetime(row.get('timestamp', datetime.now()))
                }
                
                # Create database record
                health_report = HealthReport(**report_data)
                db.add(health_report)
                processed_reports.append(report_data)
                
            except Exception as e:
                print(f"Error processing health report row: {e}")
                continue
        
        await db.commit()
        return processed_reports
    
    async def process_water_quality_csv(self, df: pd.DataFrame, db: AsyncSession) -> List[Dict]:
        """Process water quality data from CSV upload"""
        processed_data = []
        
        for _, row in df.iterrows():
            try:
                # Clean and validate data
                water_data = {
                    'sensor_id': str(row.get('sensor_id', f"sensor_{datetime.now().timestamp()}")),
                    'location_lat': float(row.get('latitude', 0)),
                    'location_lon': float(row.get('longitude', 0)),
                    'turbidity': float(row.get('turbidity', 0)) if pd.notna(row.get('turbidity')) else None,
                    'ph_level': float(row.get('ph', 7)) if pd.notna(row.get('ph')) else None,
                    'temperature': float(row.get('temperature', 25)) if pd.notna(row.get('temperature')) else None,
                    'dissolved_oxygen': float(row.get('dissolved_oxygen', 0)) if pd.notna(row.get('dissolved_oxygen')) else None,
                    'bacterial_count': float(row.get('bacterial_count', 0)) if pd.notna(row.get('bacterial_count')) else None,
                    'chlorine_residual': float(row.get('chlorine', 0)) if pd.notna(row.get('chlorine')) else None,
                    'conductivity': float(row.get('conductivity', 0)) if pd.notna(row.get('conductivity')) else None,
                    'total_dissolved_solids': float(row.get('tds', 0)) if pd.notna(row.get('tds')) else None,
                    'nitrate_level': float(row.get('nitrate', 0)) if pd.notna(row.get('nitrate')) else None,
                    'phosphate_level': float(row.get('phosphate', 0)) if pd.notna(row.get('phosphate')) else None,
                    'measurement_timestamp': self._parse_datetime(row.get('timestamp', datetime.now()))
                }
                
                # Determine contamination level
                contamination_level = self._assess_contamination(water_data)
                water_data['is_contaminated'] = contamination_level['is_contaminated']
                water_data['contamination_level'] = contamination_level['level']
                
                # Create database record
                water_quality = WaterQualityData(**water_data)
                db.add(water_quality)
                processed_data.append(water_data)
                
            except Exception as e:
                print(f"Error processing water quality row: {e}")
                continue
        
        await db.commit()
        return processed_data
    
    def process_sms_text(self, message: str) -> Dict[str, Any]:
        """Process SMS text to extract symptoms using NLP"""
        message_lower = message.lower()
        extracted_symptoms = {}
        
        for symptom, keywords in self.symptom_keywords.items():
            extracted_symptoms[symptom] = any(keyword in message_lower for keyword in keywords)
        
        # Extract severity (simple heuristic)
        severity = 1
        if any(word in message_lower for word in ['severe', 'bad', 'terrible', 'awful']):
            severity = 5
        elif any(word in message_lower for word in ['moderate', 'medium']):
            severity = 3
        elif any(word in message_lower for word in ['mild', 'slight', 'little']):
            severity = 2
        
        return {
            'symptoms': extracted_symptoms,
            'severity': severity,
            'other_symptoms': message
        }
    
    def clean_time_series_data(self, df: pd.DataFrame, time_column: str) -> pd.DataFrame:
        """Clean and prepare time series data"""
        # Convert time column to datetime
        df[time_column] = pd.to_datetime(df[time_column])
        
        # Sort by time
        df = df.sort_values(time_column)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=[time_column])
        
        # Fill missing values using forward fill
        df = df.fillna(method='ffill')
        
        # Remove outliers using IQR method
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col != time_column:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        return df
    
    def normalize_features(self, df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
        """Normalize features using Min-Max scaling"""
        from sklearn.preprocessing import MinMaxScaler
        
        scaler = MinMaxScaler()
        df[feature_columns] = scaler.fit_transform(df[feature_columns])
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame, categorical_columns: List[str]) -> pd.DataFrame:
        """Encode categorical features using one-hot encoding"""
        return pd.get_dummies(df, columns=categorical_columns, prefix=categorical_columns)
    
    def _parse_boolean(self, value: Any) -> bool:
        """Parse various boolean representations"""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ['true', '1', 'yes', 'y', 'on']
        if isinstance(value, (int, float)):
            return bool(value)
        return False
    
    def _parse_datetime(self, value: Any) -> datetime:
        """Parse various datetime representations"""
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace('Z', '+00:00'))
            except:
                return datetime.now()
        return datetime.now()
    
    def _assess_contamination(self, water_data: Dict) -> Dict[str, Any]:
        """Assess water contamination level based on parameters"""
        is_contaminated = False
        level = "low"
        
        # Bacterial count threshold
        if water_data.get('bacterial_count', 0) > 100:
            is_contaminated = True
            level = "high"
        elif water_data.get('bacterial_count', 0) > 50:
            is_contaminated = True
            level = "medium"
        
        # Turbidity threshold
        if water_data.get('turbidity', 0) > 4:
            is_contaminated = True
            if level == "low":
                level = "medium"
        
        # pH level check
        ph = water_data.get('ph_level', 7)
        if ph < 6.5 or ph > 8.5:
            is_contaminated = True
            if level == "low":
                level = "medium"
        
        return {
            'is_contaminated': is_contaminated,
            'level': level
        }
