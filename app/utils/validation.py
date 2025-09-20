"""
Data validation utilities
"""
import re
from typing import Any, Dict, List, Optional
from datetime import datetime
import pandas as pd

def validate_coordinates(lat: float, lon: float) -> bool:
    """Validate latitude and longitude coordinates"""
    return -90 <= lat <= 90 and -180 <= lon <= 180

def validate_phone_number(phone: str) -> bool:
    """Validate phone number format"""
    # Simple phone number validation
    phone_pattern = r'^\+?[\d\s\-\(\)]{10,}$'
    return bool(re.match(phone_pattern, phone))

def validate_email(email: str) -> bool:
    """Validate email format"""
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(email_pattern, email))

def validate_health_report_data(data: Dict[str, Any]) -> List[str]:
    """Validate health report data"""
    errors = []
    
    # Required fields
    required_fields = ['user_id', 'location_lat', 'location_lon']
    for field in required_fields:
        if field not in data or data[field] is None:
            errors.append(f"Missing required field: {field}")
    
    # Validate coordinates
    if 'location_lat' in data and 'location_lon' in data:
        if not validate_coordinates(data['location_lat'], data['location_lon']):
            errors.append("Invalid coordinates")
    
    # Validate symptom severity
    if 'symptom_severity' in data and data['symptom_severity'] is not None:
        if not (1 <= data['symptom_severity'] <= 5):
            errors.append("Symptom severity must be between 1 and 5")
    
    return errors

def validate_water_quality_data(data: Dict[str, Any]) -> List[str]:
    """Validate water quality data"""
    errors = []
    
    # Required fields
    required_fields = ['sensor_id', 'location_lat', 'location_lon']
    for field in required_fields:
        if field not in data or data[field] is None:
            errors.append(f"Missing required field: {field}")
    
    # Validate coordinates
    if 'location_lat' in data and 'location_lon' in data:
        if not validate_coordinates(data['location_lat'], data['location_lon']):
            errors.append("Invalid coordinates")
    
    # Validate pH level
    if 'ph_level' in data and data['ph_level'] is not None:
        if not (0 <= data['ph_level'] <= 14):
            errors.append("pH level must be between 0 and 14")
    
    # Validate temperature
    if 'temperature' in data and data['temperature'] is not None:
        if not (-50 <= data['temperature'] <= 100):
            errors.append("Temperature must be between -50°C and 100°C")
    
    return errors

def validate_csv_data(df: pd.DataFrame, data_type: str) -> List[str]:
    """Validate CSV data based on type"""
    errors = []
    
    if data_type == "health_reports":
        required_columns = ['user_id', 'latitude', 'longitude']
        for col in required_columns:
            if col not in df.columns:
                errors.append(f"Missing required column: {col}")
        
        # Check for empty dataframe
        if df.empty:
            errors.append("CSV file is empty")
        
        # Check for missing values in required columns
        for col in required_columns:
            if col in df.columns and df[col].isnull().any():
                errors.append(f"Missing values in required column: {col}")
    
    elif data_type == "water_quality":
        required_columns = ['sensor_id', 'latitude', 'longitude']
        for col in required_columns:
            if col not in df.columns:
                errors.append(f"Missing required column: {col}")
        
        # Check for empty dataframe
        if df.empty:
            errors.append("CSV file is empty")
    
    return errors

def sanitize_text(text: str) -> str:
    """Sanitize text input"""
    if not text:
        return ""
    
    # Remove potentially harmful characters
    sanitized = re.sub(r'[<>"\']', '', str(text))
    return sanitized.strip()

def validate_timestamp(timestamp: Any) -> bool:
    """Validate timestamp format"""
    try:
        if isinstance(timestamp, str):
            datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        elif isinstance(timestamp, datetime):
            return True
        return False
    except (ValueError, TypeError):
        return False
