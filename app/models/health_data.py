"""
Database models for health surveillance data
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from app.core.database import Base
import uuid

class HealthReport(Base):
    """Health reports from mobile apps/SMS"""
    __tablename__ = "health_reports"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    report_id = Column(String(50), unique=True, nullable=False)
    user_id = Column(String(50), nullable=False)
    location_lat = Column(Float, nullable=False)
    location_lon = Column(Float, nullable=False)
    location_address = Column(Text)
    
    # Symptoms
    fever = Column(Boolean, default=False)
    diarrhea = Column(Boolean, default=False)
    vomiting = Column(Boolean, default=False)
    nausea = Column(Boolean, default=False)
    abdominal_pain = Column(Boolean, default=False)
    dehydration = Column(Boolean, default=False)
    
    # Additional symptoms
    other_symptoms = Column(Text)
    symptom_severity = Column(Integer)  # 1-5 scale
    
    # Metadata
    report_source = Column(String(20))  # mobile_app, sms, web
    report_timestamp = Column(DateTime(timezone=True), server_default=func.now())
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Processing status
    is_processed = Column(Boolean, default=False)
    risk_score = Column(Float)
    predicted_disease = Column(String(50))

class WaterQualityData(Base):
    """Water quality sensor data"""
    __tablename__ = "water_quality_data"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    sensor_id = Column(String(50), nullable=False)
    location_lat = Column(Float, nullable=False)
    location_lon = Column(Float, nullable=False)
    
    # Water quality parameters
    turbidity = Column(Float)  # NTU
    ph_level = Column(Float)
    temperature = Column(Float)  # Celsius
    dissolved_oxygen = Column(Float)  # mg/L
    bacterial_count = Column(Float)  # CFU/mL
    chlorine_residual = Column(Float)  # mg/L
    
    # Additional parameters
    conductivity = Column(Float)  # μS/cm
    total_dissolved_solids = Column(Float)  # mg/L
    nitrate_level = Column(Float)  # mg/L
    phosphate_level = Column(Float)  # mg/L
    
    # Metadata
    measurement_timestamp = Column(DateTime(timezone=True), server_default=func.now())
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Quality indicators
    is_contaminated = Column(Boolean, default=False)
    contamination_level = Column(String(20))  # low, medium, high, critical

class DiseaseOutbreak(Base):
    """Disease outbreak predictions and alerts"""
    __tablename__ = "disease_outbreaks"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    outbreak_id = Column(String(50), unique=True, nullable=False)
    
    # Location
    location_lat = Column(Float, nullable=False)
    location_lon = Column(Float, nullable=False)
    location_name = Column(String(100))
    radius_km = Column(Float)  # Affected radius
    
    # Prediction details
    predicted_disease = Column(String(50), nullable=False)
    outbreak_probability = Column(Float, nullable=False)  # 0-1
    severity_level = Column(String(20))  # low, medium, high, critical
    
    # Time information
    predicted_start_date = Column(DateTime(timezone=True))
    predicted_end_date = Column(DateTime(timezone=True))
    prediction_timestamp = Column(DateTime(timezone=True), server_default=func.now())
    
    # Model information
    model_version = Column(String(20))
    confidence_score = Column(Float)
    contributing_factors = Column(JSON)  # Store factors that led to prediction
    
    # Status
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    verification_notes = Column(Text)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class HotspotCluster(Base):
    """Geospatial clusters of disease cases"""
    __tablename__ = "hotspot_clusters"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    cluster_id = Column(String(50), unique=True, nullable=False)
    
    # Cluster center
    center_lat = Column(Float, nullable=False)
    center_lon = Column(Float, nullable=False)
    
    # Cluster properties
    radius_km = Column(Float, nullable=False)
    case_count = Column(Integer, nullable=False)
    density = Column(Float)  # cases per km²
    
    # Risk assessment
    risk_level = Column(String(20))  # low, medium, high, critical
    risk_score = Column(Float)  # 0-1
    
    # Time information
    first_case_date = Column(DateTime(timezone=True))
    last_case_date = Column(DateTime(timezone=True))
    cluster_timestamp = Column(DateTime(timezone=True), server_default=func.now())
    
    # Status
    is_active = Column(Boolean, default=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class ModelPerformance(Base):
    """Track ML model performance metrics"""
    __tablename__ = "model_performance"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_name = Column(String(50), nullable=False)
    model_version = Column(String(20), nullable=False)
    
    # Performance metrics
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    auc_score = Column(Float)
    
    # Training information
    training_timestamp = Column(DateTime(timezone=True), server_default=func.now())
    training_duration_seconds = Column(Integer)
    training_samples = Column(Integer)
    
    # Validation information
    validation_samples = Column(Integer)
    validation_accuracy = Column(Float)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
