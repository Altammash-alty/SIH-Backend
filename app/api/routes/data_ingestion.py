"""
Data ingestion endpoints for health reports, water quality, and file uploads
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import json
from datetime import datetime
import uuid

from app.core.database import get_db, get_mongo
from app.models.health_data import HealthReport, WaterQualityData
from app.services.data_processor import DataProcessor
import io

router = APIRouter()

# Pydantic models for request validation
class HealthReportRequest(BaseModel):
    report_id: str
    user_id: str
    location_lat: float
    location_lon: float
    location_address: Optional[str] = None
    fever: bool = False
    diarrhea: bool = False
    vomiting: bool = False
    nausea: bool = False
    abdominal_pain: bool = False
    dehydration: bool = False
    other_symptoms: Optional[str] = None
    symptom_severity: Optional[int] = None
    report_source: str = "mobile_app"

class WaterQualityRequest(BaseModel):
    sensor_id: str
    location_lat: float
    location_lon: float
    turbidity: Optional[float] = None
    ph_level: Optional[float] = None
    temperature: Optional[float] = None
    dissolved_oxygen: Optional[float] = None
    bacterial_count: Optional[float] = None
    chlorine_residual: Optional[float] = None
    conductivity: Optional[float] = None
    total_dissolved_solids: Optional[float] = None
    nitrate_level: Optional[float] = None
    phosphate_level: Optional[float] = None

class SMSReportRequest(BaseModel):
    phone_number: str
    message: str
    location_lat: float
    location_lon: float
    timestamp: Optional[datetime] = None

@router.post("/health-reports")
async def create_health_report(
    report: HealthReportRequest,
    db: AsyncSession = Depends(get_db)
):
    """Create a new health report"""
    try:
        # Create health report record
        health_report = HealthReport(
            report_id=report.report_id,
            user_id=report.user_id,
            location_lat=report.location_lat,
            location_lon=report.location_lon,
            location_address=report.location_address,
            fever=report.fever,
            diarrhea=report.diarrhea,
            vomiting=report.vomiting,
            nausea=report.nausea,
            abdominal_pain=report.abdominal_pain,
            dehydration=report.dehydration,
            other_symptoms=report.other_symptoms,
            symptom_severity=report.symptom_severity,
            report_source=report.report_source
        )
        
        db.add(health_report)
        await db.commit()
        await db.refresh(health_report)
        
        return {
            "message": "Health report created successfully",
            "report_id": health_report.report_id,
            "id": str(health_report.id)
        }
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=400, detail=f"Failed to create health report: {str(e)}")

@router.post("/water-quality")
async def create_water_quality_data(
    data: WaterQualityRequest,
    db: AsyncSession = Depends(get_db)
):
    """Create new water quality sensor data"""
    try:
        # Determine contamination level
        contamination_level = "low"
        is_contaminated = False
        
        if data.bacterial_count and data.bacterial_count > 100:
            contamination_level = "high"
            is_contaminated = True
        elif data.turbidity and data.turbidity > 4:
            contamination_level = "medium"
            is_contaminated = True
        
        water_data = WaterQualityData(
            sensor_id=data.sensor_id,
            location_lat=data.location_lat,
            location_lon=data.location_lon,
            turbidity=data.turbidity,
            ph_level=data.ph_level,
            temperature=data.temperature,
            dissolved_oxygen=data.dissolved_oxygen,
            bacterial_count=data.bacterial_count,
            chlorine_residual=data.chlorine_residual,
            conductivity=data.conductivity,
            total_dissolved_solids=data.total_dissolved_solids,
            nitrate_level=data.nitrate_level,
            phosphate_level=data.phosphate_level,
            is_contaminated=is_contaminated,
            contamination_level=contamination_level
        )
        
        db.add(water_data)
        await db.commit()
        await db.refresh(water_data)
        
        return {
            "message": "Water quality data created successfully",
            "sensor_id": water_data.sensor_id,
            "id": str(water_data.id),
            "contamination_level": contamination_level
        }
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=400, detail=f"Failed to create water quality data: {str(e)}")

@router.post("/sms-reports")
async def process_sms_report(
    sms: SMSReportRequest,
    mongo_db = Depends(get_mongo)
):
    """Process SMS-based health reports using NLP"""
    try:
        # Store raw SMS in MongoDB for NLP processing
        sms_doc = {
            "phone_number": sms.phone_number,
            "message": sms.message,
            "location_lat": sms.location_lat,
            "location_lon": sms.location_lon,
            "timestamp": sms.timestamp or datetime.utcnow(),
            "processed": False,
            "created_at": datetime.utcnow()
        }
        
        result = await mongo_db.sms_reports.insert_one(sms_doc)
        
        # TODO: Trigger NLP processing pipeline
        # This would typically be handled by a background task
        
        return {
            "message": "SMS report received and queued for processing",
            "document_id": str(result.inserted_id)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process SMS report: {str(e)}")

@router.post("/upload-csv")
async def upload_csv_file(
    file: UploadFile = File(...),
    data_type: str = Form(...),  # "health_reports" or "water_quality"
    db: AsyncSession = Depends(get_db)
):
    """Upload and process CSV/Excel files"""
    try:
        if not file.filename.endswith(('.csv', '.xlsx')):
            raise HTTPException(status_code=400, detail="File must be CSV or Excel format")
        
        # Read file content
        content = await file.read()
        
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        else:
            df = pd.read_excel(io.BytesIO(content))
        
        # Process data based on type
        processor = DataProcessor()
        
        if data_type == "health_reports":
            processed_data = await processor.process_health_reports_csv(df, db)
        elif data_type == "water_quality":
            processed_data = await processor.process_water_quality_csv(df, db)
        else:
            raise HTTPException(status_code=400, detail="Invalid data_type. Must be 'health_reports' or 'water_quality'")
        
        return {
            "message": f"File processed successfully",
            "filename": file.filename,
            "records_processed": len(processed_data),
            "data_type": data_type
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process file: {str(e)}")

@router.get("/health-reports")
async def get_health_reports(
    limit: int = 100,
    offset: int = 0,
    db: AsyncSession = Depends(get_db)
):
    """Get health reports with pagination"""
    try:
        from sqlalchemy import select
        from sqlalchemy.orm import selectinload
        
        result = await db.execute(
            select(HealthReport)
            .offset(offset)
            .limit(limit)
            .order_by(HealthReport.created_at.desc())
        )
        reports = result.scalars().all()
        
        return {
            "reports": [
                {
                    "id": str(report.id),
                    "report_id": report.report_id,
                    "user_id": report.user_id,
                    "location": {
                        "lat": report.location_lat,
                        "lon": report.location_lon,
                        "address": report.location_address
                    },
                    "symptoms": {
                        "fever": report.fever,
                        "diarrhea": report.diarrhea,
                        "vomiting": report.vomiting,
                        "nausea": report.nausea,
                        "abdominal_pain": report.abdominal_pain,
                        "dehydration": report.dehydration,
                        "other": report.other_symptoms,
                        "severity": report.symptom_severity
                    },
                    "source": report.report_source,
                    "timestamp": report.report_timestamp,
                    "risk_score": report.risk_score,
                    "predicted_disease": report.predicted_disease
                }
                for report in reports
            ],
            "total": len(reports),
            "limit": limit,
            "offset": offset
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve health reports: {str(e)}")

@router.get("/water-quality")
async def get_water_quality_data(
    limit: int = 100,
    offset: int = 0,
    db: AsyncSession = Depends(get_db)
):
    """Get water quality data with pagination"""
    try:
        from sqlalchemy import select
        
        result = await db.execute(
            select(WaterQualityData)
            .offset(offset)
            .limit(limit)
            .order_by(WaterQualityData.created_at.desc())
        )
        water_data = result.scalars().all()
        
        return {
            "water_quality_data": [
                {
                    "id": str(data.id),
                    "sensor_id": data.sensor_id,
                    "location": {
                        "lat": data.location_lat,
                        "lon": data.location_lon
                    },
                    "parameters": {
                        "turbidity": data.turbidity,
                        "ph_level": data.ph_level,
                        "temperature": data.temperature,
                        "dissolved_oxygen": data.dissolved_oxygen,
                        "bacterial_count": data.bacterial_count,
                        "chlorine_residual": data.chlorine_residual,
                        "conductivity": data.conductivity,
                        "total_dissolved_solids": data.total_dissolved_solids,
                        "nitrate_level": data.nitrate_level,
                        "phosphate_level": data.phosphate_level
                    },
                    "contamination": {
                        "is_contaminated": data.is_contaminated,
                        "level": data.contamination_level
                    },
                    "timestamp": data.measurement_timestamp
                }
                for data in water_data
            ],
            "total": len(water_data),
            "limit": limit,
            "offset": offset
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve water quality data: {str(e)}")
