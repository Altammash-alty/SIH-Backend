"""
Prediction API endpoints
"""
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import pandas as pd

from app.core.database import get_db, get_mongo
from app.ml.model_manager import ModelManager
from app.models.health_data import DiseaseOutbreak, HotspotCluster, ModelPerformance, HealthReport, WaterQualityData
from app.services.data_processor import DataProcessor
import logging

router = APIRouter()

# Pydantic models for request validation
class PredictionRequest(BaseModel):
    health_data: Optional[List[Dict[str, Any]]] = None
    water_quality_data: Optional[List[Dict[str, Any]]] = None
    location_lat: Optional[float] = None
    location_lon: Optional[float] = None
    prediction_horizon_days: int = 7

class BatchPredictionRequest(BaseModel):
    prediction_requests: List[PredictionRequest]
    save_results: bool = True

@router.post("/outbreak")
async def predict_outbreak(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Predict disease outbreak risk"""
    try:
        # Get ML models from app state
        from app.main import ml_models
        
        if not ml_models:
            raise HTTPException(status_code=503, detail="ML models not loaded")
        
        # Prepare data
        health_df = pd.DataFrame(request.health_data or [])
        water_df = pd.DataFrame(request.water_quality_data or [])
        
        # If no data provided, get recent data from database
        if health_df.empty or water_df.empty:
            health_df, water_df = await _get_recent_data(db, request.location_lat, request.location_lon)
        
        if health_df.empty and water_df.empty:
            raise HTTPException(status_code=400, detail="No data available for prediction")
        
        # Make prediction using model manager
        model_manager = ModelManager()
        model_manager.models = ml_models
        
        prediction_result = await model_manager.predict_outbreak(health_df, water_df)
        
        # Save prediction to database if requested
        if request.save_results:
            background_tasks.add_task(
                _save_prediction_result,
                prediction_result,
                request.location_lat,
                request.location_lon,
                db
            )
        
        return {
            "prediction_id": f"pred_{datetime.now().timestamp()}",
            "timestamp": datetime.now().isoformat(),
            "prediction": prediction_result,
            "data_summary": {
                "health_reports_count": len(health_df),
                "water_quality_records_count": len(water_df)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.post("/batch")
async def batch_predict_outbreaks(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Make batch predictions for multiple locations"""
    try:
        results = []
        
        for i, pred_request in enumerate(request.prediction_requests):
            try:
                # Make individual prediction
                prediction_result = await predict_outbreak(
                    pred_request, 
                    background_tasks, 
                    db
                )
                results.append({
                    "request_id": i,
                    "status": "success",
                    "result": prediction_result
                })
            except Exception as e:
                results.append({
                    "request_id": i,
                    "status": "error",
                    "error": str(e)
                })
        
        return {
            "batch_id": f"batch_{datetime.now().timestamp()}",
            "timestamp": datetime.now().isoformat(),
            "total_requests": len(request.prediction_requests),
            "successful_predictions": len([r for r in results if r["status"] == "success"]),
            "results": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@router.get("/latest")
async def get_latest_predictions(
    limit: int = 10,
    hours_back: int = 24,
    db: AsyncSession = Depends(get_db)
):
    """Get latest predictions for frontend visualization"""
    try:
        from sqlalchemy import select, and_
        from datetime import datetime, timedelta
        
        # Calculate time threshold
        time_threshold = datetime.now() - timedelta(hours=hours_back)
        
        # Get recent outbreaks
        outbreak_result = await db.execute(
            select(DiseaseOutbreak)
            .where(DiseaseOutbreak.prediction_timestamp >= time_threshold)
            .order_by(DiseaseOutbreak.prediction_timestamp.desc())
            .limit(limit)
        )
        outbreaks = outbreak_result.scalars().all()
        
        # Get recent hotspots
        hotspot_result = await db.execute(
            select(HotspotCluster)
            .where(HotspotCluster.cluster_timestamp >= time_threshold)
            .order_by(HotspotCluster.cluster_timestamp.desc())
            .limit(limit)
        )
        hotspots = hotspot_result.scalars().all()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "outbreaks": [
                {
                    "id": str(outbreak.id),
                    "outbreak_id": outbreak.outbreak_id,
                    "location": {
                        "lat": outbreak.location_lat,
                        "lon": outbreak.location_lon,
                        "name": outbreak.location_name
                    },
                    "predicted_disease": outbreak.predicted_disease,
                    "outbreak_probability": outbreak.outbreak_probability,
                    "severity_level": outbreak.severity_level,
                    "confidence_score": outbreak.confidence_score,
                    "prediction_timestamp": outbreak.prediction_timestamp.isoformat(),
                    "is_active": outbreak.is_active
                }
                for outbreak in outbreaks
            ],
            "hotspots": [
                {
                    "id": str(hotspot.id),
                    "cluster_id": hotspot.cluster_id,
                    "center": {
                        "lat": hotspot.center_lat,
                        "lon": hotspot.center_lon
                    },
                    "radius_km": hotspot.radius_km,
                    "case_count": hotspot.case_count,
                    "density": hotspot.density,
                    "risk_level": hotspot.risk_level,
                    "risk_score": hotspot.risk_score,
                    "cluster_timestamp": hotspot.cluster_timestamp.isoformat(),
                    "is_active": hotspot.is_active
                }
                for hotspot in hotspots
            ],
            "summary": {
                "total_outbreaks": len(outbreaks),
                "total_hotspots": len(hotspots),
                "high_risk_outbreaks": len([o for o in outbreaks if o.severity_level in ['high', 'critical']]),
                "active_hotspots": len([h for h in hotspots if h.is_active])
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve latest predictions: {str(e)}")

@router.get("/outbreaks")
async def get_outbreaks(
    limit: int = 50,
    offset: int = 0,
    severity_filter: Optional[str] = None,
    disease_filter: Optional[str] = None,
    active_only: bool = True,
    db: AsyncSession = Depends(get_db)
):
    """Get outbreak predictions with filtering"""
    try:
        from sqlalchemy import select, and_
        
        # Build query
        query = select(DiseaseOutbreak)
        
        conditions = []
        if active_only:
            conditions.append(DiseaseOutbreak.is_active == True)
        if severity_filter:
            conditions.append(DiseaseOutbreak.severity_level == severity_filter)
        if disease_filter:
            conditions.append(DiseaseOutbreak.predicted_disease.ilike(f"%{disease_filter}%"))
        
        if conditions:
            query = query.where(and_(*conditions))
        
        query = query.order_by(DiseaseOutbreak.prediction_timestamp.desc()).offset(offset).limit(limit)
        
        result = await db.execute(query)
        outbreaks = result.scalars().all()
        
        return {
            "outbreaks": [
                {
                    "id": str(outbreak.id),
                    "outbreak_id": outbreak.outbreak_id,
                    "location": {
                        "lat": outbreak.location_lat,
                        "lon": outbreak.location_lon,
                        "name": outbreak.location_name,
                        "radius_km": outbreak.radius_km
                    },
                    "prediction": {
                        "disease": outbreak.predicted_disease,
                        "probability": outbreak.outbreak_probability,
                        "severity": outbreak.severity_level,
                        "confidence": outbreak.confidence_score
                    },
                    "timing": {
                        "predicted_start": outbreak.predicted_start_date.isoformat() if outbreak.predicted_start_date else None,
                        "predicted_end": outbreak.predicted_end_date.isoformat() if outbreak.predicted_end_date else None,
                        "prediction_timestamp": outbreak.prediction_timestamp.isoformat()
                    },
                    "status": {
                        "is_active": outbreak.is_active,
                        "is_verified": outbreak.is_verified,
                        "verification_notes": outbreak.verification_notes
                    },
                    "model_info": {
                        "version": outbreak.model_version,
                        "contributing_factors": outbreak.contributing_factors
                    }
                }
                for outbreak in outbreaks
            ],
            "pagination": {
                "limit": limit,
                "offset": offset,
                "total": len(outbreaks)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve outbreaks: {str(e)}")

@router.get("/hotspots")
async def get_hotspots(
    limit: int = 50,
    offset: int = 0,
    risk_level_filter: Optional[str] = None,
    active_only: bool = True,
    db: AsyncSession = Depends(get_db)
):
    """Get disease hotspots with filtering"""
    try:
        from sqlalchemy import select, and_
        
        # Build query
        query = select(HotspotCluster)
        
        conditions = []
        if active_only:
            conditions.append(HotspotCluster.is_active == True)
        if risk_level_filter:
            conditions.append(HotspotCluster.risk_level == risk_level_filter)
        
        if conditions:
            query = query.where(and_(*conditions))
        
        query = query.order_by(HotspotCluster.risk_score.desc()).offset(offset).limit(limit)
        
        result = await db.execute(query)
        hotspots = result.scalars().all()
        
        return {
            "hotspots": [
                {
                    "id": str(hotspot.id),
                    "cluster_id": hotspot.cluster_id,
                    "location": {
                        "center_lat": hotspot.center_lat,
                        "center_lon": hotspot.center_lon,
                        "radius_km": hotspot.radius_km
                    },
                    "statistics": {
                        "case_count": hotspot.case_count,
                        "density": hotspot.density,
                        "risk_level": hotspot.risk_level,
                        "risk_score": hotspot.risk_score
                    },
                    "timing": {
                        "first_case": hotspot.first_case_date.isoformat() if hotspot.first_case_date else None,
                        "last_case": hotspot.last_case_date.isoformat() if hotspot.last_case_date else None,
                        "cluster_timestamp": hotspot.cluster_timestamp.isoformat()
                    },
                    "status": {
                        "is_active": hotspot.is_active
                    }
                }
                for hotspot in hotspots
            ],
            "pagination": {
                "limit": limit,
                "offset": offset,
                "total": len(hotspots)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve hotspots: {str(e)}")

@router.get("/model-performance")
async def get_model_performance(db: AsyncSession = Depends(get_db)):
    """Get ML model performance metrics"""
    try:
        from sqlalchemy import select
        
        # Get recent model performance records
        result = await db.execute(
            select(ModelPerformance)
            .order_by(ModelPerformance.training_timestamp.desc())
            .limit(10)
        )
        performance_records = result.scalars().all()
        
        # Get current model performance from model manager
        from app.main import ml_models
        current_performance = {}
        if ml_models:
            model_manager = ModelManager()
            model_manager.models = ml_models
            current_performance = model_manager.get_model_performance()
        
        return {
            "current_performance": current_performance,
            "historical_performance": [
                {
                    "model_name": record.model_name,
                    "model_version": record.model_version,
                    "metrics": {
                        "accuracy": record.accuracy,
                        "precision": record.precision,
                        "recall": record.recall,
                        "f1_score": record.f1_score,
                        "auc_score": record.auc_score
                    },
                    "training_info": {
                        "timestamp": record.training_timestamp.isoformat(),
                        "duration_seconds": record.training_duration_seconds,
                        "training_samples": record.training_samples,
                        "validation_samples": record.validation_samples,
                        "validation_accuracy": record.validation_accuracy
                    }
                }
                for record in performance_records
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve model performance: {str(e)}")

async def _get_recent_data(db: AsyncSession, lat: Optional[float], lon: Optional[float]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Get recent health and water quality data"""
    from sqlalchemy import select, and_
    from datetime import datetime, timedelta
    
    # Get data from last 7 days
    time_threshold = datetime.now() - timedelta(days=7)
    
    # Get health reports
    health_result = await db.execute(
        select(HealthReport)
        .where(HealthReport.created_at >= time_threshold)
    )
    health_reports = health_result.scalars().all()
    
    # Get water quality data
    water_result = await db.execute(
        select(WaterQualityData)
        .where(WaterQualityData.created_at >= time_threshold)
    )
    water_data = water_result.scalars().all()
    
    # Convert to DataFrames
    health_df = pd.DataFrame([
        {
            'location_lat': report.location_lat,
            'location_lon': report.location_lon,
            'fever': report.fever,
            'diarrhea': report.diarrhea,
            'vomiting': report.vomiting,
            'nausea': report.nausea,
            'abdominal_pain': report.abdominal_pain,
            'dehydration': report.dehydration,
            'symptom_severity': report.symptom_severity,
            'report_timestamp': report.report_timestamp
        }
        for report in health_reports
    ])
    
    water_df = pd.DataFrame([
        {
            'location_lat': data.location_lat,
            'location_lon': data.location_lon,
            'turbidity': data.turbidity,
            'ph_level': data.ph_level,
            'temperature': data.temperature,
            'dissolved_oxygen': data.dissolved_oxygen,
            'bacterial_count': data.bacterial_count,
            'chlorine_residual': data.chlorine_residual,
            'conductivity': data.conductivity,
            'total_dissolved_solids': data.total_dissolved_solids,
            'nitrate_level': data.nitrate_level,
            'phosphate_level': data.phosphate_level,
            'is_contaminated': data.is_contaminated,
            'measurement_timestamp': data.measurement_timestamp
        }
        for data in water_data
    ])
    
    return health_df, water_df

async def _save_prediction_result(
    prediction_result: Dict[str, Any],
    lat: Optional[float],
    lon: Optional[float],
    db: AsyncSession
):
    """Save prediction result to database"""
    try:
        # Create outbreak record
        outbreak = DiseaseOutbreak(
            outbreak_id=f"outbreak_{datetime.now().timestamp()}",
            location_lat=lat or 0.0,
            location_lon=lon or 0.0,
            predicted_disease=prediction_result.get('predicted_disease', 'Unknown'),
            outbreak_probability=prediction_result.get('outbreak_probability', 0.0),
            severity_level=prediction_result.get('severity_level', 'low'),
            confidence_score=prediction_result.get('confidence_level', 0.0),
            contributing_factors=prediction_result.get('contributing_factors', []),
            model_version="1.0"
        )
        
        db.add(outbreak)
        await db.commit()
        
    except Exception as e:
        logger.error(f"Failed to save prediction result: {e}")
        await db.rollback()
