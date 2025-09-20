"""
Script to load sample data into the database
"""
import asyncio
import pandas as pd
import sys
import os
from pathlib import Path

# Add the app directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from app.core.database import init_db, SessionLocal
from app.models.health_data import HealthReport, WaterQualityData
from app.utils.sample_data_generator import SampleDataGenerator
from app.services.data_processor import DataProcessor

async def load_sample_data():
    """Load sample data into the database"""
    print("Loading sample data into database...")
    
    # Initialize database
    await init_db()
    
    # Generate sample data
    data_generator = SampleDataGenerator()
    scenario = data_generator.generate_outbreak_scenario()
    
    # Get database session
    async with SessionLocal() as db:
        try:
            # Load health reports
            print("Loading health reports...")
            health_reports = scenario['health_reports']
            
            for _, row in health_reports.iterrows():
                health_report = HealthReport(
                    report_id=row['report_id'],
                    user_id=row['user_id'],
                    location_lat=row['location_lat'],
                    location_lon=row['location_lon'],
                    location_address=row['location_address'],
                    fever=row['fever'],
                    diarrhea=row['diarrhea'],
                    vomiting=row['vomiting'],
                    nausea=row['nausea'],
                    abdominal_pain=row['abdominal_pain'],
                    dehydration=row['dehydration'],
                    other_symptoms=row['other_symptoms'],
                    symptom_severity=row['symptom_severity'],
                    report_source=row['report_source'],
                    report_timestamp=row['report_timestamp']
                )
                db.add(health_report)
            
            # Load water quality data
            print("Loading water quality data...")
            water_quality = scenario['water_quality']
            
            for _, row in water_quality.iterrows():
                # Determine contamination level
                contamination_level = "low"
                is_contaminated = False
                
                if row['bacterial_count'] > 100:
                    contamination_level = "high"
                    is_contaminated = True
                elif row['turbidity'] > 4:
                    contamination_level = "medium"
                    is_contaminated = True
                
                water_data = WaterQualityData(
                    sensor_id=row['sensor_id'],
                    location_lat=row['location_lat'],
                    location_lon=row['location_lon'],
                    turbidity=row['turbidity'],
                    ph_level=row['ph_level'],
                    temperature=row['temperature'],
                    dissolved_oxygen=row['dissolved_oxygen'],
                    bacterial_count=row['bacterial_count'],
                    chlorine_residual=row['chlorine_residual'],
                    conductivity=row['conductivity'],
                    total_dissolved_solids=row['total_dissolved_solids'],
                    nitrate_level=row['nitrate_level'],
                    phosphate_level=row['phosphate_level'],
                    is_contaminated=is_contaminated,
                    contamination_level=contamination_level,
                    measurement_timestamp=row['measurement_timestamp']
                )
                db.add(water_data)
            
            # Commit all changes
            await db.commit()
            
            print(f"Successfully loaded {len(health_reports)} health reports")
            print(f"Successfully loaded {len(water_quality)} water quality records")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            await db.rollback()
            raise

if __name__ == "__main__":
    asyncio.run(load_sample_data())
