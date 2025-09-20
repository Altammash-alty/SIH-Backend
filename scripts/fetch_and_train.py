"""
Script to fetch real data from news articles and government sources, then train the models
"""
import asyncio
import pandas as pd
import sys
import os
from pathlib import Path
import logging
from datetime import datetime, timedelta

# Add the app directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from app.services.data_fetcher import DataFetcher
from app.ml.model_manager import ModelManager
from app.core.database import init_db, SessionLocal
from app.models.health_data import HealthReport, WaterQualityData
from app.services.data_processor import DataProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def fetch_real_data():
    """Fetch real data from various sources"""
    logger.info("Starting data fetching from external sources...")
    
    data_fetcher = DataFetcher()
    
    # Fetch news articles
    logger.info("Fetching news articles...")
    articles = await data_fetcher.fetch_news_articles(days_back=30)
    logger.info(f"Fetched {len(articles)} news articles")
    
    # Fetch government water quality data
    logger.info("Fetching government water quality data...")
    water_data = await data_fetcher.fetch_government_water_data()
    logger.info(f"Fetched {len(water_data)} water quality records")
    
    # Process the fetched data
    logger.info("Processing fetched data...")
    processed_data = data_fetcher.process_fetched_data(articles, water_data)
    
    return processed_data

async def save_data_to_database(processed_data):
    """Save processed data to database"""
    logger.info("Saving data to database...")
    
    await init_db()
    
    async with SessionLocal() as db:
        try:
            # Save health reports
            if not processed_data['health_reports'].empty:
                logger.info(f"Saving {len(processed_data['health_reports'])} health reports...")
                
                for _, row in processed_data['health_reports'].iterrows():
                    health_report = HealthReport(
                        report_id=row['report_id'],
                        user_id=row['user_id'],
                        location_lat=row['location_lat'],
                        location_lon=row['location_lon'],
                        location_address=row.get('location_address', ''),
                        fever=row['fever'],
                        diarrhea=row['diarrhea'],
                        vomiting=row['vomiting'],
                        nausea=row['nausea'],
                        abdominal_pain=row['abdominal_pain'],
                        dehydration=row['dehydration'],
                        other_symptoms=row.get('other_symptoms'),
                        symptom_severity=row['symptom_severity'],
                        report_source=row['report_source'],
                        report_timestamp=row['report_timestamp']
                    )
                    db.add(health_report)
                
                logger.info("Health reports saved successfully")
            
            # Save water quality data
            if not processed_data['water_quality'].empty:
                logger.info(f"Saving {len(processed_data['water_quality'])} water quality records...")
                
                for _, row in processed_data['water_quality'].iterrows():
                    # Determine contamination level
                    contamination_level = "low"
                    is_contaminated = False
                    
                    if row.get('bacterial_count', 0) > 100:
                        contamination_level = "high"
                        is_contaminated = True
                    elif row.get('turbidity', 0) > 4:
                        contamination_level = "medium"
                        is_contaminated = True
                    
                    water_quality = WaterQualityData(
                        sensor_id=row.get('sensor_id', f"GOVT_{hash(str(row))}"),
                        location_lat=row.get('latitude', 0.0),
                        location_lon=row.get('longitude', 0.0),
                        turbidity=row.get('turbidity'),
                        ph_level=row.get('ph'),
                        temperature=row.get('temperature'),
                        dissolved_oxygen=row.get('dissolved_oxygen'),
                        bacterial_count=row.get('bacterial_count'),
                        chlorine_residual=row.get('chlorine'),
                        conductivity=row.get('conductivity'),
                        total_dissolved_solids=row.get('total_dissolved_solids'),
                        nitrate_level=row.get('nitrate'),
                        phosphate_level=row.get('phosphate'),
                        is_contaminated=is_contaminated,
                        contamination_level=contamination_level,
                        measurement_timestamp=datetime.now()
                    )
                    db.add(water_quality)
                
                logger.info("Water quality data saved successfully")
            
            await db.commit()
            logger.info("All data saved to database successfully")
            
        except Exception as e:
            logger.error(f"Error saving data to database: {e}")
            await db.rollback()
            raise

async def train_models_with_real_data(processed_data):
    """Train models with the fetched real data"""
    logger.info("Training models with real data...")
    
    # Initialize model manager
    model_manager = ModelManager()
    
    # Prepare training data
    training_data = {
        'health_reports': processed_data['health_reports'],
        'water_quality': processed_data['water_quality']
    }
    
    # Add some synthetic data if we don't have enough real data
    if len(training_data['health_reports']) < 50 or len(training_data['water_quality']) < 50:
        logger.info("Adding synthetic data to supplement real data...")
        from app.utils.sample_data_generator import SampleDataGenerator
        
        data_generator = SampleDataGenerator()
        synthetic_scenario = data_generator.generate_outbreak_scenario()
        
        # Combine real and synthetic data
        if training_data['health_reports'].empty:
            training_data['health_reports'] = synthetic_scenario['health_reports']
        else:
            training_data['health_reports'] = pd.concat([
                training_data['health_reports'],
                synthetic_scenario['health_reports']
            ], ignore_index=True)
        
        if training_data['water_quality'].empty:
            training_data['water_quality'] = synthetic_scenario['water_quality']
        else:
            training_data['water_quality'] = pd.concat([
                training_data['water_quality'],
                synthetic_scenario['water_quality']
            ], ignore_index=True)
    
    logger.info(f"Training with {len(training_data['health_reports'])} health reports and {len(training_data['water_quality'])} water quality records")
    
    try:
        # Train all models
        results = await model_manager.train_all_models(training_data)
        
        logger.info("Model training completed successfully!")
        
        # Print training results
        for model_name, result in results.items():
            logger.info(f"\n{model_name} Training Results:")
            if isinstance(result, dict) and 'performance_metrics' in result:
                metrics = result['performance_metrics']
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, dict):
                        logger.info(f"  {metric_name}:")
                        for sub_metric, sub_value in metric_value.items():
                            logger.info(f"    {sub_metric}: {sub_value:.4f}")
                    else:
                        logger.info(f"  {metric_name}: {metric_value}")
            else:
                logger.info(f"  Status: {result}")
        
        # Test prediction with real data
        logger.info("\nTesting prediction with real data...")
        test_prediction = await model_manager.predict_outbreak(
            training_data['health_reports'].head(10),
            training_data['water_quality'].head(10)
        )
        
        logger.info("Test prediction completed!")
        logger.info(f"Outbreak probability: {test_prediction.get('outbreak_probability', 'N/A')}")
        logger.info(f"Predicted disease: {test_prediction.get('predicted_disease', 'N/A')}")
        logger.info(f"Risk score: {test_prediction.get('risk_score', 'N/A')}")
        
        if 'hotspot_info' in test_prediction:
            hotspots = test_prediction['hotspot_info'].get('hotspots', [])
            logger.info(f"Identified {len(hotspots)} disease hotspots")
        
        if 'recommendations' in test_prediction:
            recommendations = test_prediction['recommendations']
            logger.info(f"Generated {len(recommendations)} recommendations")
            for i, rec in enumerate(recommendations, 1):
                logger.info(f"  {i}. {rec}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise

async def main():
    """Main function to fetch data and train models"""
    logger.info("=" * 60)
    logger.info("Smart Health Surveillance - Real Data Training")
    logger.info("=" * 60)
    
    try:
        # Step 1: Fetch real data from external sources
        processed_data = await fetch_real_data()
        
        # Step 2: Save data to database
        await save_data_to_database(processed_data)
        
        # Step 3: Train models with real data
        training_results = await train_models_with_real_data(processed_data)
        
        logger.info("=" * 60)
        logger.info("Training completed successfully!")
        logger.info("=" * 60)
        
        # Summary
        logger.info("\nData Summary:")
        logger.info(f"- Health reports: {len(processed_data['health_reports'])}")
        logger.info(f"- Water quality records: {len(processed_data['water_quality'])}")
        logger.info(f"- News insights: {len(processed_data['news_insights'])}")
        
        logger.info("\nNext steps:")
        logger.info("1. Start the API server: docker-compose up app")
        logger.info("2. Test predictions: python scripts/test_api.py")
        logger.info("3. View results at: http://localhost:8000/docs")
        
    except Exception as e:
        logger.error(f"Error in main process: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
