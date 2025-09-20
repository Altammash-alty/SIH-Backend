"""
Comprehensive script to fetch data from multiple sources and train models
"""
import asyncio
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add the app directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from app.services.data_fetcher import DataFetcher
from scripts.fetch_northeast_data import NortheastDataFetcher
from app.ml.model_manager import ModelManager
from app.core.database import init_db, SessionLocal
from app.models.health_data import HealthReport, WaterQualityData
from app.utils.sample_data_generator import SampleDataGenerator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_fetching.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ComprehensiveDataFetcher:
    """Comprehensive data fetcher that combines multiple sources"""
    
    def __init__(self):
        self.general_fetcher = DataFetcher()
        self.northeast_fetcher = NortheastDataFetcher()
        self.data_generator = SampleDataGenerator()
    
    async def fetch_all_data_sources(self) -> dict:
        """Fetch data from all available sources"""
        logger.info("Starting comprehensive data fetching...")
        
        all_data = {
            'health_reports': [],
            'water_quality': [],
            'news_insights': []
        }
        
        try:
            # 1. Fetch general news articles
            logger.info("Fetching general news articles...")
            general_articles = await self.general_fetcher.fetch_news_articles(days_back=30)
            logger.info(f"Fetched {len(general_articles)} general news articles")
            
            # 2. Fetch Northeast-specific data
            logger.info("Fetching Northeast India data...")
            northeast_articles = await self.northeast_fetcher.fetch_northeast_news()
            northeast_water = await self.northeast_fetcher.fetch_government_water_reports()
            logger.info(f"Fetched {len(northeast_articles)} Northeast articles and {len(northeast_water)} water records")
            
            # 3. Fetch general government water data
            logger.info("Fetching general government water data...")
            general_water = await self.general_fetcher.fetch_government_water_data()
            logger.info(f"Fetched {len(general_water)} general water quality records")
            
            # 4. Process all data
            logger.info("Processing all fetched data...")
            
            # Process general articles
            general_processed = self.general_fetcher.process_fetched_data(general_articles, general_water)
            
            # Process Northeast data
            northeast_processed = self.northeast_fetcher.process_northeast_data(northeast_articles, northeast_water)
            
            # Combine all data
            all_data = self._combine_processed_data(general_processed, northeast_processed)
            
            # Add synthetic data to ensure we have enough training data
            logger.info("Adding synthetic data to supplement real data...")
            synthetic_scenario = self.data_generator.generate_outbreak_scenario()
            
            all_data = self._add_synthetic_data(all_data, synthetic_scenario)
            
            logger.info(f"Total data collected:")
            logger.info(f"- Health reports: {len(all_data['health_reports'])}")
            logger.info(f"- Water quality records: {len(all_data['water_quality'])}")
            logger.info(f"- News insights: {len(all_data['news_insights'])}")
            
            return all_data
            
        except Exception as e:
            logger.error(f"Error in comprehensive data fetching: {e}")
            # Fallback to synthetic data only
            logger.info("Falling back to synthetic data...")
            synthetic_scenario = self.data_generator.generate_outbreak_scenario()
            return {
                'health_reports': synthetic_scenario['health_reports'],
                'water_quality': synthetic_scenario['water_quality'],
                'news_insights': []
            }
    
    def _combine_processed_data(self, general_data: dict, northeast_data: dict) -> dict:
        """Combine data from different sources"""
        import pandas as pd
        
        combined = {
            'health_reports': pd.DataFrame(),
            'water_quality': pd.DataFrame(),
            'news_insights': pd.DataFrame()
        }
        
        # Combine health reports
        health_reports = []
        if not general_data['health_reports'].empty:
            health_reports.append(general_data['health_reports'])
        if not northeast_data['health_reports'].empty:
            health_reports.append(northeast_data['health_reports'])
        
        if health_reports:
            combined['health_reports'] = pd.concat(health_reports, ignore_index=True)
        
        # Combine water quality data
        water_quality = []
        if not general_data['water_quality'].empty:
            water_quality.append(general_data['water_quality'])
        if not northeast_data['water_quality'].empty:
            water_quality.append(northeast_data['water_quality'])
        
        if water_quality:
            combined['water_quality'] = pd.concat(water_quality, ignore_index=True)
        
        # Combine news insights
        news_insights = []
        if not general_data['news_insights'].empty:
            news_insights.append(general_data['news_insights'])
        
        if news_insights:
            combined['news_insights'] = pd.concat(news_insights, ignore_index=True)
        
        return combined
    
    def _add_synthetic_data(self, real_data: dict, synthetic_data: dict) -> dict:
        """Add synthetic data to supplement real data"""
        import pandas as pd
        
        # Ensure we have at least 100 samples of each type
        min_samples = 100
        
        # Add synthetic health reports if needed
        if len(real_data['health_reports']) < min_samples:
            needed = min_samples - len(real_data['health_reports'])
            additional_reports = synthetic_data['health_reports'].head(needed)
            
            if real_data['health_reports'].empty:
                real_data['health_reports'] = additional_reports
            else:
                real_data['health_reports'] = pd.concat([real_data['health_reports'], additional_reports], ignore_index=True)
        
        # Add synthetic water quality data if needed
        if len(real_data['water_quality']) < min_samples:
            needed = min_samples - len(real_data['water_quality'])
            additional_water = synthetic_data['water_quality'].head(needed)
            
            if real_data['water_quality'].empty:
                real_data['water_quality'] = additional_water
            else:
                real_data['water_quality'] = pd.concat([real_data['water_quality'], additional_water], ignore_index=True)
        
        return real_data
    
    async def save_data_to_database(self, data: dict):
        """Save all data to database"""
        logger.info("Saving data to database...")
        
        await init_db()
        
        async with SessionLocal() as db:
            try:
                # Save health reports
                if not data['health_reports'].empty:
                    logger.info(f"Saving {len(data['health_reports'])} health reports...")
                    
                    for _, row in data['health_reports'].iterrows():
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
                
                # Save water quality data
                if not data['water_quality'].empty:
                    logger.info(f"Saving {len(data['water_quality'])} water quality records...")
                    
                    for _, row in data['water_quality'].iterrows():
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
                            sensor_id=row.get('sensor_id', f"FETCHED_{hash(str(row))}"),
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
                
                await db.commit()
                logger.info("All data saved to database successfully")
                
            except Exception as e:
                logger.error(f"Error saving data to database: {e}")
                await db.rollback()
                raise
    
    async def train_models(self, data: dict):
        """Train models with the fetched data"""
        logger.info("Training ML models with fetched data...")
        
        model_manager = ModelManager()
        
        # Prepare training data
        training_data = {
            'health_reports': data['health_reports'],
            'water_quality': data['water_quality']
        }
        
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
            
            # Test prediction
            logger.info("\nTesting prediction with trained models...")
            test_prediction = await model_manager.predict_outbreak(
                training_data['health_reports'].head(5),
                training_data['water_quality'].head(5)
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
                for i, rec in enumerate(recommendations[:3], 1):  # Show first 3
                    logger.info(f"  {i}. {rec}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise

async def main():
    """Main function"""
    logger.info("=" * 80)
    logger.info("Smart Health Surveillance - Comprehensive Data Fetching & Training")
    logger.info("=" * 80)
    
    try:
        # Initialize comprehensive fetcher
        fetcher = ComprehensiveDataFetcher()
        
        # Step 1: Fetch data from all sources
        data = await fetcher.fetch_all_data_sources()
        
        # Step 2: Save data to database
        await fetcher.save_data_to_database(data)
        
        # Step 3: Train models
        training_results = await fetcher.train_models(data)
        
        logger.info("=" * 80)
        logger.info("COMPREHENSIVE DATA FETCHING AND TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        
        # Final summary
        logger.info(f"\nFinal Data Summary:")
        logger.info(f"- Health reports: {len(data['health_reports'])}")
        logger.info(f"- Water quality records: {len(data['water_quality'])}")
        logger.info(f"- News insights: {len(data['news_insights'])}")
        
        logger.info(f"\nTrained Models:")
        for model_name in training_results.keys():
            logger.info(f"- {model_name}")
        
        logger.info(f"\nNext Steps:")
        logger.info(f"1. Start the API server: docker-compose up app")
        logger.info(f"2. Test the API: python scripts/test_api.py")
        logger.info(f"3. View API documentation: http://localhost:8000/docs")
        logger.info(f"4. Monitor Airflow pipeline: http://localhost:8080")
        
        logger.info(f"\nThe system is now ready to predict water-borne disease outbreaks!")
        
    except Exception as e:
        logger.error(f"Error in main process: {e}")
        logger.error("Check the logs for detailed error information.")
        raise

if __name__ == "__main__":
    asyncio.run(main())
