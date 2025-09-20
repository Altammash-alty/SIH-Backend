"""
Script to train ML models for the health surveillance system
"""
import asyncio
import pandas as pd
import sys
import os
from pathlib import Path

# Add the app directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from app.ml.model_manager import ModelManager
from app.utils.sample_data_generator import SampleDataGenerator
from app.core.database import init_db
from app.core.config import Config

async def main():
    """Main training function"""
    print("Starting ML model training...")
    
    # Initialize database
    await init_db()
    
    # Generate sample data for training
    print("Generating sample training data...")
    data_generator = SampleDataGenerator()
    scenario = data_generator.generate_outbreak_scenario()
    
    # Prepare training data
    training_data = {
        'health_reports': scenario['health_reports'],
        'water_quality': scenario['water_quality']
    }
    
    print(f"Generated {len(training_data['health_reports'])} health reports")
    print(f"Generated {len(training_data['water_quality'])} water quality records")
    
    # Initialize model manager
    model_manager = ModelManager()
    
    # Train all models
    print("Training ML models...")
    try:
        results = await model_manager.train_all_models(training_data)
        
        print("Model training completed successfully!")
        print("\nTraining Results:")
        for model_name, result in results.items():
            print(f"\n{model_name}:")
            if isinstance(result, dict) and 'performance_metrics' in result:
                metrics = result['performance_metrics']
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, dict):
                        print(f"  {metric_name}:")
                        for sub_metric, sub_value in metric_value.items():
                            print(f"    {sub_metric}: {sub_value:.4f}")
                    else:
                        print(f"  {metric_name}: {metric_value}")
            else:
                print(f"  Status: {result}")
        
        # Test prediction
        print("\nTesting prediction...")
        test_prediction = await model_manager.predict_outbreak(
            training_data['health_reports'].head(10),
            training_data['water_quality'].head(10)
        )
        
        print("Test prediction completed!")
        print(f"Outbreak probability: {test_prediction.get('outbreak_probability', 'N/A')}")
        print(f"Predicted disease: {test_prediction.get('predicted_disease', 'N/A')}")
        print(f"Risk score: {test_prediction.get('risk_score', 'N/A')}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
