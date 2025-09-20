"""
Airflow DAG for Smart Health Surveillance Pipeline
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.http.operators.http import SimpleHttpOperator
from airflow.sensors.filesystem import FileSensor
import pandas as pd
import requests
import logging

# Default arguments
default_args = {
    'owner': 'health-surveillance-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'catchup': False
}

# Create DAG
dag = DAG(
    'health_surveillance_pipeline',
    default_args=default_args,
    description='Smart Health Surveillance and Early Warning System Pipeline',
    schedule_interval='@hourly',  # Run every hour
    max_active_runs=1,
    tags=['health', 'surveillance', 'ml', 'outbreak-prediction']
)

def extract_health_data(**context):
    """Extract health data from various sources"""
    logging.info("Extracting health data...")
    
    # Extract from database
    import psycopg2
    from sqlalchemy import create_engine
    
    # Database connection
    engine = create_engine("postgresql://postgres:password@postgres:5432/health_surveillance")
    
    # Extract recent health reports
    health_query = """
    SELECT * FROM health_reports 
    WHERE created_at >= NOW() - INTERVAL '24 hours'
    """
    health_df = pd.read_sql(health_query, engine)
    
    # Extract recent water quality data
    water_query = """
    SELECT * FROM water_quality_data 
    WHERE created_at >= NOW() - INTERVAL '24 hours'
    """
    water_df = pd.read_sql(water_query, engine)
    
    # Save to XCom for next task
    context['task_instance'].xcom_push(key='health_data', value=health_df.to_json())
    context['task_instance'].xcom_push(key='water_data', value=water_df.to_json())
    
    logging.info(f"Extracted {len(health_df)} health reports and {len(water_df)} water quality records")
    return f"Extracted {len(health_df)} health reports and {len(water_df)} water quality records"

def process_and_clean_data(**context):
    """Process and clean the extracted data"""
    logging.info("Processing and cleaning data...")
    
    # Get data from previous task
    health_json = context['task_instance'].xcom_pull(key='health_data')
    water_json = context['task_instance'].xcom_pull(key='water_data')
    
    health_df = pd.read_json(health_json)
    water_df = pd.read_json(water_json)
    
    # Data cleaning and preprocessing
    from app.services.data_processor import DataProcessor
    processor = DataProcessor()
    
    # Clean health data
    if not health_df.empty:
        # Remove duplicates
        health_df = health_df.drop_duplicates()
        
        # Fill missing values
        health_df['symptom_severity'] = health_df['symptom_severity'].fillna(1)
        
        # Clean location data
        health_df = health_df.dropna(subset=['location_lat', 'location_lon'])
    
    # Clean water quality data
    if not water_df.empty:
        # Remove duplicates
        water_df = water_df.drop_duplicates()
        
        # Fill missing values with median
        numeric_columns = water_df.select_dtypes(include=['number']).columns
        water_df[numeric_columns] = water_df[numeric_columns].fillna(water_df[numeric_columns].median())
        
        # Clean location data
        water_df = water_df.dropna(subset=['location_lat', 'location_lon'])
    
    # Save processed data
    context['task_instance'].xcom_push(key='processed_health_data', value=health_df.to_json())
    context['task_instance'].xcom_push(key='processed_water_data', value=water_df.to_json())
    
    logging.info(f"Processed {len(health_df)} health reports and {len(water_df)} water quality records")
    return f"Processed {len(health_df)} health reports and {len(water_df)} water quality records"

def train_ml_models(**context):
    """Train ML models with processed data"""
    logging.info("Training ML models...")
    
    # Get processed data
    health_json = context['task_instance'].xcom_pull(key='processed_health_data')
    water_json = context['task_instance'].xcom_pull(key='processed_water_data')
    
    health_df = pd.read_json(health_json)
    water_df = pd.read_json(water_json)
    
    if health_df.empty and water_df.empty:
        logging.warning("No data available for model training")
        return "No data available for training"
    
    # Train models
    from app.ml.model_manager import ModelManager
    model_manager = ModelManager()
    
    training_data = {
        'health_reports': health_df,
        'water_quality': water_df
    }
    
    try:
        # Train models
        results = await model_manager.train_all_models(training_data)
        
        # Save model performance metrics
        context['task_instance'].xcom_push(key='model_performance', value=results)
        
        logging.info("ML models trained successfully")
        return "ML models trained successfully"
        
    except Exception as e:
        logging.error(f"Model training failed: {e}")
        raise

def make_predictions(**context):
    """Make predictions using trained models"""
    logging.info("Making predictions...")
    
    # Get processed data
    health_json = context['task_instance'].xcom_pull(key='processed_health_data')
    water_json = context['task_instance'].xcom_pull(key='processed_water_data')
    
    health_df = pd.read_json(health_json)
    water_df = pd.read_json(water_json)
    
    if health_df.empty and water_df.empty:
        logging.warning("No data available for predictions")
        return "No data available for predictions"
    
    # Load trained models
    from app.ml.model_manager import ModelManager
    model_manager = ModelManager()
    await model_manager.load_all_models()
    
    try:
        # Make predictions
        predictions = await model_manager.predict_outbreak(health_df, water_df)
        
        # Save predictions
        context['task_instance'].xcom_push(key='predictions', value=predictions)
        
        logging.info("Predictions made successfully")
        return "Predictions made successfully"
        
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        raise

def save_predictions_to_db(**context):
    """Save predictions to database"""
    logging.info("Saving predictions to database...")
    
    predictions = context['task_instance'].xcom_pull(key='predictions')
    
    if not predictions:
        logging.warning("No predictions to save")
        return "No predictions to save"
    
    # Save to database
    import psycopg2
    from sqlalchemy import create_engine
    
    engine = create_engine("postgresql://postgres:password@postgres:5432/health_surveillance")
    
    # Create outbreak records
    outbreak_data = {
        'outbreak_id': f"outbreak_{datetime.now().timestamp()}",
        'location_lat': 0.0,  # Default location
        'location_lon': 0.0,
        'predicted_disease': predictions.get('predicted_disease', 'Unknown'),
        'outbreak_probability': predictions.get('outbreak_probability', 0.0),
        'severity_level': 'low' if predictions.get('outbreak_probability', 0) < 0.5 else 'high',
        'confidence_score': predictions.get('confidence_level', 0.0),
        'contributing_factors': predictions.get('contributing_factors', []),
        'model_version': '1.0',
        'prediction_timestamp': datetime.now()
    }
    
    # Insert into database
    outbreak_df = pd.DataFrame([outbreak_data])
    outbreak_df.to_sql('disease_outbreaks', engine, if_exists='append', index=False)
    
    logging.info("Predictions saved to database")
    return "Predictions saved to database"

def send_alerts(**context):
    """Send alerts for high-risk predictions"""
    logging.info("Checking for alerts...")
    
    predictions = context['task_instance'].xcom_pull(key='predictions')
    
    if not predictions:
        return "No predictions to check for alerts"
    
    outbreak_probability = predictions.get('outbreak_probability', 0)
    
    if outbreak_probability > 0.7:
        # High risk - send alert
        alert_message = f"HIGH RISK ALERT: Outbreak probability {outbreak_probability:.2f}"
        logging.warning(alert_message)
        
        # Here you would integrate with your alerting system
        # For example, send email, SMS, or webhook notification
        
        return f"Alert sent: {alert_message}"
    else:
        return f"No alert needed. Risk level: {outbreak_probability:.2f}"

# Define tasks
extract_task = PythonOperator(
    task_id='extract_health_data',
    python_callable=extract_health_data,
    dag=dag
)

process_task = PythonOperator(
    task_id='process_and_clean_data',
    python_callable=process_and_clean_data,
    dag=dag
)

train_models_task = PythonOperator(
    task_id='train_ml_models',
    python_callable=train_ml_models,
    dag=dag
)

predict_task = PythonOperator(
    task_id='make_predictions',
    python_callable=make_predictions,
    dag=dag
)

save_predictions_task = PythonOperator(
    task_id='save_predictions_to_db',
    python_callable=save_predictions_to_db,
    dag=dag
)

alert_task = PythonOperator(
    task_id='send_alerts',
    python_callable=send_alerts,
    dag=dag
)

# Define task dependencies
extract_task >> process_task >> train_models_task >> predict_task >> save_predictions_task >> alert_task
