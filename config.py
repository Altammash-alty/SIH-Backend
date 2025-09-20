import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Database Configuration
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/health_surveillance")
    MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://admin:password@localhost:27017/health_reports?authSource=admin")
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # API Configuration
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", 8000))
    API_DEBUG = os.getenv("API_DEBUG", "True").lower() == "true"
    
    # Model Configuration
    MODEL_PATH = os.getenv("MODEL_PATH", "./models")
    MODEL_RETRAIN_INTERVAL = int(os.getenv("MODEL_RETRAIN_INTERVAL", 86400))  # 24 hours
    
    # External APIs (Optional)
    WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "")
    HEALTH_API_KEY = os.getenv("HEALTH_API_KEY", "")
    
    # Security
    SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
    ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30))
    
    # Data Sources
    SMS_GATEWAY_URL = os.getenv("SMS_GATEWAY_URL", "https://api.smsgateway.com")
    IOT_SENSOR_ENDPOINT = os.getenv("IOT_SENSOR_ENDPOINT", "https://sensors.health.gov")
