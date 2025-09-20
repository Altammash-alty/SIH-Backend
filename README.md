# Smart Health Surveillance and Early Warning System

A comprehensive Python-based backend system for predicting water-borne disease outbreaks using machine learning and real-time data processing.

## 🎯 Overview

This system provides early warning capabilities for water-borne disease outbreaks by analyzing:
- Mobile app and SMS-based health reports
- IoT water quality sensor data
- CSV/Excel uploads from local clinics
- Public health and weather reports (optional)

## 🏗️ Architecture

### Tech Stack
- **Backend**: Python 3.11, FastAPI, Uvicorn
- **Databases**: PostgreSQL + TimescaleDB, MongoDB, Redis
- **ML Libraries**: scikit-learn, XGBoost, Prophet, statsmodels
- **Pipeline**: Apache Airflow
- **Deployment**: Docker, Docker Compose

### Database Design
- **PostgreSQL + TimescaleDB**: Structured health and water quality data, time-series analysis
- **MongoDB**: Unstructured text data (SMS reports, NLP processing)
- **Redis**: Caching and session management

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- Python 3.11+ (for local development)

### 1. Clone and Setup
```bash
git clone <repository-url>
cd SIH_Backend
```

### 2. Environment Configuration
Create a `.env` file:
```env
DATABASE_URL=postgresql://postgres:password@localhost:5432/health_surveillance
MONGODB_URL=mongodb://admin:password@localhost:27017/health_reports?authSource=admin
REDIS_URL=redis://localhost:6379
API_HOST=0.0.0.0
API_PORT=8000
SECRET_KEY=your-secret-key-here
```

### 3. Start Services
```bash
# Start all services with Docker Compose
docker-compose up -d

# Or start individual services
docker-compose up postgres mongodb redis
docker-compose up app
docker-compose up airflow-webserver
```

### 4. Load Sample Data
```bash
# Load sample data for testing
python scripts/load_sample_data.py

# Train ML models
python scripts/train_models.py
```

### 5. Access Services
- **API Documentation**: http://localhost:8000/docs
- **Airflow UI**: http://localhost:8080 (admin/admin)
- **Health Check**: http://localhost:8000/health

## 📊 ML Models

### 1. Outbreak Classification
- **Random Forest**: Baseline classification
- **XGBoost**: Gradient boosting for higher accuracy
- **AdaBoost**: Adaptive boosting ensemble

### 2. Anomaly Detection
- **Isolation Forest**: Detect unusual water quality patterns

### 3. Geospatial Clustering
- **KMeans**: Identify disease hotspots
- **DBSCAN**: Density-based clustering for irregular patterns

### 4. Time Series Forecasting
- **ARIMA**: Autoregressive integrated moving average
- **Prophet**: Facebook's forecasting tool for seasonal patterns

### 5. Ensemble Prediction
- Combines all models with weighted voting
- Provides confidence scores and recommendations

## 🔌 API Endpoints

### Data Ingestion
- `POST /api/v1/data/health-reports` - Submit health reports
- `POST /api/v1/data/water-quality` - Submit water quality data
- `POST /api/v1/data/sms-reports` - Process SMS reports
- `POST /api/v1/data/upload-csv` - Upload CSV/Excel files

### Predictions
- `POST /api/v1/predict/outbreak` - Predict outbreak risk
- `POST /api/v1/predict/batch` - Batch predictions
- `GET /api/v1/predict/latest` - Get recent predictions
- `GET /api/v1/predict/outbreaks` - List outbreak predictions
- `GET /api/v1/predict/hotspots` - List disease hotspots

### Health & Monitoring
- `GET /health` - System health check
- `GET /health/db` - Database connectivity
- `GET /api/v1/predict/model-performance` - ML model metrics

## 📈 Data Pipeline

### Airflow DAG: `health_surveillance_pipeline`
1. **Extract**: Pull data from databases and external sources
2. **Process**: Clean, validate, and preprocess data
3. **Train**: Retrain ML models with new data
4. **Predict**: Generate outbreak predictions
5. **Save**: Store results in database
6. **Alert**: Send notifications for high-risk predictions

### Pipeline Schedule
- **Frequency**: Hourly
- **Retry**: 2 attempts with 5-minute delay
- **Monitoring**: Email alerts on failure

## 🗄️ Database Schema

### Core Tables
- `health_reports` - Individual health reports
- `water_quality_data` - Sensor measurements
- `disease_outbreaks` - Predicted outbreaks
- `hotspot_clusters` - Geospatial clusters
- `model_performance` - ML model metrics

### Key Features
- **TimescaleDB**: Optimized for time-series queries
- **Geospatial**: Location-based indexing
- **JSON Fields**: Flexible metadata storage
- **Audit Trails**: Created/updated timestamps

## 🔧 Configuration

### Model Configuration
```python
# app/core/config.py
MODEL_RETRAIN_INTERVAL = 86400  # 24 hours
MODEL_PATH = "./models"
```

### API Configuration
```python
API_HOST = "0.0.0.0"
API_PORT = 8000
API_DEBUG = True
```

### Database Configuration
```python
DATABASE_URL = "postgresql://user:pass@host:port/db"
MONGODB_URL = "mongodb://user:pass@host:port/db"
REDIS_URL = "redis://host:port"
```

## 🧪 Testing

### Load Sample Data
```bash
python scripts/load_sample_data.py
```

### Train Models
```bash
python scripts/train_models.py
```

### Test API
```bash
# Health check
curl http://localhost:8000/health

# Submit health report
curl -X POST "http://localhost:8000/api/v1/data/health-reports" \
  -H "Content-Type: application/json" \
  -d '{
    "report_id": "TEST_001",
    "user_id": "USER_123",
    "location_lat": 28.6139,
    "location_lon": 77.2090,
    "fever": true,
    "diarrhea": true,
    "symptom_severity": 4
  }'

# Make prediction
curl -X POST "http://localhost:8000/api/v1/predict/outbreak" \
  -H "Content-Type: application/json" \
  -d '{
    "location_lat": 28.6139,
    "location_lon": 77.2090,
    "prediction_horizon_days": 7
  }'
```

## 🚀 Deployment

### Production Considerations
1. **Environment Variables**: Use secure secret management
2. **Database Security**: Enable SSL, use strong passwords
3. **API Security**: Implement authentication and rate limiting
4. **Monitoring**: Add logging and metrics collection
5. **Scaling**: Use load balancers and horizontal scaling

### Docker Production
```bash
# Build production image
docker build -t health-surveillance:latest .

# Run with production settings
docker run -d \
  --name health-surveillance \
  -p 8000:8000 \
  -e DATABASE_URL="postgresql://..." \
  -e MONGODB_URL="mongodb://..." \
  health-surveillance:latest
```

## 🔗 Frontend Integration

### WebSocket Support
```javascript
// Real-time predictions
const ws = new WebSocket('ws://localhost:8000/ws/predictions');
ws.onmessage = (event) => {
  const prediction = JSON.parse(event.data);
  updateMap(prediction.hotspots);
  updateAlerts(prediction.outbreaks);
};
```

### REST API Integration
```javascript
// Fetch latest predictions
fetch('/api/v1/predict/latest')
  .then(response => response.json())
  .then(data => {
    displayOutbreaks(data.outbreaks);
    displayHotspots(data.hotspots);
  });
```

### Map Integration
```javascript
// Add outbreak markers to map
data.outbreaks.forEach(outbreak => {
  L.marker([outbreak.location.lat, outbreak.location.lon])
    .addTo(map)
    .bindPopup(`
      <b>${outbreak.predicted_disease}</b><br>
      Probability: ${(outbreak.outbreak_probability * 100).toFixed(1)}%<br>
      Severity: ${outbreak.severity_level}
    `);
});
```

## 📊 Monitoring & Analytics

### Key Metrics
- **Prediction Accuracy**: Model performance over time
- **Response Time**: API latency and throughput
- **Data Quality**: Missing values, validation errors
- **System Health**: Database connections, service status

### Logging
- **Application Logs**: FastAPI request/response logs
- **ML Logs**: Model training and prediction logs
- **Pipeline Logs**: Airflow task execution logs
- **Error Logs**: Exception tracking and debugging

## 🤝 Contributing

### Development Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/

# Code formatting
black app/
flake8 app/
```

### Code Structure
```
app/
├── api/           # FastAPI routes
├── core/          # Configuration and database
├── ml/            # Machine learning models
├── models/        # Database models
├── services/      # Business logic
└── utils/         # Utility functions
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For questions and support:
- Create an issue in the repository
- Check the API documentation at `/docs`
- Review the Airflow logs for pipeline issues

## 🔮 Future Enhancements

- **Deep Learning**: LSTM/GRU for time series
- **Real-time Streaming**: Apache Kafka integration
- **Mobile SDK**: Native mobile app integration
- **Advanced NLP**: BERT for SMS processing
- **Geospatial Analysis**: Advanced clustering algorithms
- **Dashboard**: Real-time monitoring dashboard
