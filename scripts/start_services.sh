#!/bin/bash

# Start Smart Health Surveillance System Services

echo "Starting Smart Health Surveillance System..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running. Please start Docker first."
    exit 1
fi

# Create necessary directories
mkdir -p models
mkdir -p data
mkdir -p airflow/logs
mkdir -p airflow/plugins

# Set permissions
chmod 755 models data
chmod 755 airflow/logs airflow/plugins

# Start services with Docker Compose
echo "Starting services with Docker Compose..."
docker-compose up -d

# Wait for services to be ready
echo "Waiting for services to start..."
sleep 30

# Check service health
echo "Checking service health..."

# Check PostgreSQL
if docker-compose exec -T postgres pg_isready -U postgres > /dev/null 2>&1; then
    echo "✓ PostgreSQL is ready"
else
    echo "✗ PostgreSQL is not ready"
fi

# Check MongoDB
if docker-compose exec -T mongodb mongosh --eval "db.runCommand('ping')" > /dev/null 2>&1; then
    echo "✓ MongoDB is ready"
else
    echo "✗ MongoDB is not ready"
fi

# Check Redis
if docker-compose exec -T redis redis-cli ping > /dev/null 2>&1; then
    echo "✓ Redis is ready"
else
    echo "✗ Redis is not ready"
fi

# Check FastAPI app
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✓ FastAPI application is ready"
else
    echo "✗ FastAPI application is not ready"
fi

# Check Airflow
if curl -s http://localhost:8080 > /dev/null 2>&1; then
    echo "✓ Airflow webserver is ready"
else
    echo "✗ Airflow webserver is not ready"
fi

echo ""
echo "Services started successfully!"
echo ""
echo "Access points:"
echo "- API Documentation: http://localhost:8000/docs"
echo "- Health Check: http://localhost:8000/health"
echo "- Airflow UI: http://localhost:8080 (admin/admin)"
echo ""
echo "Next steps:"
echo "1. Load sample data: python scripts/load_sample_data.py"
echo "2. Train models: python scripts/train_models.py"
echo "3. Test API endpoints using the documentation at /docs"
