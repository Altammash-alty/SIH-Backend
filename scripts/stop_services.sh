#!/bin/bash

# Stop Smart Health Surveillance System Services

echo "Stopping Smart Health Surveillance System..."

# Stop all services
docker-compose down

# Optional: Remove volumes (uncomment if you want to reset all data)
# echo "Removing volumes..."
# docker-compose down -v

echo "Services stopped successfully!"
