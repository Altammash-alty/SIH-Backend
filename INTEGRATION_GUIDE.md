# Frontend Integration Guide

This guide explains how to integrate the Smart Health Surveillance API with your frontend application.

## 🔌 API Integration

### Base URL
```
http://localhost:8000  # Development
https://your-domain.com  # Production
```

### Authentication
Currently, the API doesn't require authentication. For production, implement JWT or API key authentication.

## 📊 Real-time Data Integration

### 1. Fetch Latest Predictions
```javascript
async function fetchLatestPredictions() {
  try {
    const response = await fetch('/api/v1/predict/latest');
    const data = await response.json();
    
    // Update your UI with predictions
    updateOutbreakMap(data.outbreaks);
    updateHotspotClusters(data.hotspots);
    updateSummaryStats(data.summary);
  } catch (error) {
    console.error('Failed to fetch predictions:', error);
  }
}

// Fetch every 5 minutes
setInterval(fetchLatestPredictions, 5 * 60 * 1000);
```

### 2. Submit Health Reports
```javascript
async function submitHealthReport(reportData) {
  try {
    const response = await fetch('/api/v1/data/health-reports', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        report_id: `RPT_${Date.now()}`,
        user_id: getCurrentUserId(),
        location_lat: reportData.latitude,
        location_lon: reportData.longitude,
        location_address: reportData.address,
        fever: reportData.symptoms.fever,
        diarrhea: reportData.symptoms.diarrhea,
        vomiting: reportData.symptoms.vomiting,
        nausea: reportData.symptoms.nausea,
        abdominal_pain: reportData.symptoms.abdominalPain,
        dehydration: reportData.symptoms.dehydration,
        symptom_severity: reportData.severity,
        report_source: 'mobile_app'
      })
    });
    
    if (response.ok) {
      showSuccessMessage('Health report submitted successfully');
    } else {
      throw new Error('Failed to submit report');
    }
  } catch (error) {
    console.error('Error submitting health report:', error);
    showErrorMessage('Failed to submit health report');
  }
}
```

### 3. Submit Water Quality Data
```javascript
async function submitWaterQualityData(sensorData) {
  try {
    const response = await fetch('/api/v1/data/water-quality', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        sensor_id: sensorData.sensorId,
        location_lat: sensorData.latitude,
        location_lon: sensorData.longitude,
        turbidity: sensorData.turbidity,
        ph_level: sensorData.phLevel,
        temperature: sensorData.temperature,
        dissolved_oxygen: sensorData.dissolvedOxygen,
        bacterial_count: sensorData.bacterialCount,
        chlorine_residual: sensorData.chlorineResidual,
        conductivity: sensorData.conductivity,
        total_dissolved_solids: sensorData.totalDissolvedSolids,
        nitrate_level: sensorData.nitrateLevel,
        phosphate_level: sensorData.phosphateLevel
      })
    });
    
    if (response.ok) {
      const result = await response.json();
      console.log('Water quality data submitted:', result);
    }
  } catch (error) {
    console.error('Error submitting water quality data:', error);
  }
}
```

## 🗺️ Map Integration

### Leaflet.js Integration
```javascript
// Initialize map
const map = L.map('map').setView([28.6139, 77.2090], 10);

// Add tile layer
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  attribution: '© OpenStreetMap contributors'
}).addTo(map);

// Add outbreak markers
function addOutbreakMarkers(outbreaks) {
  outbreaks.forEach(outbreak => {
    const marker = L.circleMarker(
      [outbreak.location.lat, outbreak.location.lon],
      {
        radius: getRadiusBySeverity(outbreak.severity_level),
        color: getColorBySeverity(outbreak.severity_level),
        fillColor: getColorBySeverity(outbreak.severity_level),
        fillOpacity: 0.7
      }
    );
    
    marker.bindPopup(`
      <div class="outbreak-popup">
        <h3>${outbreak.predicted_disease}</h3>
        <p><strong>Probability:</strong> ${(outbreak.outbreak_probability * 100).toFixed(1)}%</p>
        <p><strong>Severity:</strong> ${outbreak.severity_level}</p>
        <p><strong>Confidence:</strong> ${outbreak.confidence_score}</p>
        <p><strong>Location:</strong> ${outbreak.location.name || 'Unknown'}</p>
      </div>
    `);
    
    marker.addTo(map);
  });
}

// Add hotspot clusters
function addHotspotClusters(hotspots) {
  hotspots.forEach(hotspot => {
    const circle = L.circle(
      [hotspot.center_lat, hotspot.center_lon],
      {
        radius: hotspot.radius_km * 1000, // Convert to meters
        color: getColorByRiskLevel(hotspot.risk_level),
        fillColor: getColorByRiskLevel(hotspot.risk_level),
        fillOpacity: 0.3
      }
    );
    
    circle.bindPopup(`
      <div class="hotspot-popup">
        <h3>Disease Hotspot</h3>
        <p><strong>Cases:</strong> ${hotspot.case_count}</p>
        <p><strong>Density:</strong> ${hotspot.density.toFixed(2)} cases/km²</p>
        <p><strong>Risk Level:</strong> ${hotspot.risk_level}</p>
        <p><strong>Risk Score:</strong> ${hotspot.risk_score}</p>
      </div>
    `);
    
    circle.addTo(map);
  });
}

// Helper functions
function getRadiusBySeverity(severity) {
  const sizes = { low: 8, medium: 12, high: 16, critical: 20 };
  return sizes[severity] || 8;
}

function getColorBySeverity(severity) {
  const colors = { 
    low: '#4CAF50', 
    medium: '#FF9800', 
    high: '#F44336', 
    critical: '#9C27B0' 
  };
  return colors[severity] || '#4CAF50';
}

function getColorByRiskLevel(riskLevel) {
  const colors = { 
    low: '#4CAF50', 
    medium: '#FF9800', 
    high: '#F44336', 
    critical: '#9C27B0' 
  };
  return colors[riskLevel] || '#4CAF50';
}
```

## 📱 Mobile App Integration

### React Native Example
```javascript
import React, { useState, useEffect } from 'react';
import { View, Text, Button, Alert } from 'react-native';

const HealthReportScreen = () => {
  const [symptoms, setSymptoms] = useState({
    fever: false,
    diarrhea: false,
    vomiting: false,
    nausea: false,
    abdominalPain: false,
    dehydration: false
  });
  const [severity, setSeverity] = useState(1);
  const [location, setLocation] = useState(null);

  const submitReport = async () => {
    try {
      const response = await fetch('http://your-api-domain.com/api/v1/data/health-reports', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          report_id: `RPT_${Date.now()}`,
          user_id: 'USER_123',
          location_lat: location.latitude,
          location_lon: location.longitude,
          fever: symptoms.fever,
          diarrhea: symptoms.diarrhea,
          vomiting: symptoms.vomiting,
          nausea: symptoms.nausea,
          abdominal_pain: symptoms.abdominalPain,
          dehydration: symptoms.dehydration,
          symptom_severity: severity,
          report_source: 'mobile_app'
        })
      });

      if (response.ok) {
        Alert.alert('Success', 'Health report submitted successfully');
      } else {
        Alert.alert('Error', 'Failed to submit health report');
      }
    } catch (error) {
      Alert.alert('Error', 'Network error occurred');
    }
  };

  return (
    <View>
      {/* Your UI components here */}
      <Button title="Submit Report" onPress={submitReport} />
    </View>
  );
};
```

## 🔔 Real-time Notifications

### WebSocket Integration (Future Enhancement)
```javascript
// WebSocket connection for real-time updates
const ws = new WebSocket('ws://localhost:8000/ws/predictions');

ws.onopen = () => {
  console.log('Connected to prediction updates');
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  if (data.type === 'outbreak_alert') {
    showOutbreakAlert(data.outbreak);
  } else if (data.type === 'hotspot_update') {
    updateHotspotMap(data.hotspots);
  }
};

ws.onclose = () => {
  console.log('Disconnected from prediction updates');
  // Implement reconnection logic
};
```

## 📊 Dashboard Integration

### Chart.js Integration
```javascript
// Outbreak trend chart
function createOutbreakTrendChart(predictions) {
  const ctx = document.getElementById('outbreakTrendChart').getContext('2d');
  
  const chart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: predictions.map(p => p.date),
      datasets: [{
        label: 'Outbreak Probability',
        data: predictions.map(p => p.probability * 100),
        borderColor: '#F44336',
        backgroundColor: 'rgba(244, 67, 54, 0.1)',
        tension: 0.4
      }]
    },
    options: {
      responsive: true,
      scales: {
        y: {
          beginAtZero: true,
          max: 100,
          title: {
            display: true,
            text: 'Probability (%)'
          }
        },
        x: {
          title: {
            display: true,
            text: 'Date'
          }
        }
      }
    }
  });
}

// Risk level distribution pie chart
function createRiskDistributionChart(hotspots) {
  const ctx = document.getElementById('riskDistributionChart').getContext('2d');
  
  const riskLevels = ['low', 'medium', 'high', 'critical'];
  const counts = riskLevels.map(level => 
    hotspots.filter(h => h.risk_level === level).length
  );
  
  const chart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: riskLevels,
      datasets: [{
        data: counts,
        backgroundColor: [
          '#4CAF50',
          '#FF9800',
          '#F44336',
          '#9C27B0'
        ]
      }]
    },
    options: {
      responsive: true,
      plugins: {
        legend: {
          position: 'bottom'
        }
      }
    }
  });
}
```

## 🔧 Error Handling

### API Error Handling
```javascript
class HealthSurveillanceAPI {
  constructor(baseURL) {
    this.baseURL = baseURL;
  }

  async request(endpoint, options = {}) {
    try {
      const response = await fetch(`${this.baseURL}${endpoint}`, {
        headers: {
          'Content-Type': 'application/json',
          ...options.headers
        },
        ...options
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new APIError(response.status, errorData.detail || 'API request failed');
      }

      return await response.json();
    } catch (error) {
      if (error instanceof APIError) {
        throw error;
      }
      throw new NetworkError('Network request failed');
    }
  }

  async getHealthReports(limit = 100, offset = 0) {
    return this.request(`/api/v1/data/health-reports?limit=${limit}&offset=${offset}`);
  }

  async submitHealthReport(reportData) {
    return this.request('/api/v1/data/health-reports', {
      method: 'POST',
      body: JSON.stringify(reportData)
    });
  }

  async getLatestPredictions() {
    return this.request('/api/v1/predict/latest');
  }
}

class APIError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
    this.name = 'APIError';
  }
}

class NetworkError extends Error {
  constructor(message) {
    super(message);
    this.name = 'NetworkError';
  }
}

// Usage
const api = new HealthSurveillanceAPI('http://localhost:8000');

try {
  const predictions = await api.getLatestPredictions();
  updateDashboard(predictions);
} catch (error) {
  if (error instanceof APIError) {
    console.error(`API Error ${error.status}: ${error.message}`);
  } else if (error instanceof NetworkError) {
    console.error('Network Error:', error.message);
  }
}
```

## 🚀 Production Considerations

### Environment Configuration
```javascript
const config = {
  development: {
    apiUrl: 'http://localhost:8000',
    wsUrl: 'ws://localhost:8000'
  },
  production: {
    apiUrl: 'https://api.healthsurveillance.com',
    wsUrl: 'wss://api.healthsurveillance.com'
  }
};

const environment = process.env.NODE_ENV || 'development';
const { apiUrl, wsUrl } = config[environment];
```

### Caching Strategy
```javascript
// Simple in-memory cache
const cache = new Map();

async function getCachedPredictions() {
  const cacheKey = 'latest_predictions';
  const cached = cache.get(cacheKey);
  
  if (cached && Date.now() - cached.timestamp < 5 * 60 * 1000) {
    return cached.data;
  }
  
  const data = await api.getLatestPredictions();
  cache.set(cacheKey, {
    data,
    timestamp: Date.now()
  });
  
  return data;
}
```

This integration guide provides comprehensive examples for integrating the Health Surveillance API with various frontend technologies. Choose the examples that best fit your technology stack and requirements.
