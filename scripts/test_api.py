"""
Test script for the Health Surveillance API
"""
import requests
import json
import time
from datetime import datetime

# API base URL
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test health check endpoint"""
    print("Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✓ Health check passed")
            return True
        else:
            print(f"✗ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False

def test_database_health():
    """Test database health check"""
    print("Testing database health...")
    try:
        response = requests.get(f"{BASE_URL}/health/db")
        if response.status_code == 200:
            print("✓ Database health check passed")
            return True
        else:
            print(f"✗ Database health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Database health check failed: {e}")
        return False

def test_health_report_submission():
    """Test health report submission"""
    print("Testing health report submission...")
    try:
        health_report = {
            "report_id": f"TEST_{int(time.time())}",
            "user_id": "TEST_USER_001",
            "location_lat": 28.6139,
            "location_lon": 77.2090,
            "location_address": "Test Location, Delhi",
            "fever": True,
            "diarrhea": True,
            "vomiting": False,
            "nausea": True,
            "abdominal_pain": True,
            "dehydration": False,
            "symptom_severity": 4,
            "report_source": "test"
        }
        
        response = requests.post(
            f"{BASE_URL}/api/v1/data/health-reports",
            json=health_report
        )
        
        if response.status_code == 200:
            print("✓ Health report submission passed")
            return True
        else:
            print(f"✗ Health report submission failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"✗ Health report submission failed: {e}")
        return False

def test_water_quality_submission():
    """Test water quality data submission"""
    print("Testing water quality data submission...")
    try:
        water_data = {
            "sensor_id": f"SENSOR_{int(time.time())}",
            "location_lat": 28.6139,
            "location_lon": 77.2090,
            "turbidity": 2.5,
            "ph_level": 7.2,
            "temperature": 25.0,
            "dissolved_oxygen": 8.5,
            "bacterial_count": 45,
            "chlorine_residual": 1.2,
            "conductivity": 500,
            "total_dissolved_solids": 250,
            "nitrate_level": 2.0,
            "phosphate_level": 0.5
        }
        
        response = requests.post(
            f"{BASE_URL}/api/v1/data/water-quality",
            json=water_data
        )
        
        if response.status_code == 200:
            print("✓ Water quality data submission passed")
            return True
        else:
            print(f"✗ Water quality data submission failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"✗ Water quality data submission failed: {e}")
        return False

def test_prediction():
    """Test outbreak prediction"""
    print("Testing outbreak prediction...")
    try:
        prediction_request = {
            "location_lat": 28.6139,
            "location_lon": 77.2090,
            "prediction_horizon_days": 7
        }
        
        response = requests.post(
            f"{BASE_URL}/api/v1/predict/outbreak",
            json=prediction_request
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✓ Outbreak prediction passed")
            print(f"  Prediction ID: {result.get('prediction_id', 'N/A')}")
            print(f"  Data Summary: {result.get('data_summary', {})}")
            return True
        else:
            print(f"✗ Outbreak prediction failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"✗ Outbreak prediction failed: {e}")
        return False

def test_latest_predictions():
    """Test getting latest predictions"""
    print("Testing latest predictions retrieval...")
    try:
        response = requests.get(f"{BASE_URL}/api/v1/predict/latest")
        
        if response.status_code == 200:
            result = response.json()
            print("✓ Latest predictions retrieval passed")
            print(f"  Outbreaks: {len(result.get('outbreaks', []))}")
            print(f"  Hotspots: {len(result.get('hotspots', []))}")
            return True
        else:
            print(f"✗ Latest predictions retrieval failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Latest predictions retrieval failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("Health Surveillance API Test Suite")
    print("=" * 50)
    
    tests = [
        test_health_check,
        test_database_health,
        test_health_report_submission,
        test_water_quality_submission,
        test_prediction,
        test_latest_predictions
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    print("=" * 50)
    
    if passed == total:
        print("🎉 All tests passed! The API is working correctly.")
    else:
        print("❌ Some tests failed. Please check the logs above.")

if __name__ == "__main__":
    main()
