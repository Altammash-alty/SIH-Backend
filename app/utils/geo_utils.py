"""
Geospatial utilities for distance calculations and location processing
"""
import math
from typing import Tuple, List, Dict, Any

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points using Haversine formula"""
    R = 6371  # Earth's radius in kilometers
    
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    
    a = (math.sin(dlat/2) * math.sin(dlat/2) + 
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
         math.sin(dlon/2) * math.sin(dlon/2))
    
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c
    
    return distance

def find_nearby_points(center_lat: float, center_lon: float, 
                      points: List[Dict[str, Any]], radius_km: float) -> List[Dict[str, Any]]:
    """Find points within a given radius of a center point"""
    nearby_points = []
    
    for point in points:
        distance = calculate_distance(
            center_lat, center_lon,
            point['lat'], point['lon']
        )
        
        if distance <= radius_km:
            point['distance_km'] = distance
            nearby_points.append(point)
    
    return nearby_points

def calculate_bounding_box(center_lat: float, center_lon: float, radius_km: float) -> Tuple[float, float, float, float]:
    """Calculate bounding box for a given center point and radius"""
    # Approximate conversion: 1 degree latitude ≈ 111 km
    lat_delta = radius_km / 111.0
    
    # Longitude delta depends on latitude
    lon_delta = radius_km / (111.0 * math.cos(math.radians(center_lat)))
    
    min_lat = center_lat - lat_delta
    max_lat = center_lat + lat_delta
    min_lon = center_lon - lon_delta
    max_lon = center_lon + lon_delta
    
    return min_lat, min_lon, max_lat, max_lon

def calculate_cluster_center(points: List[Dict[str, Any]]) -> Tuple[float, float]:
    """Calculate the center point of a cluster of points"""
    if not points:
        return 0.0, 0.0
    
    total_lat = sum(point['lat'] for point in points)
    total_lon = sum(point['lon'] for point in points)
    
    center_lat = total_lat / len(points)
    center_lon = total_lon / len(points)
    
    return center_lat, center_lon

def calculate_cluster_radius(points: List[Dict[str, Any]], center_lat: float, center_lon: float) -> float:
    """Calculate the radius of a cluster of points"""
    if not points:
        return 0.0
    
    max_distance = 0.0
    for point in points:
        distance = calculate_distance(center_lat, center_lon, point['lat'], point['lon'])
        max_distance = max(max_distance, distance)
    
    return max_distance
