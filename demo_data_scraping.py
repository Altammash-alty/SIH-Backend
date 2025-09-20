#!/usr/bin/env python3
"""
Demo script showing data scraping and collection for health surveillance
"""
import requests
import pandas as pd
import json
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import re
import time
import asyncio
import aiohttp
from typing import List, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataScrapingDemo:
    """Demo class for data scraping and collection"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Keywords for health surveillance
        self.health_keywords = [
            'water borne disease', 'cholera', 'typhoid', 'diarrhea outbreak',
            'water contamination', 'bacterial infection', 'gastroenteritis',
            'health emergency', 'disease outbreak', 'water quality'
        ]
        
        # Northeast India states
        self.northeast_states = [
            'Assam', 'Arunachal Pradesh', 'Manipur', 'Meghalaya',
            'Mizoram', 'Nagaland', 'Tripura', 'Sikkim'
        ]
    
    def demo_news_scraping(self):
        """Demo: Scrape news articles for health information"""
        print("=" * 60)
        print("DEMO 1: NEWS ARTICLE SCRAPING")
        print("=" * 60)
        
        # Sample news sources (using public APIs or demo sites)
        news_sources = [
            {
                'name': 'Sample Health News',
                'url': 'https://httpbin.org/html',  # Demo endpoint
                'type': 'demo'
            }
        ]
        
        scraped_articles = []
        
        for source in news_sources:
            print(f"\nScraping from: {source['name']}")
            try:
                if source['type'] == 'demo':
                    # Create demo article data
                    demo_articles = self._create_demo_news_articles()
                    scraped_articles.extend(demo_articles)
                    print(f"✓ Created {len(demo_articles)} demo articles")
                else:
                    # Real scraping would go here
                    articles = self._scrape_news_source(source['url'])
                    scraped_articles.extend(articles)
                    print(f"✓ Scraped {len(articles)} articles")
                    
            except Exception as e:
                print(f"✗ Error scraping {source['name']}: {e}")
        
        print(f"\nTotal articles scraped: {len(scraped_articles)}")
        return scraped_articles
    
    def _create_demo_news_articles(self) -> List[Dict[str, Any]]:
        """Create demo news articles for demonstration"""
        demo_articles = [
            {
                'title': 'Water Contamination Alert in Assam: 50 Cases of Diarrhea Reported',
                'content': 'Health officials in Assam have reported 50 cases of diarrhea in Guwahati district. Water quality tests show high bacterial contamination in local water sources. Residents are advised to boil water before consumption.',
                'location': 'Assam',
                'publish_date': datetime.now() - timedelta(days=2),
                'url': 'https://demo-news.com/assam-water-contamination',
                'source': 'demo_news'
            },
            {
                'title': 'Cholera Outbreak in Manipur: 12 Cases Confirmed',
                'content': 'A cholera outbreak has been confirmed in Imphal, Manipur. Health department has set up emergency response teams. Water quality monitoring shows contamination in several areas.',
                'location': 'Manipur',
                'publish_date': datetime.now() - timedelta(days=1),
                'url': 'https://demo-news.com/manipur-cholera-outbreak',
                'source': 'demo_news'
            },
            {
                'title': 'Water Quality Crisis in Meghalaya: Multiple Districts Affected',
                'content': 'Water quality tests in Meghalaya show contamination in 5 districts. Health officials report increased cases of gastroenteritis. Emergency water treatment measures are being implemented.',
                'location': 'Meghalaya',
                'publish_date': datetime.now() - timedelta(hours=12),
                'url': 'https://demo-news.com/meghalaya-water-crisis',
                'source': 'demo_news'
            },
            {
                'title': 'Typhoid Cases Rise in Nagaland: Water Source Investigation',
                'content': 'Nagaland health department reports 25 typhoid cases in Kohima. Investigation reveals contaminated water supply. Public health advisory issued for water treatment.',
                'location': 'Nagaland',
                'publish_date': datetime.now() - timedelta(hours=6),
                'url': 'https://demo-news.com/nagaland-typhoid-cases',
                'source': 'demo_news'
            }
        ]
        
        return demo_articles
    
    def demo_water_quality_scraping(self):
        """Demo: Scrape water quality data"""
        print("\n" + "=" * 60)
        print("DEMO 2: WATER QUALITY DATA SCRAPING")
        print("=" * 60)
        
        # Create demo water quality data
        water_quality_data = self._create_demo_water_quality_data()
        
        print(f"✓ Generated {len(water_quality_data)} water quality records")
        return water_quality_data
    
    def _create_demo_water_quality_data(self) -> List[Dict[str, Any]]:
        """Create demo water quality data"""
        import random
        
        water_data = []
        locations = [
            {'name': 'Guwahati, Assam', 'lat': 26.1445, 'lon': 91.7362},
            {'name': 'Imphal, Manipur', 'lat': 24.8170, 'lon': 93.9368},
            {'name': 'Shillong, Meghalaya', 'lat': 25.5788, 'lon': 91.8933},
            {'name': 'Kohima, Nagaland', 'lat': 25.6751, 'lon': 94.1106},
            {'name': 'Aizawl, Mizoram', 'lat': 23.7271, 'lon': 92.7176},
            {'name': 'Agartala, Tripura', 'lat': 23.8315, 'lon': 91.2862},
            {'name': 'Gangtok, Sikkim', 'lat': 27.3389, 'lon': 88.6065},
            {'name': 'Itanagar, Arunachal Pradesh', 'lat': 28.2180, 'lon': 94.7278}
        ]
        
        for i, location in enumerate(locations):
            # Simulate contaminated water in some locations
            is_contaminated = random.random() < 0.3  # 30% chance of contamination
            
            if is_contaminated:
                turbidity = random.uniform(5.0, 15.0)
                bacterial_count = random.uniform(100, 1000)
                ph_level = random.uniform(6.0, 8.5)
                contamination_level = "high" if bacterial_count > 500 else "medium"
            else:
                turbidity = random.uniform(0.1, 4.0)
                bacterial_count = random.uniform(0, 50)
                ph_level = random.uniform(6.5, 8.0)
                contamination_level = "low"
            
            water_record = {
                'sensor_id': f'NE_SENSOR_{i+1:03d}',
                'location_name': location['name'],
                'latitude': location['lat'],
                'longitude': location['lon'],
                'turbidity': round(turbidity, 2),
                'ph_level': round(ph_level, 2),
                'temperature': round(random.uniform(20, 35), 1),
                'dissolved_oxygen': round(random.uniform(5, 12), 2),
                'bacterial_count': round(bacterial_count, 0),
                'chlorine_residual': round(random.uniform(0.0, 2.0), 2),
                'conductivity': round(random.uniform(100, 1000), 0),
                'total_dissolved_solids': round(random.uniform(50, 500), 0),
                'nitrate_level': round(random.uniform(0, 10), 2),
                'phosphate_level': round(random.uniform(0, 2), 2),
                'is_contaminated': is_contaminated,
                'contamination_level': contamination_level,
                'measurement_timestamp': datetime.now() - timedelta(hours=random.randint(0, 168))
            }
            
            water_data.append(water_record)
        
        return water_data
    
    def demo_data_processing(self, articles: List[Dict], water_data: List[Dict]):
        """Demo: Process scraped data into structured format"""
        print("\n" + "=" * 60)
        print("DEMO 3: DATA PROCESSING & STRUCTURING")
        print("=" * 60)
        
        # Process articles into health reports
        health_reports = []
        for article in articles:
            health_report = self._extract_health_insights(article)
            if health_report:
                health_reports.append(health_report)
        
        print(f"✓ Processed {len(health_reports)} health reports from articles")
        
        # Process water quality data
        water_quality_df = pd.DataFrame(water_data)
        print(f"✓ Processed {len(water_quality_df)} water quality records")
        
        return health_reports, water_quality_df
    
    def _extract_health_insights(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """Extract health insights from article"""
        content = f"{article['title']} {article['content']}".lower()
        
        # Extract symptoms
        symptoms = {
            'fever': 'fever' in content or 'temperature' in content,
            'diarrhea': 'diarrhea' in content or 'loose motion' in content,
            'vomiting': 'vomiting' in content or 'vomit' in content,
            'nausea': 'nausea' in content,
            'abdominal_pain': 'stomach pain' in content or 'abdominal' in content,
            'dehydration': 'dehydration' in content
        }
        
        # Calculate severity
        severity = 1
        if 'outbreak' in content or 'epidemic' in content:
            severity = 5
        elif 'severe' in content or 'crisis' in content:
            severity = 4
        elif 'moderate' in content:
            severity = 3
        elif 'mild' in content:
            severity = 2
        
        # Only create report if symptoms are mentioned
        if any(symptoms.values()):
            return {
                'report_id': f"DEMO_{hash(article['url'])}",
                'user_id': 'demo_user',
                'location_lat': self._get_coordinates_for_location(article['location'])[0],
                'location_lon': self._get_coordinates_for_location(article['location'])[1],
                'location_address': article['location'],
                'fever': symptoms['fever'],
                'diarrhea': symptoms['diarrhea'],
                'vomiting': symptoms['vomiting'],
                'nausea': symptoms['nausea'],
                'abdominal_pain': symptoms['abdominal_pain'],
                'dehydration': symptoms['dehydration'],
                'symptom_severity': severity,
                'other_symptoms': self._extract_other_symptoms(content),
                'report_source': 'news_scraping',
                'report_timestamp': article['publish_date'],
                'article_url': article['url'],
                'confidence': 0.8
            }
        
        return None
    
    def _get_coordinates_for_location(self, location: str) -> tuple:
        """Get coordinates for location"""
        coordinates = {
            'Assam': (26.1445, 91.7362),
            'Manipur': (24.8170, 93.9368),
            'Meghalaya': (25.5788, 91.8933),
            'Nagaland': (25.6751, 94.1106),
            'Mizoram': (23.7271, 92.7176),
            'Tripura': (23.8315, 91.2862),
            'Sikkim': (27.3389, 88.6065),
            'Arunachal Pradesh': (28.2180, 94.7278)
        }
        
        for state, coords in coordinates.items():
            if state.lower() in location.lower():
                return coords
        
        return (26.0, 92.0)  # Default to center of Northeast
    
    def _extract_other_symptoms(self, content: str) -> str:
        """Extract other symptoms from content"""
        other_symptoms = []
        symptom_patterns = ['headache', 'body ache', 'weakness', 'fatigue', 'cramps']
        
        for pattern in symptom_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                other_symptoms.append(pattern)
        
        return ', '.join(other_symptoms) if other_symptoms else None
    
    def demo_dataset_creation(self, health_reports: List[Dict], water_quality_df: pd.DataFrame):
        """Demo: Create training and testing datasets"""
        print("\n" + "=" * 60)
        print("DEMO 4: DATASET CREATION FOR ML TRAINING")
        print("=" * 60)
        
        # Convert health reports to DataFrame
        health_df = pd.DataFrame(health_reports)
        
        # Add synthetic data to ensure we have enough samples
        synthetic_data = self._generate_synthetic_data()
        
        # Combine real and synthetic data
        combined_health = pd.concat([health_df, synthetic_data['health_reports']], ignore_index=True)
        combined_water = pd.concat([water_quality_df, synthetic_data['water_quality']], ignore_index=True)
        
        print(f"✓ Combined dataset:")
        print(f"  - Health reports: {len(combined_health)} (Real: {len(health_df)}, Synthetic: {len(synthetic_data['health_reports'])})")
        print(f"  - Water quality: {len(combined_water)} (Real: {len(water_quality_df)}, Synthetic: {len(synthetic_data['water_quality'])})")
        
        # Split into training and testing
        train_health = combined_health.sample(frac=0.8, random_state=42)
        test_health = combined_health.drop(train_health.index)
        
        train_water = combined_water.sample(frac=0.8, random_state=42)
        test_water = combined_water.drop(train_water.index)
        
        print(f"\n✓ Dataset split:")
        print(f"  - Training: {len(train_health)} health reports, {len(train_water)} water records")
        print(f"  - Testing: {len(test_health)} health reports, {len(test_water)} water records")
        
        return {
            'train_health': train_health,
            'test_health': test_health,
            'train_water': train_water,
            'test_water': test_water,
            'combined_health': combined_health,
            'combined_water': combined_water
        }
    
    def _generate_synthetic_data(self) -> Dict[str, pd.DataFrame]:
        """Generate synthetic data to supplement real data"""
        import numpy as np
        
        # Generate synthetic health reports
        synthetic_health = []
        for i in range(50):  # Generate 50 synthetic reports
            report = {
                'report_id': f"SYNTH_{i+1:03d}",
                'user_id': f"USER_{i+1:03d}",
                'location_lat': 26.0 + np.random.uniform(-2, 2),
                'location_lon': 92.0 + np.random.uniform(-2, 2),
                'location_address': f"Synthetic Location {i+1}",
                'fever': np.random.choice([True, False], p=[0.3, 0.7]),
                'diarrhea': np.random.choice([True, False], p=[0.4, 0.6]),
                'vomiting': np.random.choice([True, False], p=[0.2, 0.8]),
                'nausea': np.random.choice([True, False], p=[0.3, 0.7]),
                'abdominal_pain': np.random.choice([True, False], p=[0.25, 0.75]),
                'dehydration': np.random.choice([True, False], p=[0.15, 0.85]),
                'symptom_severity': np.random.randint(1, 6),
                'other_symptoms': None,
                'report_source': 'synthetic',
                'report_timestamp': datetime.now() - timedelta(days=np.random.randint(0, 30)),
                'article_url': None,
                'confidence': 0.5
            }
            synthetic_health.append(report)
        
        # Generate synthetic water quality data
        synthetic_water = []
        for i in range(50):
            is_contaminated = np.random.choice([True, False], p=[0.2, 0.8])
            
            water_record = {
                'sensor_id': f'SYNTH_SENSOR_{i+1:03d}',
                'location_name': f'Synthetic Location {i+1}',
                'latitude': 26.0 + np.random.uniform(-2, 2),
                'longitude': 92.0 + np.random.uniform(-2, 2),
                'turbidity': round(np.random.uniform(0.1, 15.0), 2),
                'ph_level': round(np.random.uniform(6.0, 8.5), 2),
                'temperature': round(np.random.uniform(20, 35), 1),
                'dissolved_oxygen': round(np.random.uniform(5, 12), 2),
                'bacterial_count': round(np.random.uniform(0, 1000), 0),
                'chlorine_residual': round(np.random.uniform(0.0, 2.0), 2),
                'conductivity': round(np.random.uniform(100, 1000), 0),
                'total_dissolved_solids': round(np.random.uniform(50, 500), 0),
                'nitrate_level': round(np.random.uniform(0, 10), 2),
                'phosphate_level': round(np.random.uniform(0, 2), 2),
                'is_contaminated': is_contaminated,
                'contamination_level': 'high' if is_contaminated else 'low',
                'measurement_timestamp': datetime.now() - timedelta(hours=np.random.randint(0, 168))
            }
            synthetic_water.append(water_record)
        
        return {
            'health_reports': pd.DataFrame(synthetic_health),
            'water_quality': pd.DataFrame(synthetic_water)
        }
    
    def demo_data_analysis(self, datasets: Dict[str, pd.DataFrame]):
        """Demo: Analyze the datasets"""
        print("\n" + "=" * 60)
        print("DEMO 5: DATASET ANALYSIS")
        print("=" * 60)
        
        health_df = datasets['combined_health']
        water_df = datasets['combined_water']
        
        print("\n📊 HEALTH REPORTS ANALYSIS:")
        print(f"Total reports: {len(health_df)}")
        print(f"Date range: {health_df['report_timestamp'].min()} to {health_df['report_timestamp'].max()}")
        
        print("\nSymptom distribution:")
        symptoms = ['fever', 'diarrhea', 'vomiting', 'nausea', 'abdominal_pain', 'dehydration']
        for symptom in symptoms:
            count = health_df[symptom].sum()
            percentage = (count / len(health_df)) * 100
            print(f"  {symptom}: {count} ({percentage:.1f}%)")
        
        print(f"\nSeverity distribution:")
        severity_counts = health_df['symptom_severity'].value_counts().sort_index()
        for severity, count in severity_counts.items():
            percentage = (count / len(health_df)) * 100
            print(f"  Level {severity}: {count} ({percentage:.1f}%)")
        
        print(f"\nSource distribution:")
        source_counts = health_df['report_source'].value_counts()
        for source, count in source_counts.items():
            percentage = (count / len(health_df)) * 100
            print(f"  {source}: {count} ({percentage:.1f}%)")
        
        print("\n🌊 WATER QUALITY ANALYSIS:")
        print(f"Total water quality records: {len(water_df)}")
        
        print(f"\nContamination status:")
        contamination_counts = water_df['is_contaminated'].value_counts()
        for status, count in contamination_counts.items():
            percentage = (count / len(water_df)) * 100
            print(f"  {'Contaminated' if status else 'Clean'}: {count} ({percentage:.1f}%)")
        
        print(f"\nContamination levels:")
        level_counts = water_df['contamination_level'].value_counts()
        for level, count in level_counts.items():
            percentage = (count / len(water_df)) * 100
            print(f"  {level}: {count} ({percentage:.1f}%)")
        
        print(f"\nWater quality parameters (mean ± std):")
        numeric_cols = ['turbidity', 'ph_level', 'temperature', 'bacterial_count', 'chlorine_residual']
        for col in numeric_cols:
            if col in water_df.columns:
                mean_val = water_df[col].mean()
                std_val = water_df[col].std()
                print(f"  {col}: {mean_val:.2f} ± {std_val:.2f}")
    
    def demo_prediction_simulation(self, datasets: Dict[str, pd.DataFrame]):
        """Demo: Simulate ML predictions"""
        print("\n" + "=" * 60)
        print("DEMO 6: PREDICTION SIMULATION")
        print("=" * 60)
        
        # Simulate outbreak predictions
        health_df = datasets['combined_health']
        water_df = datasets['combined_water']
        
        print("🔮 SIMULATING OUTBREAK PREDICTIONS...")
        
        # Simple heuristic-based prediction
        predictions = []
        
        for i in range(min(10, len(health_df))):  # Show first 10 predictions
            health_record = health_df.iloc[i]
            
            # Find nearest water quality data
            nearest_water = self._find_nearest_water_quality(health_record, water_df)
            
            # Calculate risk score
            risk_score = self._calculate_risk_score(health_record, nearest_water)
            
            # Determine outbreak probability
            outbreak_probability = min(1.0, risk_score / 10.0)
            
            # Predict disease type
            predicted_disease = self._predict_disease_type(health_record, nearest_water)
            
            prediction = {
                'location': f"{health_record['location_address']} ({health_record['location_lat']:.3f}, {health_record['location_lon']:.3f})",
                'risk_score': round(risk_score, 2),
                'outbreak_probability': round(outbreak_probability, 3),
                'predicted_disease': predicted_disease,
                'severity_level': 'High' if outbreak_probability > 0.7 else 'Medium' if outbreak_probability > 0.4 else 'Low',
                'confidence': round(min(0.9, risk_score / 5.0), 2)
            }
            
            predictions.append(prediction)
        
        # Display predictions
        print(f"\nGenerated {len(predictions)} outbreak predictions:")
        print("-" * 80)
        print(f"{'Location':<30} {'Risk':<6} {'Prob':<6} {'Disease':<15} {'Level':<8} {'Conf':<6}")
        print("-" * 80)
        
        for pred in predictions:
            print(f"{pred['location'][:29]:<30} {pred['risk_score']:<6} {pred['outbreak_probability']:<6} {pred['predicted_disease'][:14]:<15} {pred['severity_level']:<8} {pred['confidence']:<6}")
        
        # Summary statistics
        high_risk = len([p for p in predictions if p['outbreak_probability'] > 0.7])
        medium_risk = len([p for p in predictions if 0.4 < p['outbreak_probability'] <= 0.7])
        low_risk = len([p for p in predictions if p['outbreak_probability'] <= 0.4])
        
        print(f"\nRisk Level Summary:")
        print(f"  High Risk (Prob > 0.7): {high_risk} predictions")
        print(f"  Medium Risk (0.4 < Prob ≤ 0.7): {medium_risk} predictions")
        print(f"  Low Risk (Prob ≤ 0.4): {low_risk} predictions")
        
        return predictions
    
    def _find_nearest_water_quality(self, health_record, water_df):
        """Find nearest water quality record"""
        if water_df.empty:
            return None
        
        # Simple distance calculation
        distances = []
        for _, water_record in water_df.iterrows():
            dist = ((health_record['location_lat'] - water_record['latitude'])**2 + 
                   (health_record['location_lon'] - water_record['longitude'])**2)**0.5
            distances.append(dist)
        
        nearest_idx = distances.index(min(distances))
        return water_df.iloc[nearest_idx]
    
    def _calculate_risk_score(self, health_record, water_record):
        """Calculate risk score based on health and water data"""
        risk = 0
        
        # Health symptoms contribute to risk
        symptoms = ['fever', 'diarrhea', 'vomiting', 'nausea', 'abdominal_pain', 'dehydration']
        symptom_count = sum(health_record[symptom] for symptom in symptoms)
        risk += symptom_count * 2
        
        # Severity contributes to risk
        risk += health_record['symptom_severity'] * 1.5
        
        # Water contamination contributes to risk
        if water_record is not None:
            if water_record['is_contaminated']:
                risk += 3
            if water_record['bacterial_count'] > 100:
                risk += 2
            if water_record['turbidity'] > 4:
                risk += 1
        
        return min(10, risk)  # Cap at 10
    
    def _predict_disease_type(self, health_record, water_record):
        """Predict disease type based on symptoms and water quality"""
        symptoms = []
        if health_record['diarrhea']:
            symptoms.append('diarrhea')
        if health_record['fever']:
            symptoms.append('fever')
        if health_record['vomiting']:
            symptoms.append('vomiting')
        
        if water_record is not None and water_record['is_contaminated']:
            if 'diarrhea' in symptoms and 'fever' in symptoms:
                return 'Cholera'
            elif 'fever' in symptoms:
                return 'Typhoid'
            else:
                return 'Gastroenteritis'
        else:
            return 'Gastroenteritis'
    
    def save_datasets(self, datasets: Dict[str, pd.DataFrame]):
        """Save datasets to files for inspection"""
        print("\n" + "=" * 60)
        print("DEMO 7: SAVING DATASETS")
        print("=" * 60)
        
        # Create data directory
        import os
        os.makedirs('demo_data', exist_ok=True)
        
        # Save datasets
        datasets['combined_health'].to_csv('demo_data/health_reports_dataset.csv', index=False)
        datasets['combined_water'].to_csv('demo_data/water_quality_dataset.csv', index=False)
        datasets['train_health'].to_csv('demo_data/train_health_dataset.csv', index=False)
        datasets['test_health'].to_csv('demo_data/test_health_dataset.csv', index=False)
        datasets['train_water'].to_csv('demo_data/train_water_dataset.csv', index=False)
        datasets['test_water'].to_csv('demo_data/test_water_dataset.csv', index=False)
        
        print("✓ Datasets saved to 'demo_data/' directory:")
        print("  - health_reports_dataset.csv (Complete health reports)")
        print("  - water_quality_dataset.csv (Complete water quality data)")
        print("  - train_health_dataset.csv (Training health data)")
        print("  - test_health_dataset.csv (Testing health data)")
        print("  - train_water_dataset.csv (Training water data)")
        print("  - test_water_dataset.csv (Testing water data)")
        
        # Show sample data
        print(f"\n📋 SAMPLE HEALTH REPORTS DATA:")
        print(datasets['combined_health'].head().to_string())
        
        print(f"\n📋 SAMPLE WATER QUALITY DATA:")
        print(datasets['combined_water'].head().to_string())

def main():
    """Main demo function"""
    print("🚀 SMART HEALTH SURVEILLANCE - DATA SCRAPING DEMO")
    print("=" * 80)
    print("This demo shows how the system scrapes data from various sources")
    print("and processes it for machine learning model training.")
    print("=" * 80)
    
    # Initialize demo
    demo = DataScrapingDemo()
    
    try:
        # Step 1: Scrape news articles
        articles = demo.demo_news_scraping()
        
        # Step 2: Scrape water quality data
        water_data = demo.demo_water_quality_scraping()
        
        # Step 3: Process data
        health_reports, water_quality_df = demo.demo_data_processing(articles, water_data)
        
        # Step 4: Create datasets
        datasets = demo.demo_dataset_creation(health_reports, water_quality_df)
        
        # Step 5: Analyze datasets
        demo.demo_data_analysis(datasets)
        
        # Step 6: Simulate predictions
        predictions = demo.demo_prediction_simulation(datasets)
        
        # Step 7: Save datasets
        demo.save_datasets(datasets)
        
        print("\n" + "=" * 80)
        print("🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("The system has demonstrated:")
        print("✓ News article scraping and processing")
        print("✓ Water quality data collection")
        print("✓ Data structuring and validation")
        print("✓ Dataset creation for ML training")
        print("✓ Data analysis and insights")
        print("✓ Outbreak prediction simulation")
        print("✓ Dataset export for inspection")
        
        print(f"\nNext steps:")
        print(f"1. Check the 'demo_data/' directory for exported datasets")
        print(f"2. Run the full training: python run_fetch_and_train.py")
        print(f"3. Start the API: docker-compose up app")
        print(f"4. Test predictions: python scripts/test_api.py")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
