"""
Specialized script to fetch water quality and health data specifically from Northeast India
"""
import asyncio
import pandas as pd
import requests
import json
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import re
import logging
from typing import List, Dict, Any
import sys
from pathlib import Path

# Add the app directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from app.core.database import init_db, SessionLocal
from app.models.health_data import HealthReport, WaterQualityData
from app.ml.model_manager import ModelManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NortheastDataFetcher:
    """Specialized fetcher for Northeast India data"""
    
    def __init__(self):
        self.northeast_states = [
            'Assam', 'Arunachal Pradesh', 'Manipur', 'Meghalaya', 
            'Mizoram', 'Nagaland', 'Tripura', 'Sikkim'
        ]
        
        self.northeast_coordinates = {
            'Assam': {'lat': 26.2006, 'lon': 92.9376},
            'Arunachal Pradesh': {'lat': 28.2180, 'lon': 94.7278},
            'Manipur': {'lat': 24.6637, 'lon': 93.9063},
            'Meghalaya': {'lat': 25.4670, 'lon': 91.3662},
            'Mizoram': {'lat': 23.1645, 'lon': 92.9376},
            'Nagaland': {'lat': 26.1584, 'lon': 94.5624},
            'Tripura': {'lat': 23.9408, 'lon': 91.9882},
            'Sikkim': {'lat': 27.5330, 'lon': 88.5122}
        }
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    async def fetch_northeast_news(self) -> List[Dict[str, Any]]:
        """Fetch news articles specifically from Northeast India"""
        logger.info("Fetching Northeast India news articles...")
        
        articles = []
        
        # Northeast-specific news sources
        sources = [
            'https://www.assamtribune.com',
            'https://www.sentinelassam.com',
            'https://www.nagalandpost.com',
            'https://www.manipuronline.in',
            'https://www.mizoram.gov.in',
            'https://www.meghalaya.gov.in'
        ]
        
        search_terms = [
            'water quality', 'water contamination', 'cholera', 'typhoid',
            'diarrhea outbreak', 'water borne disease', 'health emergency',
            'water testing', 'bacterial infection', 'gastroenteritis'
        ]
        
        for source in sources:
            for term in search_terms:
                try:
                    articles.extend(await self._search_northeast_news(source, term))
                    await asyncio.sleep(1)  # Rate limiting
                except Exception as e:
                    logger.error(f"Error fetching from {source}: {e}")
                    continue
        
        logger.info(f"Fetched {len(articles)} Northeast news articles")
        return articles
    
    async def _search_northeast_news(self, base_url: str, search_term: str) -> List[Dict[str, Any]]:
        """Search for news articles on Northeast websites"""
        articles = []
        
        try:
            # Construct search URL
            search_url = f"{base_url}/search?q={search_term}"
            
            response = self.session.get(search_url, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract article links
                article_links = self._extract_article_links(soup, base_url)
                
                for link in article_links[:5]:  # Limit per search
                    article_data = await self._scrape_northeast_article(link)
                    if article_data:
                        articles.append(article_data)
                        
        except Exception as e:
            logger.error(f"Error searching {base_url}: {e}")
        
        return articles
    
    def _extract_article_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract article links from search results"""
        links = []
        
        # Common selectors for news articles
        selectors = [
            'a[href*="/article/"]',
            'a[href*="/news/"]',
            'a[href*="/story/"]',
            '.story a',
            '.article a',
            '.news-item a',
            'h3 a',
            'h2 a'
        ]
        
        for selector in selectors:
            elements = soup.select(selector)
            for element in elements:
                href = element.get('href')
                if href and self._is_northeast_relevant(element.get_text()):
                    full_url = f"{base_url.rstrip('/')}/{href.lstrip('/')}"
                    links.append(full_url)
        
        return list(set(links))
    
    def _is_northeast_relevant(self, text: str) -> bool:
        """Check if text is relevant to Northeast India"""
        text_lower = text.lower()
        return any(state.lower() in text_lower for state in self.northeast_states) or \
               any(keyword in text_lower for keyword in ['northeast', 'north east', 'ne india'])
    
    async def _scrape_northeast_article(self, url: str) -> Dict[str, Any]:
        """Scrape individual Northeast article"""
        try:
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                title = self._extract_title(soup)
                content = self._extract_content(soup)
                publish_date = self._extract_publish_date(soup)
                location = self._extract_northeast_location(soup, content)
                
                if title and content:
                    return {
                        'url': url,
                        'title': title,
                        'content': content,
                        'publish_date': publish_date,
                        'location': location,
                        'scraped_at': datetime.now(),
                        'source': 'northeast_news'
                    }
        except Exception as e:
            logger.error(f"Error scraping article {url}: {e}")
        
        return None
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract article title"""
        selectors = ['h1', '.article-title', '.story-title', '.headline', 'title']
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                return element.get_text().strip()
        
        return ""
    
    def _extract_content(self, soup: BeautifulSoup) -> str:
        """Extract article content"""
        selectors = [
            '.article-content', '.story-content', '.article-body',
            '.content', 'article', '.post-content', '.news-content'
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                # Remove script and style elements
                for script in element(["script", "style"]):
                    script.decompose()
                return element.get_text().strip()
        
        return ""
    
    def _extract_publish_date(self, soup: BeautifulSoup) -> datetime:
        """Extract publish date"""
        selectors = [
            'time[datetime]', '.publish-date', '.article-date',
            '.story-date', '.date', '.published'
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                date_str = element.get('datetime') or element.get_text()
                try:
                    return pd.to_datetime(date_str).to_pydatetime()
                except:
                    continue
        
        return datetime.now()
    
    def _extract_northeast_location(self, soup: BeautifulSoup, content: str) -> str:
        """Extract location from Northeast article"""
        content_lower = content.lower()
        
        for state in self.northeast_states:
            if state.lower() in content_lower:
                return state
        
        # Look for city names
        cities = [
            'guwahati', 'shillong', 'aizawl', 'kohima', 'imphal',
            'agartala', 'gangtok', 'itanagar', 'dimapur', 'jorhat'
        ]
        
        for city in cities:
            if city in content_lower:
                return city.title()
        
        return "Northeast India"
    
    async def fetch_government_water_reports(self) -> List[Dict[str, Any]]:
        """Fetch water quality reports from government sources"""
        logger.info("Fetching government water quality reports...")
        
        water_data = []
        
        # Government water quality data sources
        sources = [
            'https://www.cgwb.gov.in/water-quality-data',
            'https://www.cpcb.nic.in/water-quality-data',
            'https://www.mowr.gov.in/water-quality-monitoring'
        ]
        
        for source in sources:
            try:
                data = await self._scrape_government_water_data(source)
                water_data.extend(data)
            except Exception as e:
                logger.error(f"Error fetching from {source}: {e}")
                continue
        
        logger.info(f"Fetched {len(water_data)} water quality records")
        return water_data
    
    async def _scrape_government_water_data(self, url: str) -> List[Dict[str, Any]]:
        """Scrape water quality data from government website"""
        water_data = []
        
        try:
            response = self.session.get(url, timeout=15)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Look for data tables
                tables = soup.find_all('table')
                
                for table in tables:
                    data = self._parse_water_quality_table(table)
                    water_data.extend(data)
                
                # Look for downloadable data files
                data_links = soup.find_all('a', href=re.compile(r'\.(pdf|xlsx|csv)$'))
                for link in data_links:
                    file_url = link.get('href')
                    if file_url:
                        file_data = await self._process_data_file(file_url)
                        water_data.extend(file_data)
                        
        except Exception as e:
            logger.error(f"Error scraping government data from {url}: {e}")
        
        return water_data
    
    def _parse_water_quality_table(self, table) -> List[Dict[str, Any]]:
        """Parse water quality data from HTML table"""
        water_data = []
        
        try:
            rows = table.find_all('tr')
            if len(rows) < 2:
                return water_data
            
            # Get headers
            headers = [th.get_text().strip().lower() for th in rows[0].find_all(['th', 'td'])]
            
            # Parse data rows
            for row in rows[1:]:
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 3:
                    row_data = {}
                    for i, cell in enumerate(cells):
                        if i < len(headers):
                            value = cell.get_text().strip()
                            row_data[headers[i]] = value
                    
                    # Convert to water quality format
                    water_record = self._convert_to_water_quality_format(row_data)
                    if water_record:
                        water_data.append(water_record)
        
        except Exception as e:
            logger.error(f"Error parsing water quality table: {e}")
        
        return water_data
    
    def _convert_to_water_quality_format(self, row_data: Dict[str, str]) -> Dict[str, Any]:
        """Convert table row to water quality data format"""
        try:
            # Map column names
            column_mapping = {
                'location': ['location', 'place', 'site', 'station', 'district'],
                'latitude': ['lat', 'latitude', 'lat.'],
                'longitude': ['lon', 'longitude', 'lng', 'long.'],
                'turbidity': ['turbidity', 'turb', 'ntu'],
                'ph': ['ph', 'ph level', 'ph value'],
                'temperature': ['temp', 'temperature', 'temp.'],
                'bacterial_count': ['bacteria', 'bacterial count', 'e coli', 'coliform', 'tcb'],
                'chlorine': ['chlorine', 'chlorine residual', 'cl2'],
                'nitrate': ['nitrate', 'no3', 'nitrate level'],
                'phosphate': ['phosphate', 'po4', 'phosphate level'],
                'dissolved_oxygen': ['do', 'dissolved oxygen', 'oxygen']
            }
            
            water_record = {}
            
            for our_key, possible_keys in column_mapping.items():
                for key, value in row_data.items():
                    if any(possible_key in key.lower() for possible_key in possible_keys):
                        try:
                            if our_key in ['latitude', 'longitude']:
                                water_record[our_key] = float(value)
                            elif our_key in ['turbidity', 'ph', 'temperature', 'bacterial_count', 
                                          'chlorine', 'nitrate', 'phosphate', 'dissolved_oxygen']:
                                water_record[our_key] = float(value) if value.replace('.', '').replace('-', '').isdigit() else None
                            else:
                                water_record[our_key] = value
                        except ValueError:
                            continue
            
            # Add default coordinates if not present
            if 'location' in water_record and 'latitude' not in water_record:
                location = water_record['location'].lower()
                for state, coords in self.northeast_coordinates.items():
                    if state.lower() in location:
                        water_record['latitude'] = coords['lat'] + (hash(location) % 100 - 50) / 1000
                        water_record['longitude'] = coords['lon'] + (hash(location) % 100 - 50) / 1000
                        break
            
            # Only return if we have essential data
            if 'location' in water_record and ('latitude' in water_record or 'longitude' in water_record):
                return water_record
        
        except Exception as e:
            logger.error(f"Error converting to water quality format: {e}")
        
        return None
    
    async def _process_data_file(self, file_url: str) -> List[Dict[str, Any]]:
        """Process downloadable data files"""
        # This would handle PDF, Excel, and CSV files
        # For now, return empty list
        return []
    
    def process_northeast_data(self, articles: List[Dict], water_data: List[Dict]) -> Dict[str, pd.DataFrame]:
        """Process Northeast-specific data"""
        processed_data = {
            'health_reports': pd.DataFrame(),
            'water_quality': pd.DataFrame(),
            'news_insights': pd.DataFrame()
        }
        
        # Process articles into health reports
        health_reports = []
        for article in articles:
            health_report = self._extract_health_insights_from_northeast_article(article)
            if health_report:
                health_reports.append(health_report)
        
        if health_reports:
            processed_data['health_reports'] = pd.DataFrame(health_reports)
        
        # Process water quality data
        if water_data:
            processed_data['water_quality'] = pd.DataFrame(water_data)
        
        return processed_data
    
    def _extract_health_insights_from_northeast_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """Extract health insights from Northeast article"""
        content = f"{article.get('title', '')} {article.get('content', '')}".lower()
        
        # Disease keywords
        disease_keywords = {
            'cholera': ['cholera', 'vibrio cholerae'],
            'typhoid': ['typhoid', 'salmonella typhi'],
            'diarrhea': ['diarrhea', 'diarrhoea', 'loose motion'],
            'dysentery': ['dysentery', 'shigella'],
            'hepatitis': ['hepatitis a', 'jaundice'],
            'gastroenteritis': ['gastroenteritis', 'stomach flu']
        }
        
        # Extract symptoms
        symptoms = {
            'fever': 'fever' in content or 'temperature' in content,
            'diarrhea': 'diarrhea' in content or 'loose motion' in content,
            'vomiting': 'vomiting' in content or 'vomit' in content,
            'nausea': 'nausea' in content or 'nauseous' in content,
            'abdominal_pain': 'stomach pain' in content or 'abdominal' in content,
            'dehydration': 'dehydration' in content or 'dehydrated' in content
        }
        
        # Calculate severity
        severity = 1
        if 'outbreak' in content or 'epidemic' in content:
            severity = 5
        elif 'severe' in content or 'serious' in content:
            severity = 4
        elif 'moderate' in content:
            severity = 3
        elif 'mild' in content:
            severity = 2
        
        # Only create report if symptoms or diseases are mentioned
        if any(symptoms.values()) or any(any(keywords in content for keywords in disease_list) 
                                        for disease_list in disease_keywords.values()):
            
            # Get coordinates for the location
            location = article.get('location', 'Northeast India')
            lat, lon = self._get_coordinates_for_location(location)
            
            return {
                'report_id': f"NE_{hash(article['url'])}",
                'user_id': 'northeast_news',
                'location_lat': lat,
                'location_lon': lon,
                'location_address': location,
                'fever': symptoms['fever'],
                'diarrhea': symptoms['diarrhea'],
                'vomiting': symptoms['vomiting'],
                'nausea': symptoms['nausea'],
                'abdominal_pain': symptoms['abdominal_pain'],
                'dehydration': symptoms['dehydration'],
                'symptom_severity': severity,
                'other_symptoms': self._extract_other_symptoms(content),
                'report_source': 'northeast_news',
                'report_timestamp': article.get('publish_date', datetime.now()),
                'article_url': article['url'],
                'confidence': 0.8  # High confidence for news articles
            }
        
        return None
    
    def _get_coordinates_for_location(self, location: str) -> tuple:
        """Get coordinates for a location in Northeast India"""
        location_lower = location.lower()
        
        # Check for state names
        for state, coords in self.northeast_coordinates.items():
            if state.lower() in location_lower:
                return coords['lat'], coords['lon']
        
        # Check for city names
        city_coords = {
            'guwahati': (26.1445, 91.7362),
            'shillong': (25.5788, 91.8933),
            'aizawl': (23.7271, 92.7176),
            'kohima': (25.6751, 94.1106),
            'imphal': (24.8170, 93.9368),
            'agartala': (23.8315, 91.2862),
            'gangtok': (27.3389, 88.6065),
            'itanagar': (28.2180, 94.7278)
        }
        
        for city, coords in city_coords.items():
            if city in location_lower:
                return coords
        
        # Default to center of Northeast India
        return 26.0, 92.0
    
    def _extract_other_symptoms(self, content: str) -> str:
        """Extract other symptoms from content"""
        other_symptoms = []
        
        symptom_patterns = [
            'headache', 'body ache', 'weakness', 'fatigue',
            'cramps', 'bloating', 'loss of appetite', 'dizziness'
        ]
        
        for pattern in symptom_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                other_symptoms.append(pattern)
        
        return ', '.join(other_symptoms) if other_symptoms else None

async def main():
    """Main function to fetch Northeast data and train models"""
    logger.info("=" * 60)
    logger.info("Northeast India Health Surveillance Data Fetching")
    logger.info("=" * 60)
    
    try:
        # Initialize fetcher
        fetcher = NortheastDataFetcher()
        
        # Fetch Northeast news articles
        articles = await fetcher.fetch_northeast_news()
        
        # Fetch government water quality data
        water_data = await fetcher.fetch_government_water_reports()
        
        # Process the data
        processed_data = fetcher.process_northeast_data(articles, water_data)
        
        # Save to database
        await init_db()
        async with SessionLocal() as db:
            # Save health reports
            if not processed_data['health_reports'].empty:
                logger.info(f"Saving {len(processed_data['health_reports'])} Northeast health reports...")
                
                for _, row in processed_data['health_reports'].iterrows():
                    health_report = HealthReport(
                        report_id=row['report_id'],
                        user_id=row['user_id'],
                        location_lat=row['location_lat'],
                        location_lon=row['location_lon'],
                        location_address=row['location_address'],
                        fever=row['fever'],
                        diarrhea=row['diarrhea'],
                        vomiting=row['vomiting'],
                        nausea=row['nausea'],
                        abdominal_pain=row['abdominal_pain'],
                        dehydration=row['dehydration'],
                        other_symptoms=row.get('other_symptoms'),
                        symptom_severity=row['symptom_severity'],
                        report_source=row['report_source'],
                        report_timestamp=row['report_timestamp']
                    )
                    db.add(health_report)
            
            # Save water quality data
            if not processed_data['water_quality'].empty:
                logger.info(f"Saving {len(processed_data['water_quality'])} Northeast water quality records...")
                
                for _, row in processed_data['water_quality'].iterrows():
                    # Determine contamination level
                    contamination_level = "low"
                    is_contaminated = False
                    
                    if row.get('bacterial_count', 0) > 100:
                        contamination_level = "high"
                        is_contaminated = True
                    elif row.get('turbidity', 0) > 4:
                        contamination_level = "medium"
                        is_contaminated = True
                    
                    water_quality = WaterQualityData(
                        sensor_id=row.get('sensor_id', f"NE_{hash(str(row))}"),
                        location_lat=row.get('latitude', 0.0),
                        location_lon=row.get('longitude', 0.0),
                        turbidity=row.get('turbidity'),
                        ph_level=row.get('ph'),
                        temperature=row.get('temperature'),
                        dissolved_oxygen=row.get('dissolved_oxygen'),
                        bacterial_count=row.get('bacterial_count'),
                        chlorine_residual=row.get('chlorine'),
                        conductivity=row.get('conductivity'),
                        total_dissolved_solids=row.get('total_dissolved_solids'),
                        nitrate_level=row.get('nitrate'),
                        phosphate_level=row.get('phosphate'),
                        is_contaminated=is_contaminated,
                        contamination_level=contamination_level,
                        measurement_timestamp=datetime.now()
                    )
                    db.add(water_quality)
            
            await db.commit()
            logger.info("Northeast data saved to database successfully")
        
        # Train models with Northeast data
        logger.info("Training models with Northeast data...")
        model_manager = ModelManager()
        
        # Add some synthetic data to supplement real data
        from app.utils.sample_data_generator import SampleDataGenerator
        data_generator = SampleDataGenerator()
        synthetic_scenario = data_generator.generate_outbreak_scenario()
        
        # Combine real and synthetic data
        training_data = {
            'health_reports': pd.concat([
                processed_data['health_reports'],
                synthetic_scenario['health_reports']
            ], ignore_index=True) if not processed_data['health_reports'].empty else synthetic_scenario['health_reports'],
            'water_quality': pd.concat([
                processed_data['water_quality'],
                synthetic_scenario['water_quality']
            ], ignore_index=True) if not processed_data['water_quality'].empty else synthetic_scenario['water_quality']
        }
        
        # Train models
        results = await model_manager.train_all_models(training_data)
        
        logger.info("=" * 60)
        logger.info("Northeast data processing and model training completed!")
        logger.info("=" * 60)
        
        logger.info(f"\nData Summary:")
        logger.info(f"- Northeast health reports: {len(processed_data['health_reports'])}")
        logger.info(f"- Northeast water quality records: {len(processed_data['water_quality'])}")
        logger.info(f"- Total training samples: {len(training_data['health_reports'])} health + {len(training_data['water_quality'])} water")
        
        logger.info("\nNext steps:")
        logger.info("1. Start the API: docker-compose up app")
        logger.info("2. Test predictions: python scripts/test_api.py")
        logger.info("3. View API docs: http://localhost:8000/docs")
        
    except Exception as e:
        logger.error(f"Error in main process: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
