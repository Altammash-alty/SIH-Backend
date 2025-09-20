"""
Data fetching service for external sources like news articles, government websites, and health reports
"""
import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import re
import json
import logging
from urllib.parse import urljoin, urlparse
import time
import asyncio
import aiohttp
from app.core.config import Config

logger = logging.getLogger(__name__)

class DataFetcher:
    """Fetches data from various external sources"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Keywords for water-borne diseases
        self.disease_keywords = [
            'cholera', 'typhoid', 'dysentery', 'hepatitis a', 'giardiasis',
            'cryptosporidiosis', 'water borne disease', 'outbreak', 'epidemic',
            'diarrhea', 'gastroenteritis', 'water contamination', 'bacterial infection'
        ]
        
        # Water quality keywords
        self.water_quality_keywords = [
            'water quality', 'turbidity', 'ph level', 'bacterial count',
            'chlorine residual', 'contamination', 'water testing', 'water treatment',
            'e coli', 'coliform', 'nitrate', 'phosphate', 'dissolved oxygen'
        ]
        
        # Northeast India specific keywords
        self.northeast_keywords = [
            'assam', 'arunachal pradesh', 'manipur', 'meghalaya', 'mizoram',
            'nagaland', 'tripura', 'sikkim', 'northeast india', 'north east'
        ]
    
    async def fetch_news_articles(self, sources: List[str] = None, days_back: int = 30) -> List[Dict[str, Any]]:
        """Fetch news articles related to water-borne diseases and water quality"""
        if sources is None:
            sources = [
                'https://www.thehindu.com',
                'https://timesofindia.indiatimes.com',
                'https://www.indiatoday.in',
                'https://www.ndtv.com',
                'https://www.news18.com'
            ]
        
        articles = []
        
        for source in sources:
            try:
                # Search for water quality and disease related articles
                search_terms = self.disease_keywords + self.water_quality_keywords
                
                for term in search_terms[:5]:  # Limit to avoid rate limiting
                    articles.extend(await self._search_news_articles(source, term, days_back))
                    await asyncio.sleep(1)  # Rate limiting
                    
            except Exception as e:
                logger.error(f"Error fetching from {source}: {e}")
                continue
        
        return articles
    
    async def _search_news_articles(self, base_url: str, search_term: str, days_back: int) -> List[Dict[str, Any]]:
        """Search for articles on a specific news website"""
        articles = []
        
        try:
            # Construct search URL (this varies by website)
            search_url = self._construct_search_url(base_url, search_term)
            
            async with aiohttp.ClientSession() as session:
                async with session.get(search_url) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # Extract article links and metadata
                        article_links = self._extract_article_links(soup, base_url)
                        
                        for link in article_links[:10]:  # Limit per search term
                            article_data = await self._scrape_article(session, link)
                            if article_data and self._is_relevant_article(article_data):
                                articles.append(article_data)
                                
        except Exception as e:
            logger.error(f"Error searching {base_url} for {search_term}: {e}")
        
        return articles
    
    def _construct_search_url(self, base_url: str, search_term: str) -> str:
        """Construct search URL based on website"""
        if 'thehindu.com' in base_url:
            return f"{base_url}/search/?q={search_term}&order=DESC&sort=publishdate"
        elif 'timesofindia.indiatimes.com' in base_url:
            return f"{base_url}/topic/{search_term}"
        elif 'indiatoday.in' in base_url:
            return f"{base_url}/search?q={search_term}"
        elif 'ndtv.com' in base_url:
            return f"{base_url}/search?q={search_term}"
        else:
            return f"{base_url}/search?q={search_term}"
    
    def _extract_article_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract article links from search results"""
        links = []
        
        # Common selectors for article links
        selectors = [
            'a[href*="/article/"]',
            'a[href*="/news/"]',
            'a[href*="/story/"]',
            '.story a',
            '.article a',
            '.news-item a'
        ]
        
        for selector in selectors:
            elements = soup.select(selector)
            for element in elements:
                href = element.get('href')
                if href:
                    full_url = urljoin(base_url, href)
                    if self._is_valid_article_url(full_url):
                        links.append(full_url)
        
        return list(set(links))  # Remove duplicates
    
    async def _scrape_article(self, session: aiohttp.ClientSession, url: str) -> Optional[Dict[str, Any]]:
        """Scrape individual article content"""
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Extract article content
                    title = self._extract_title(soup)
                    content = self._extract_content(soup)
                    publish_date = self._extract_publish_date(soup)
                    location = self._extract_location(soup, content)
                    
                    if title and content:
                        return {
                            'url': url,
                            'title': title,
                            'content': content,
                            'publish_date': publish_date,
                            'location': location,
                            'scraped_at': datetime.now(),
                            'source': urlparse(url).netloc
                        }
        except Exception as e:
            logger.error(f"Error scraping article {url}: {e}")
        
        return None
    
    def _extract_title(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract article title"""
        selectors = ['h1', '.article-title', '.story-title', '.headline', 'title']
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                return element.get_text().strip()
        
        return None
    
    def _extract_content(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract article content"""
        selectors = [
            '.article-content',
            '.story-content',
            '.article-body',
            '.content',
            'article',
            '.post-content'
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                # Remove script and style elements
                for script in element(["script", "style"]):
                    script.decompose()
                return element.get_text().strip()
        
        return None
    
    def _extract_publish_date(self, soup: BeautifulSoup) -> Optional[datetime]:
        """Extract article publish date"""
        selectors = [
            'time[datetime]',
            '.publish-date',
            '.article-date',
            '.story-date',
            '.date'
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                date_str = element.get('datetime') or element.get_text()
                try:
                    return pd.to_datetime(date_str).to_pydatetime()
                except:
                    continue
        
        return None
    
    def _extract_location(self, soup: BeautifulSoup, content: str) -> Optional[str]:
        """Extract location information from article"""
        # Look for location in content
        location_patterns = [
            r'in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'from\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'at\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
        ]
        
        for pattern in location_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if any(keyword in match.lower() for keyword in self.northeast_keywords):
                    return match
        
        return None
    
    def _is_valid_article_url(self, url: str) -> bool:
        """Check if URL is a valid article URL"""
        invalid_patterns = [
            r'/tag/',
            r'/category/',
            r'/author/',
            r'/search',
            r'/login',
            r'/register',
            r'#',
            r'javascript:',
            r'mailto:'
        ]
        
        for pattern in invalid_patterns:
            if re.search(pattern, url):
                return False
        
        return True
    
    def _is_relevant_article(self, article: Dict[str, Any]) -> bool:
        """Check if article is relevant to water quality or diseases"""
        if not article.get('title') or not article.get('content'):
            return False
        
        text = f"{article['title']} {article['content']}".lower()
        
        # Check for disease keywords
        disease_matches = sum(1 for keyword in self.disease_keywords if keyword in text)
        
        # Check for water quality keywords
        water_matches = sum(1 for keyword in self.water_quality_keywords if keyword in text)
        
        # Check for northeast keywords
        northeast_matches = sum(1 for keyword in self.northeast_keywords if keyword in text)
        
        # Article is relevant if it has disease/water keywords and is from northeast
        return (disease_matches > 0 or water_matches > 0) and northeast_matches > 0
    
    async def fetch_government_water_data(self) -> List[Dict[str, Any]]:
        """Fetch water quality data from government websites"""
        government_sources = [
            'https://www.cgwb.gov.in',  # Central Ground Water Board
            'https://www.cpcb.nic.in',  # Central Pollution Control Board
            'https://www.mowr.gov.in',  # Ministry of Water Resources
            'https://www.northeastindia.com',  # Northeast India portal
        ]
        
        water_data = []
        
        for source in government_sources:
            try:
                data = await self._scrape_government_water_data(source)
                water_data.extend(data)
            except Exception as e:
                logger.error(f"Error fetching from {source}: {e}")
                continue
        
        return water_data
    
    async def _scrape_government_water_data(self, base_url: str) -> List[Dict[str, Any]]:
        """Scrape water quality data from government website"""
        water_data = []
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(base_url) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # Look for water quality reports and data
                        data_links = self._find_water_data_links(soup, base_url)
                        
                        for link in data_links:
                            data = await self._extract_water_data_from_document(session, link)
                            if data:
                                water_data.extend(data)
                                
        except Exception as e:
            logger.error(f"Error scraping government data from {base_url}: {e}")
        
        return water_data
    
    def _find_water_data_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Find links to water quality data and reports"""
        links = []
        
        # Look for PDFs and data files
        selectors = [
            'a[href$=".pdf"]',
            'a[href$=".xlsx"]',
            'a[href$=".csv"]',
            'a[href*="water"]',
            'a[href*="quality"]',
            'a[href*="report"]'
        ]
        
        for selector in selectors:
            elements = soup.select(selector)
            for element in elements:
                href = element.get('href')
                if href:
                    full_url = urljoin(base_url, href)
                    links.append(full_url)
        
        return list(set(links))
    
    async def _extract_water_data_from_document(self, session: aiohttp.ClientSession, url: str) -> List[Dict[str, Any]]:
        """Extract water quality data from document"""
        try:
            if url.endswith('.pdf'):
                return await self._extract_from_pdf(session, url)
            elif url.endswith(('.xlsx', '.csv')):
                return await self._extract_from_spreadsheet(session, url)
            else:
                return await self._extract_from_webpage(session, url)
        except Exception as e:
            logger.error(f"Error extracting data from {url}: {e}")
            return []
    
    async def _extract_from_webpage(self, session: aiohttp.ClientSession, url: str) -> List[Dict[str, Any]]:
        """Extract water quality data from webpage"""
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Look for tables with water quality data
                    tables = soup.find_all('table')
                    water_data = []
                    
                    for table in tables:
                        data = self._parse_water_quality_table(table)
                        water_data.extend(data)
                    
                    return water_data
        except Exception as e:
            logger.error(f"Error extracting from webpage {url}: {e}")
            return []
    
    def _parse_water_quality_table(self, table) -> List[Dict[str, Any]]:
        """Parse water quality data from HTML table"""
        water_data = []
        
        try:
            rows = table.find_all('tr')
            headers = []
            
            # Get headers from first row
            if rows:
                header_row = rows[0]
                headers = [th.get_text().strip() for th in header_row.find_all(['th', 'td'])]
            
            # Parse data rows
            for row in rows[1:]:
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 3:  # Minimum columns for water quality data
                    row_data = {}
                    for i, cell in enumerate(cells):
                        if i < len(headers):
                            value = cell.get_text().strip()
                            row_data[headers[i].lower()] = value
                    
                    # Convert to water quality format
                    water_record = self._convert_to_water_quality_format(row_data)
                    if water_record:
                        water_data.append(water_record)
        
        except Exception as e:
            logger.error(f"Error parsing water quality table: {e}")
        
        return water_data
    
    def _convert_to_water_quality_format(self, row_data: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """Convert table row to water quality data format"""
        try:
            # Map common column names to our format
            column_mapping = {
                'location': ['location', 'place', 'site', 'station'],
                'latitude': ['lat', 'latitude', 'lat.'],
                'longitude': ['lon', 'longitude', 'lng', 'long.'],
                'turbidity': ['turbidity', 'turb', 'ntu'],
                'ph': ['ph', 'ph level', 'ph value'],
                'temperature': ['temp', 'temperature', 'temp.'],
                'bacterial_count': ['bacteria', 'bacterial count', 'e coli', 'coliform'],
                'chlorine': ['chlorine', 'chlorine residual', 'cl2'],
                'nitrate': ['nitrate', 'no3', 'nitrate level'],
                'phosphate': ['phosphate', 'po4', 'phosphate level']
            }
            
            water_record = {}
            
            for our_key, possible_keys in column_mapping.items():
                for key, value in row_data.items():
                    if any(possible_key in key.lower() for possible_key in possible_keys):
                        try:
                            if our_key in ['latitude', 'longitude']:
                                water_record[our_key] = float(value)
                            elif our_key in ['turbidity', 'ph', 'temperature', 'bacterial_count', 'chlorine', 'nitrate', 'phosphate']:
                                water_record[our_key] = float(value) if value.replace('.', '').isdigit() else None
                            else:
                                water_record[our_key] = value
                        except ValueError:
                            continue
            
            # Only return if we have essential data
            if 'location' in water_record and ('latitude' in water_record or 'longitude' in water_record):
                return water_record
        
        except Exception as e:
            logger.error(f"Error converting to water quality format: {e}")
        
        return None
    
    async def fetch_weather_data(self) -> List[Dict[str, Any]]:
        """Fetch weather data that might affect water quality"""
        weather_data = []
        
        try:
            # Use OpenWeatherMap API if key is available
            if Config.WEATHER_API_KEY:
                weather_data = await self._fetch_from_weather_api()
            else:
                # Fallback to web scraping
                weather_data = await self._scrape_weather_data()
        except Exception as e:
            logger.error(f"Error fetching weather data: {e}")
        
        return weather_data
    
    async def _fetch_from_weather_api(self) -> List[Dict[str, Any]]:
        """Fetch weather data from API"""
        # Implementation for weather API
        pass
    
    async def _scrape_weather_data(self) -> List[Dict[str, Any]]:
        """Scrape weather data from websites"""
        # Implementation for weather scraping
        pass
    
    def process_fetched_data(self, articles: List[Dict], water_data: List[Dict]) -> Dict[str, pd.DataFrame]:
        """Process and structure fetched data for model training"""
        processed_data = {
            'health_reports': pd.DataFrame(),
            'water_quality': pd.DataFrame(),
            'news_insights': pd.DataFrame()
        }
        
        # Process articles into health reports
        health_reports = []
        for article in articles:
            health_report = self._extract_health_insights_from_article(article)
            if health_report:
                health_reports.append(health_report)
        
        if health_reports:
            processed_data['health_reports'] = pd.DataFrame(health_reports)
        
        # Process water quality data
        if water_data:
            processed_data['water_quality'] = pd.DataFrame(water_data)
        
        # Process news insights
        news_insights = []
        for article in articles:
            insight = self._extract_news_insights(article)
            if insight:
                news_insights.append(insight)
        
        if news_insights:
            processed_data['news_insights'] = pd.DataFrame(news_insights)
        
        return processed_data
    
    def _extract_health_insights_from_article(self, article: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract health report insights from news article"""
        content = f"{article.get('title', '')} {article.get('content', '')}".lower()
        
        # Extract symptoms mentioned
        symptoms = {
            'fever': 'fever' in content or 'temperature' in content,
            'diarrhea': 'diarrhea' in content or 'loose motion' in content,
            'vomiting': 'vomiting' in content or 'vomit' in content,
            'nausea': 'nausea' in content or 'nauseous' in content,
            'abdominal_pain': 'stomach pain' in content or 'abdominal' in content,
            'dehydration': 'dehydration' in content or 'dehydrated' in content
        }
        
        # Calculate severity based on keywords
        severity_keywords = {
            1: ['mild', 'slight', 'minor'],
            2: ['moderate', 'some'],
            3: ['severe', 'serious', 'bad'],
            4: ['very severe', 'critical', 'emergency'],
            5: ['outbreak', 'epidemic', 'crisis']
        }
        
        severity = 1
        for level, keywords in severity_keywords.items():
            if any(keyword in content for keyword in keywords):
                severity = level
        
        # Only create report if symptoms are mentioned
        if any(symptoms.values()):
            return {
                'report_id': f"NEWS_{hash(article['url'])}",
                'user_id': 'news_analysis',
                'location_lat': self._extract_latitude_from_article(article),
                'location_lon': self._extract_longitude_from_article(article),
                'location_address': article.get('location', 'Unknown'),
                'fever': symptoms['fever'],
                'diarrhea': symptoms['diarrhea'],
                'vomiting': symptoms['vomiting'],
                'nausea': symptoms['nausea'],
                'abdominal_pain': symptoms['abdominal_pain'],
                'dehydration': symptoms['dehydration'],
                'symptom_severity': severity,
                'other_symptoms': self._extract_other_symptoms(content),
                'report_source': 'news_analysis',
                'report_timestamp': article.get('publish_date', datetime.now()),
                'article_url': article['url'],
                'confidence': self._calculate_confidence_score(article)
            }
        
        return None
    
    def _extract_news_insights(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """Extract general insights from news article"""
        content = f"{article.get('title', '')} {article.get('content', '')}".lower()
        
        return {
            'article_url': article['url'],
            'title': article['title'],
            'publish_date': article.get('publish_date'),
            'location': article.get('location'),
            'source': article.get('source'),
            'disease_mentioned': any(disease in content for disease in self.disease_keywords),
            'water_quality_mentioned': any(term in content for term in self.water_quality_keywords),
            'northeast_mentioned': any(term in content for term in self.northeast_keywords),
            'outbreak_mentioned': 'outbreak' in content or 'epidemic' in content,
            'severity_score': self._calculate_severity_score(content),
            'scraped_at': datetime.now()
        }
    
    def _extract_latitude_from_article(self, article: Dict[str, Any]) -> float:
        """Extract latitude from article content"""
        # Simple extraction - in real implementation, use geocoding
        content = f"{article.get('title', '')} {article.get('content', '')}"
        
        # Look for coordinate patterns
        lat_pattern = r'(\d+\.\d+)\s*°?\s*[Nn]'
        match = re.search(lat_pattern, content)
        if match:
            return float(match.group(1))
        
        # Default to northeast India coordinates
        return 26.0 + np.random.uniform(-2, 2)
    
    def _extract_longitude_from_article(self, article: Dict[str, Any]) -> float:
        """Extract longitude from article content"""
        content = f"{article.get('title', '')} {article.get('content', '')}"
        
        # Look for coordinate patterns
        lon_pattern = r'(\d+\.\d+)\s*°?\s*[Ee]'
        match = re.search(lon_pattern, content)
        if match:
            return float(match.group(1))
        
        # Default to northeast India coordinates
        return 91.0 + np.random.uniform(-2, 2)
    
    def _extract_other_symptoms(self, content: str) -> str:
        """Extract other symptoms mentioned in content"""
        other_symptoms = []
        
        symptom_patterns = [
            r'headache', r'body ache', r'weakness', r'fatigue',
            r'cramps', r'bloating', r'loss of appetite'
        ]
        
        for pattern in symptom_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                other_symptoms.append(pattern)
        
        return ', '.join(other_symptoms) if other_symptoms else None
    
    def _calculate_confidence_score(self, article: Dict[str, Any]) -> float:
        """Calculate confidence score for extracted data"""
        score = 0.0
        
        # Base score
        if article.get('title') and article.get('content'):
            score += 0.3
        
        # Location information
        if article.get('location'):
            score += 0.2
        
        # Publish date
        if article.get('publish_date'):
            score += 0.1
        
        # Content quality (length and keywords)
        content = f"{article.get('title', '')} {article.get('content', '')}"
        if len(content) > 500:
            score += 0.2
        
        keyword_count = sum(1 for keyword in self.disease_keywords + self.water_quality_keywords 
                           if keyword in content.lower())
        score += min(0.3, keyword_count * 0.05)
        
        return min(1.0, score)
    
    def _calculate_severity_score(self, content: str) -> float:
        """Calculate severity score from content"""
        severity_indicators = {
            'outbreak': 1.0,
            'epidemic': 1.0,
            'crisis': 0.9,
            'emergency': 0.8,
            'severe': 0.7,
            'serious': 0.6,
            'moderate': 0.4,
            'mild': 0.2
        }
        
        max_score = 0.0
        for indicator, score in severity_indicators.items():
            if indicator in content:
                max_score = max(max_score, score)
        
        return max_score
