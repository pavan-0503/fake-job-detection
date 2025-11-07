"""
Fake Job Detection - Web Scraper Module
Scrapes job posting details from various job portals
"""

import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urlparse
import time
import warnings
warnings.filterwarnings('ignore')

# Try to import Selenium for JavaScript-rendered sites
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False


class JobScraper:
    def __init__(self):
        """Initialize the web scraper"""
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        self.timeout = 15
        self.use_selenium = SELENIUM_AVAILABLE
    
    def _setup_selenium_driver(self):
        """Setup Selenium WebDriver for JavaScript-rendered sites"""
        try:
            from selenium.webdriver.chrome.service import Service
            from webdriver_manager.chrome import ChromeDriverManager
            
            chrome_options = Options()
            chrome_options.add_argument('--headless')  # Run in background
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--window-size=1920,1080')
            chrome_options.add_argument(f'user-agent={self.headers["User-Agent"]}')
            chrome_options.add_argument('--disable-blink-features=AutomationControlled')
            chrome_options.add_experimental_option('excludeSwitches', ['enable-automation'])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            
            # Disable images and CSS to speed up
            prefs = {
                'profile.managed_default_content_settings.images': 2,
                'profile.managed_default_content_settings.stylesheets': 2
            }
            chrome_options.add_experimental_option('prefs', prefs)
            
            # Auto-install ChromeDriver
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=chrome_options)
            driver.set_page_load_timeout(20)
            
            # Make Selenium undetectable
            driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            
            return driver
        except Exception as e:
            print(f"Selenium setup failed: {e}")
            return None
    
    def scrape_url(self, url):
        """
        Scrape job posting from a URL
        
        Args:
            url: Job posting URL
        
        Returns:
            dict: {
                'title': str,
                'company': str,
                'description': str,
                'location': str,
                'full_text': str,
                'source': str,
                'error': str (if any)
            }
        """
        try:
            # Parse URL to determine source
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            # Check if it's a JavaScript-heavy site
            js_sites = ['naukri', 'linkedin', 'indeed', 'glassdoor', 'monster', 'jobhai']
            needs_selenium = any(site in domain for site in js_sites)
            
            # Try Selenium first for known JS sites if available
            if needs_selenium and self.use_selenium:
                result = self._scrape_with_selenium(url)
                if result and len(result.get('full_text', '')) > 100:
                    return result
            
            # Determine scraping strategy based on domain
            if 'linkedin' in domain:
                return self._scrape_linkedin(url)
            elif 'naukri' in domain:
                return self._scrape_naukri(url)
            elif 'indeed' in domain:
                return self._scrape_indeed(url)
            elif 'timesjobs' in domain or 'timesascent' in domain:
                return self._scrape_timesjobs(url)
            elif 'shine' in domain:
                return self._scrape_shine(url)
            elif 'monster' in domain:
                return self._scrape_monster(url)
            elif 'foundit' in domain or 'monsterindia' in domain:
                return self._scrape_foundit(url)
            elif 'glassdoor' in domain:
                return self._scrape_glassdoor(url)
            elif 'jobhai' in domain:
                return self._scrape_jobhai(url)
            else:
                # Generic scraper for unknown sites
                return self._scrape_generic(url)
        
        except Exception as e:
            return {
                'title': '',
                'company': '',
                'description': '',
                'location': '',
                'full_text': '',
                'source': url,
                'error': f"Scraping failed: {str(e)}"
            }
    
    def _scrape_with_selenium(self, url):
        """
        Universal scraper using Selenium for JavaScript-rendered content
        """
        driver = None
        try:
            driver = self._setup_selenium_driver()
            if not driver:
                return None
            
            print(f"Using Selenium for: {url}")
            driver.get(url)
            
            # Wait for page to load - wait for body or main content
            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
                # Additional wait for dynamic content
                time.sleep(3)
            except:
                pass
            
            # Check for 404 or error pages
            page_title = driver.title.lower()
            current_url = driver.current_url.lower()
            page_text = driver.page_source.lower()
            
            # Common error indicators
            error_indicators = [
                '404', 'not found', 'page not found', 'error',
                'no longer available', 'job expired', 'job closed',
                'page doesn\'t exist', 'oops', 'sorry'
            ]
            
            if any(indicator in page_title for indicator in error_indicators):
                driver.quit()
                return {
                    'title': '',
                    'company': '',
                    'description': '',
                    'location': '',
                    'full_text': '',
                    'source': 'Selenium',
                    'error': f"Job posting not found (404 or removed). Page title: {driver.title}"
                }
            
            # LinkedIn-specific: Check if we landed on search results instead of job page
            if 'linkedin.com' in url and ('jobs in' in page_title or 'hardware design engineer jobs' in page_title or 'sign in' in page_title):
                # This is likely the search results page or login wall, not the actual job
                driver.quit()
                return {
                    'title': '',
                    'company': '',
                    'description': '',
                    'location': '',
                    'full_text': '',
                    'source': 'Selenium',
                    'error': f"Invalid LinkedIn URL - landed on search results or login page instead of job posting. The job may not exist or the URL is incorrect."
                }
            
            # Get page source after JavaScript execution
            page_source = driver.page_source
            soup = BeautifulSoup(page_source, 'html.parser')
            
            # Remove noise
            for element in soup(['script', 'style', 'noscript']):
                element.decompose()
            
            # Extract title
            title = ''
            title_elem = soup.find('h1') or soup.find('title')
            if title_elem:
                title = self._clean_text(title_elem.get_text())
            
            # Try structured data
            structured_data = self._extract_structured_data(soup)
            if structured_data and len(structured_data['full_text']) > 100:
                driver.quit()
                return structured_data
            
            # Extract all content
            all_text_parts = []
            
            # Get headings
            for heading in soup.find_all(['h1', 'h2', 'h3', 'h4']):
                text = self._clean_text(heading.get_text())
                if text and len(text) > 3:
                    all_text_parts.append(text)
            
            # Get paragraphs
            for p in soup.find_all('p'):
                text = self._clean_text(p.get_text())
                if text and len(text) > 10:
                    all_text_parts.append(text)
            
            # Get lists
            for li in soup.find_all('li'):
                text = self._clean_text(li.get_text())
                if text and len(text) > 5:
                    all_text_parts.append(text)
            
            # Get spans with substantial text
            for span in soup.find_all('span'):
                text = self._clean_text(span.get_text())
                if text and 20 < len(text) < 500:
                    all_text_parts.append(text)
            
            description = ' '.join(all_text_parts)
            
            # Extract company and location
            company = self._extract_company(soup)
            location = self._extract_location(soup)
            
            full_text = f"{title} {company} {location} {description}"
            
            driver.quit()
            
            return {
                'title': title,
                'company': company,
                'description': description,
                'location': location,
                'full_text': full_text,
                'source': 'Selenium',
                'error': None
            }
            
        except Exception as e:
            if driver:
                driver.quit()
            print(f"Selenium scraping failed: {e}")
            return None
    
    def _fetch_page(self, url):
        """Fetch webpage content"""
        response = requests.get(url, headers=self.headers, timeout=self.timeout)
        response.raise_for_status()
        return BeautifulSoup(response.content, 'html.parser')
    
    def _clean_text(self, text):
        """Clean extracted text"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def _scrape_linkedin(self, url):
        """Scrape LinkedIn job posting"""
        try:
            soup = self._fetch_page(url)
            
            # Extract title
            title_elem = soup.find('h1', class_=re.compile('job.*title|title')) or \
                        soup.find('h1') or \
                        soup.find('title')
            title = self._clean_text(title_elem.get_text()) if title_elem else ''
            
            # Extract company
            company_elem = soup.find('a', class_=re.compile('company')) or \
                          soup.find('span', class_=re.compile('company'))
            company = self._clean_text(company_elem.get_text()) if company_elem else ''
            
            # Extract location
            location_elem = soup.find('span', class_=re.compile('location|city'))
            location = self._clean_text(location_elem.get_text()) if location_elem else ''
            
            # Extract description
            desc_elem = soup.find('div', class_=re.compile('description|job-description')) or \
                       soup.find('section', class_=re.compile('description'))
            description = self._clean_text(desc_elem.get_text()) if desc_elem else ''
            
            # Combine all text
            full_text = f"{title} {company} {location} {description}"
            
            return {
                'title': title,
                'company': company,
                'description': description,
                'location': location,
                'full_text': full_text,
                'source': 'LinkedIn',
                'error': None
            }
        except Exception as e:
            # Fallback to generic scraper
            return self._scrape_generic(url)
            company = self._clean_text(company_elem.get_text()) if company_elem else ''
            
            # Extract location
            location_elem = soup.find('span', class_=re.compile('location|city'))
            location = self._clean_text(location_elem.get_text()) if location_elem else ''
            
            # Extract description
            desc_elem = soup.find('div', class_=re.compile('description|job-description')) or \
                       soup.find('section', class_=re.compile('description'))
            description = self._clean_text(desc_elem.get_text()) if desc_elem else ''
            
            # Combine all text
            full_text = f"{title} {company} {location} {description}"
            
            return {
                'title': title,
                'company': company,
                'description': description,
                'location': location,
                'full_text': full_text,
                'source': 'LinkedIn',
                'error': None
            }
        except Exception as e:
            return self._scrape_generic(url)
    
    def _scrape_naukri(self, url):
        """Scrape Naukri.com job posting"""
        try:
            soup = self._fetch_page(url)
            
            # Title - try multiple selectors
            title = ''
            title_elem = soup.find('h1', class_=re.compile('title|jd-header-title')) or \
                        soup.find('h1') or \
                        soup.find('span', class_=re.compile('title'))
            if title_elem:
                title = self._clean_text(title_elem.get_text())
            
            # Company - try multiple selectors
            company = ''
            company_elem = soup.find('a', class_=re.compile('company|comp-name')) or \
                          soup.find('div', class_=re.compile('company|comp-name')) or \
                          soup.find('span', class_=re.compile('company'))
            if company_elem:
                company = self._clean_text(company_elem.get_text())
            
            # Location - try multiple selectors
            location = ''
            location_elem = soup.find('span', class_=re.compile('location|loc-name')) or \
                           soup.find('a', class_=re.compile('location')) or \
                           soup.find('div', class_=re.compile('location'))
            if location_elem:
                location = self._clean_text(location_elem.get_text())
            
            # Description - try multiple selectors for Naukri's structure
            description = ''
            desc_elem = soup.find('div', class_=re.compile('job.*desc|jd-desc|description|styles_JDC')) or \
                       soup.find('section', class_=re.compile('job.*desc|description')) or \
                       soup.find('div', itemprop='description')
            
            if desc_elem:
                description = self._clean_text(desc_elem.get_text())
            
            # If description is empty or too short, try broader extraction
            if len(description) < 100:
                # Try getting all relevant content sections
                sections = soup.find_all(['section', 'article', 'div'], class_=re.compile('detail|content|jd|desc'))
                desc_parts = []
                for section in sections:
                    text = self._clean_text(section.get_text())
                    if len(text) > 50:
                        desc_parts.append(text)
                if desc_parts:
                    description = ' '.join(desc_parts)
            
            full_text = f"{title} {company} {location} {description}"
            
            # If still too short, fallback to generic scraper
            if len(full_text.strip()) < 100:
                return self._scrape_generic(url)
            
            return {
                'title': title,
                'company': company,
                'description': description,
                'location': location,
                'full_text': full_text,
                'source': 'Naukri',
                'error': None
            }
        except Exception as e:
            return self._scrape_generic(url)
    
    def _scrape_indeed(self, url):
        """Scrape Indeed job posting"""
        try:
            soup = self._fetch_page(url)
            
            # Title
            title_elem = soup.find('h1', class_=re.compile('jobTitle|job-title')) or \
                        soup.find('h1')
            title = self._clean_text(title_elem.get_text()) if title_elem else ''
            
            # Company
            company_elem = soup.find('div', class_=re.compile('company')) or \
                          soup.find('span', class_=re.compile('company'))
            company = self._clean_text(company_elem.get_text()) if company_elem else ''
            
            # Location
            location_elem = soup.find('div', class_=re.compile('location'))
            location = self._clean_text(location_elem.get_text()) if location_elem else ''
            
            # Description
            desc_elem = soup.find('div', id=re.compile('jobDescriptionText')) or \
                       soup.find('div', class_=re.compile('description'))
            description = self._clean_text(desc_elem.get_text()) if desc_elem else ''
            
            full_text = f"{title} {company} {location} {description}"
            
            return {
                'title': title,
                'company': company,
                'description': description,
                'location': location,
                'full_text': full_text,
                'source': 'Indeed',
                'error': None
            }
        except Exception as e:
            return self._scrape_generic(url)
            return self._scrape_generic(url)
    
    def _scrape_timesjobs(self, url):
        """Scrape TimesJobs posting"""
        try:
            soup = self._fetch_page(url)
            
            # Title
            title_elem = soup.find('h1') or soup.find('title')
            title = self._clean_text(title_elem.get_text()) if title_elem else ''
            
            # Company
            company_elem = soup.find('h2', class_=re.compile('company'))
            company = self._clean_text(company_elem.get_text()) if company_elem else ''
            
            # Location
            location_elem = soup.find('span', class_=re.compile('location'))
            location = self._clean_text(location_elem.get_text()) if location_elem else ''
            
            # Description
            desc_elem = soup.find('div', class_=re.compile('jd-sec|job-description'))
            description = self._clean_text(desc_elem.get_text()) if desc_elem else ''
            
            full_text = f"{title} {company} {location} {description}"
            
            return {
                'title': title,
                'company': company,
                'description': description,
                'location': location,
                'full_text': full_text,
                'source': 'TimesJobs',
                'error': None
            }
        except Exception as e:
            return self._scrape_generic(url)
    
    def _scrape_shine(self, url):
        """Scrape Shine.com posting"""
        try:
            soup = self._fetch_page(url)
            
            # Title
            title_elem = soup.find('h1') or soup.find('title')
            title = self._clean_text(title_elem.get_text()) if title_elem else ''
            
            # Company
            company_elem = soup.find('div', class_=re.compile('company'))
            company = self._clean_text(company_elem.get_text()) if company_elem else ''
            
            # Location
            location_elem = soup.find('div', class_=re.compile('location'))
            location = self._clean_text(location_elem.get_text()) if location_elem else ''
            
            # Description
            desc_elem = soup.find('div', class_=re.compile('jd'))
            description = self._clean_text(desc_elem.get_text()) if desc_elem else ''
            
            full_text = f"{title} {company} {location} {description}"
            
            return {
                'title': title,
                'company': company,
                'description': description,
                'location': location,
                'full_text': full_text,
                'source': 'Shine',
                'error': None
            }
        except Exception as e:
            return self._scrape_generic(url)
    
    def _scrape_monster(self, url):
        """Scrape Monster.com job posting"""
        try:
            soup = self._fetch_page(url)
            
            title_elem = soup.find('h1') or soup.find('title')
            title = self._clean_text(title_elem.get_text()) if title_elem else ''
            
            company_elem = soup.find('span', class_=re.compile('company'))
            company = self._clean_text(company_elem.get_text()) if company_elem else ''
            
            location_elem = soup.find('span', class_=re.compile('location'))
            location = self._clean_text(location_elem.get_text()) if location_elem else ''
            
            desc_elem = soup.find('div', class_=re.compile('job.*desc|description'))
            description = self._clean_text(desc_elem.get_text()) if desc_elem else ''
            
            full_text = f"{title} {company} {location} {description}"
            
            return {
                'title': title,
                'company': company,
                'description': description,
                'location': location,
                'full_text': full_text,
                'source': 'Monster',
                'error': None
            }
        except Exception as e:
            return self._scrape_generic(url)
    
    def _scrape_foundit(self, url):
        """Scrape Foundit (MonsterIndia) job posting"""
        try:
            soup = self._fetch_page(url)
            
            title_elem = soup.find('h1') or soup.find('title')
            title = self._clean_text(title_elem.get_text()) if title_elem else ''
            
            company_elem = soup.find('div', class_=re.compile('company'))
            company = self._clean_text(company_elem.get_text()) if company_elem else ''
            
            location_elem = soup.find('div', class_=re.compile('location'))
            location = self._clean_text(location_elem.get_text()) if location_elem else ''
            
            desc_elem = soup.find('div', class_=re.compile('description'))
            description = self._clean_text(desc_elem.get_text()) if desc_elem else ''
            
            full_text = f"{title} {company} {location} {description}"
            
            return {
                'title': title,
                'company': company,
                'description': description,
                'location': location,
                'full_text': full_text,
                'source': 'Foundit',
                'error': None
            }
        except Exception as e:
            return self._scrape_generic(url)
    
    def _scrape_glassdoor(self, url):
        """Scrape Glassdoor job posting"""
        try:
            soup = self._fetch_page(url)
            
            title_elem = soup.find('h1') or soup.find('div', class_=re.compile('job.*title'))
            title = self._clean_text(title_elem.get_text()) if title_elem else ''
            
            company_elem = soup.find('div', class_=re.compile('employer'))
            company = self._clean_text(company_elem.get_text()) if company_elem else ''
            
            location_elem = soup.find('div', class_=re.compile('location'))
            location = self._clean_text(location_elem.get_text()) if location_elem else ''
            
            desc_elem = soup.find('div', class_=re.compile('desc|job.*desc'))
            description = self._clean_text(desc_elem.get_text()) if desc_elem else ''
            
            full_text = f"{title} {company} {location} {description}"
            
            return {
                'title': title,
                'company': company,
                'description': description,
                'location': location,
                'full_text': full_text,
                'source': 'Glassdoor',
                'error': None
            }
        except Exception as e:
            return self._scrape_generic(url)
    
    def _scrape_jobhai(self, url):
        """Scrape JobHai.com job posting"""
        try:
            soup = self._fetch_page(url)
            
            title_elem = soup.find('h1') or soup.find('div', class_=re.compile('job.*title'))
            title = self._clean_text(title_elem.get_text()) if title_elem else ''
            
            company_elem = soup.find('span', class_=re.compile('company')) or soup.find('div', class_=re.compile('company'))
            company = self._clean_text(company_elem.get_text()) if company_elem else ''
            
            location_elem = soup.find('span', class_=re.compile('location')) or soup.find('div', class_=re.compile('location'))
            location = self._clean_text(location_elem.get_text()) if location_elem else ''
            
            desc_elem = soup.find('div', class_=re.compile('desc|detail|content'))
            description = self._clean_text(desc_elem.get_text()) if desc_elem else ''
            
            full_text = f"{title} {company} {location} {description}"
            
            return {
                'title': title,
                'company': company,
                'description': description,
                'location': location,
                'full_text': full_text,
                'source': 'JobHai',
                'error': None
            }
        except Exception as e:
            return self._scrape_generic(url)
    
    def _scrape_generic(self, url):
        """
        Universal scraper - extracts ALL content and lets predictor decide if it's a job
        """
        try:
            soup = self._fetch_page(url)
            
            # Remove noise elements but keep content
            for element in soup(['script', 'style', 'noscript']):
                element.decompose()
            
            # Extract title
            title = ''
            title_elem = soup.find('h1')
            if title_elem:
                title = self._clean_text(title_elem.get_text())
            if not title:
                title_elem = soup.find('title')
                if title_elem:
                    title = self._clean_text(title_elem.get_text())
            
            # Try structured data first (best quality)
            structured_data = self._extract_structured_data(soup)
            if structured_data:
                return structured_data
            
            # Strategy: Extract ALL text content from the page
            # Let the predictor's is_job_description() validate it
            
            all_text_parts = []
            
            # 1. Get all headings (h1-h6) - usually contain important info
            for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                text = self._clean_text(heading.get_text())
                if text and len(text) > 3:
                    all_text_parts.append(text)
            
            # 2. Get all paragraphs
            for p in soup.find_all('p'):
                text = self._clean_text(p.get_text())
                if text and len(text) > 10:
                    all_text_parts.append(text)
            
            # 3. Get all list items (requirements, responsibilities)
            for li in soup.find_all('li'):
                text = self._clean_text(li.get_text())
                if text and len(text) > 5:
                    all_text_parts.append(text)
            
            # 4. Get all table cells (salary, location often in tables)
            for td in soup.find_all(['td', 'th']):
                text = self._clean_text(td.get_text())
                if text and len(text) > 3:
                    all_text_parts.append(text)
            
            # 5. Get all divs and spans with substantial text
            for elem in soup.find_all(['div', 'span', 'section', 'article']):
                # Get direct text only (not nested)
                text = self._clean_text(elem.get_text())
                # Only add if it has reasonable length and not already captured
                if text and 20 < len(text) < 2000:
                    all_text_parts.append(text)
            
            # 6. Get all strong/bold text (often highlights important info)
            for elem in soup.find_all(['strong', 'b', 'em']):
                text = self._clean_text(elem.get_text())
                if text and len(text) > 3:
                    all_text_parts.append(text)
            
            # Combine all extracted text
            description = ' '.join(all_text_parts)
            
            # Remove duplicates (same text appearing multiple times)
            words = description.split()
            # Keep order but remove excessive repetition
            seen = set()
            unique_words = []
            for word in words:
                word_lower = word.lower()
                if word_lower not in seen or len(seen) < 100:  # Allow some repetition
                    unique_words.append(word)
                    seen.add(word_lower)
            
            description = ' '.join(unique_words)
            
            # Try to extract company and location
            company = self._extract_company(soup)
            location = self._extract_location(soup)
            
            # Build full text
            full_text = f"{title} {company} {location} {description}"
            
            # No validation here - let predictor handle it
            # Always return the scraped content
            return {
                'title': title,
                'company': company,
                'description': description,
                'location': location,
                'full_text': full_text,
                'source': 'Generic',
                'error': None
            }
            
        except Exception as e:
            # Even on error, try to return something
            return {
                'title': '',
                'company': '',
                'description': '',
                'location': '',
                'full_text': '',
                'source': 'Generic',
                'error': f"Scraping failed: {str(e)}. Please copy and paste the job description text directly."
            }
    
    def _extract_structured_data(self, soup):
        """Extract job data from JSON-LD structured data"""
        try:
            scripts = soup.find_all('script', type='application/ld+json')
            for script in scripts:
                import json
                data = json.loads(script.string)
                
                # Check if it's a JobPosting schema
                if isinstance(data, dict) and data.get('@type') == 'JobPosting':
                    title = data.get('title', '')
                    company = data.get('hiringOrganization', {}).get('name', '') if isinstance(data.get('hiringOrganization'), dict) else ''
                    description = data.get('description', '')
                    location_data = data.get('jobLocation', {})
                    location = location_data.get('address', {}).get('addressLocality', '') if isinstance(location_data, dict) else ''
                    
                    full_text = f"{title} {company} {location} {description}"
                    
                    return {
                        'title': title,
                        'company': company,
                        'description': description,
                        'location': location,
                        'full_text': full_text,
                        'source': 'Structured Data',
                        'error': None
                    }
        except:
            pass
        return None
    
    def _extract_company(self, soup):
        """Extract company name from various sources"""
        # Try meta tags first
        meta_company = soup.find('meta', attrs={'name': re.compile('company|organization', re.I)}) or \
                      soup.find('meta', attrs={'property': re.compile('company|organization', re.I)})
        if meta_company:
            return self._clean_text(meta_company.get('content', ''))
        
        # Try common class names
        company_elem = soup.find(class_=re.compile('company|employer|organization', re.I)) or \
                      soup.find(id=re.compile('company|employer', re.I))
        if company_elem:
            return self._clean_text(company_elem.get_text())
        
        return ''
    
    def _extract_location(self, soup):
        """Extract location from various sources"""
        # Try meta tags
        meta_location = soup.find('meta', attrs={'name': re.compile('location|locality', re.I)}) or \
                       soup.find('meta', attrs={'property': re.compile('location', re.I)})
        if meta_location:
            return self._clean_text(meta_location.get('content', ''))
        
        # Try common class names
        location_elem = soup.find(class_=re.compile('location|locality|city|address', re.I)) or \
                       soup.find(id=re.compile('location|city', re.I))
        if location_elem:
            return self._clean_text(location_elem.get_text())
        
        return ''


# For standalone testing
if __name__ == "__main__":
    print("=" * 60)
    print("🧪 TESTING JOB SCRAPER")
    print("=" * 60)
    
    scraper = JobScraper()
    
    # Test with a sample URL (replace with actual job posting URL)
    test_url = "https://example.com/job-posting"
    
    print(f"\n📡 Scraping: {test_url}")
    print("-" * 60)
    
    result = scraper.scrape_url(test_url)
    
    if result.get('error'):
        print(f"❌ Error: {result['error']}")
    else:
        print(f"✓ Source: {result['source']}")
        print(f"✓ Title: {result['title'][:100]}...")
        print(f"✓ Company: {result['company'][:100]}...")
        print(f"✓ Location: {result['location'][:100]}...")
        print(f"✓ Description length: {len(result['description'])} characters")
        print(f"✓ Full text length: {len(result['full_text'])} characters")
    
    print("\n" + "=" * 60)
    print("✅ Testing Complete!")
    print("=" * 60)
