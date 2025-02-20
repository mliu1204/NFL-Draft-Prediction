import logging
import os
import time
import undetected_chromedriver as uc
from selenium.webdriver.support import expected_conditions as EC
import random

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException

class SeleniumScraper:
    def __init__(self, use_proxy: bool = False):
        self.use_proxy = use_proxy
        self.setup_driver()
        
    def setup_driver(self, max_retries=3):
        """Initialize undetected Chrome WebDriver."""
        for attempt in range(max_retries):
            try:
                logging.info(f"Attempting to initialize undetected Chrome WebDriver (attempt {attempt + 1}/{max_retries})")
                
                # Configure Chrome options
                options = uc.ChromeOptions()
                
                # Disable images to improve load times
                prefs = {"profile.managed_default_content_settings.images": 2}
                options.add_experimental_option("prefs", prefs)
                
                options.add_argument("--window-size=1920,1080")
                options.add_argument("--no-sandbox")
                options.add_argument("--disable-dev-shm-usage")
                options.add_argument("--disable-gpu")
                options.add_argument("--mute-audio")
                
                # Suppress logging
                options.add_argument("--log-level=3")
                options.add_argument("--silent")
                
                # Add proxy if enabled
                if self.use_proxy:
                    proxy = "socks5://127.0.0.1:9050"  # Example using local SOCKS proxy
                    options.add_argument(f'--proxy-server={proxy}')
                
                # Initialize undetected Chrome WebDriver with version-specific config
                version_main = None  # Let undetected-chromedriver detect version
                self.driver = uc.Chrome(
                    options=options,
                    version_main=132,
                    driver_executable_path=None,  # Let it find the driver
                    browser_executable_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                    headless=False,  # Change to True if you want headless mode
                    use_subprocess=True
                )
                self.driver.set_page_load_timeout(30)
                logging.info("Undetected Chrome WebDriver initialized successfully")
                return
            except Exception as e:
                logging.error(f"Failed to initialize undetected Chrome WebDriver: {str(e)}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff
                
    def get_html(self, url: str) -> str:
        """Get HTML content from a given URL."""
        try:
            self.driver.get(url)
            time.sleep(random.randint(10, 20))
            logging.info(f"Navigated to {url}")
            # Wait until the target element is present (adjust the By selector to suit your needs)
            WebDriverWait(self.driver, 20).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "table"))
            )
            logging.info(f"Target element found on {url}")
            
            return self.driver.page_source
        except TimeoutException as te:
            logging.error(f"Timeout while waiting for the target element on {url}: {te}")
            raise
        except Exception as e:
            logging.error(f"Failed to get HTML from {url}: {str(e)}")
            raise
        
    def close(self):
        """Close the WebDriver."""
        if hasattr(self, 'driver'):
            self.driver.quit()
        logging.info("WebDriver closed successfully")
