import requests
from bs4 import BeautifulSoup
import csv
import time
import urllib.parse
from datetime import datetime, timedelta
import re

class YahooSearchScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.results = []
    
    def search_yahoo(self, keyword, top_results=5):
        """Search Yahoo for a keyword and return top N results from last 24 hours"""
        print(f"Searching for: {keyword}")
        
        # URL encode the keyword
        encoded_keyword = urllib.parse.quote(keyword)
        
        # Yahoo search URL with time filter for last 24 hours
        url = f"https://news.search.yahoo.com/search?p={encoded_keyword}&fr=yfp-t&fp=1&toggle=1&cop=mss&ei=UTF-8&age=1d"
        print(url)
        
        page_results = []
        
        try:
            print(f"  Fetching top {top_results} results...")
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            print(response.content)
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find search result containers
            search_results = soup.select('div.dd')
            # print(search_results)
            
            if not search_results:
                print(f"  No results found")
                return page_results
            
            # Limit to top N results
            for i, result in enumerate(search_results[:top_results]):
                try:
                    # Extract title
                    title_elem = result.find('h4')
                    if title_elem:
                        title_link = title_elem.find('a')
                        if title_link:
                            title = title_link.get_text(strip=True)
                            yahoo_url = title_link.get('href', '')
                            match = re.search(r'RU=([^/&]+)', yahoo_url)
                            if match:
                                encoded_url = match.group(1)
                                # URL decode the extracted URL
                                result_url = urllib.parse.unquote(encoded_url)
                                # Handle double encoding if present
                                if result_url.startswith('http%'):
                                    result_url = urllib.parse.unquote(result_url)
                        else:
                            continue
                    else:
                        continue
                    
                    # Clean up URL if it's a Yahoo redirect
                    if result_url.startswith('/'):
                        continue
                    
                    # Extract snippet/description
                    # snippet_elem = result.find('div', class_='compText')
                    # snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                    
                    page_results.append({
                        'keyword': keyword,
                        'title': title,
                        'url': result_url,
                        # 'snippet': snippet,
                        # 'search_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    })
                    
                except Exception as e:
                    print(f"    Error parsing result {i+1}: {e}")
                    continue
                
        except Exception as e:
            print(f"  Error fetching results: {e}")
        
        print(f"  Found {len(page_results)} results for '{keyword}'")
        return page_results
    
    def scrape_multiple_keywords(self, keywords, top_results=5):
        """Scrape Yahoo search results for multiple keywords"""
        all_results = []
        
        for i, keyword in enumerate(keywords, 1):
            print(f"\n[{i}/{len(keywords)}] Processing keyword: {keyword}")
            
            try:
                results = self.search_yahoo(keyword, top_results)
                all_results.extend(results)
                
                # Add delay between different keyword searches
                if i < len(keywords):
                    print("  Waiting before next search...")
                    time.sleep(3)
                    
            except Exception as e:
                print(f"  Error searching for '{keyword}': {e}")
                continue
        
        return all_results
    
    def save_to_csv(self, results, filename='yahoo_search_results.csv'):
        """Save results to CSV file"""
        if not results:
            print("No results to save.")
            return
        
        fieldnames = ['keyword', 'title', 'url']
        
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
            
            print(f"\nResults saved to '{filename}'")
            print(f"Total results: {len(results)}")
            
        except Exception as e:
            print(f"Error saving to CSV: {e}")

def main():
    # List of keywords to search for
    keywords = [
        "your-keyword-here"
    ]
    
    # You can modify this list with your own keywords
    print("Yahoo Search Scraper - Last 24 Hours")
    print("=" * 50)
    
    # Create scraper instance
    scraper = YahooSearchScraper()
    
    # Scrape results for all keywords (top 5 results each)
    all_results = scraper.scrape_multiple_keywords(keywords, top_results=5)
    
    # Save results to CSV
    scraper.save_to_csv(all_results, 'yahoo_search_results_24h.csv')
    
    # Display summary
    if all_results:
        print(f"\nSummary:")
        print(f"Total keywords searched: {len(keywords)}")
        print(f"Total results found: {len(all_results)}")
        
        # Show results per keyword
        keyword_counts = {}
        for result in all_results:
            keyword = result['keyword']
            keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
        
        print("\nResults per keyword:")
        for keyword, count in keyword_counts.items():
            print(f"  {keyword}: {count} results")

if __name__ == "__main__":
    main()