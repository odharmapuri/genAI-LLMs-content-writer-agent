# This script uses Google Gemini 1.5 Pro to enrich news articles with real-time search data, generating SEO-optimized blog content.
# It fetches the article content, summarizes it, enriches it with real-time search results
# and generates a comprehensive blog article with citations and sources.
# It uses SerpAPI for real-time Google search and news search, and BeautifulSoup for parsing HTML content.
# Make sure to set up your environment variables in a .env file with GOOGLE_API_KEY and SERPAPI_KEY.

import google.generativeai as genai
import os
import requests
from bs4 import BeautifulSoup
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
import time
from dotenv import load_dotenv
import re
import json

# Load environment variables from .env file
load_dotenv(dotenv_path="../../creds.env")

# --- Configuration ---
API_KEY = os.getenv("GOOGLE_API_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")  # You'll need to get this from serpapi.com

if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found. Please set it in your .env file.")

if not SERPAPI_KEY:
    print("Warning: SERPAPI_KEY not found. Real-time search will be limited.")
    print("Get your free API key from https://serpapi.com/")

genai.configure(api_key=API_KEY)

# Choose your Gemini model
GEMINI_MODEL = 'gemini-1.5-flash' # or 'gemini-1.5-pro'

# Initialize the Gemini model
model = genai.GenerativeModel(model_name=GEMINI_MODEL)

# --- Helper Functions ---

def search_google(query, num_results=5):
    """
    Performs real-time Google search using SerpAPI
    Returns search results with titles, snippets, and links
    """
    if not SERPAPI_KEY:
        print(f"Skipping search for '{query}' - No SERPAPI_KEY provided")
        return []
    
    try:
        url = "https://serpapi.com/search"
        params = {
            "engine": "google",
            "q": query,
            "api_key": SERPAPI_KEY,
            "num": num_results,
            "hl": "en",
            "gl": "us"
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        results = []
        if "organic_results" in data:
            for result in data["organic_results"]:
                results.append({
                    "title": result.get("title", ""),
                    "snippet": result.get("snippet", ""),
                    "link": result.get("link", ""),
                    "date": result.get("date", "")
                })
        print(results)
        if not results:
            print(f"No search results found for '{query}'")
        return results
    
    except Exception as e:
        print(f"Error during search for '{query}': {e}")
        return []

def search_news(query, num_results=5):
    """
    Performs real-time news search using SerpAPI
    Returns recent news results
    """
    if not SERPAPI_KEY:
        print(f"Skipping news search for '{query}' - No SERPAPI_KEY provided")
        return []
    
    try:
        url = "https://serpapi.com/search"
        params = {
            "engine": "google",
            "q": query,
            "api_key": SERPAPI_KEY,
            "tbm": "nws",  # News search
            "num": num_results,
            "hl": "en",
            "gl": "us"
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        results = []
        if "news_results" in data:
            for result in data["news_results"]:
                results.append({
                    "title": result.get("title", ""),
                    "snippet": result.get("snippet", ""),
                    "link": result.get("link", ""),
                    "date": result.get("date", ""),
                    "source": result.get("source", "")
                })
        print(results)
        if not results:
            print(f"No news results found for '{query}'")
        return results
    
    except Exception as e:
        print(f"Error during news search for '{query}': {e}")
        return []

@retry(wait=wait_exponential(multiplier=1, min=4, max=60), stop=stop_after_attempt(5),
       retry=retry_if_exception_type(Exception))
def generate_content_with_retry(prompt_parts, generation_config=None):
    """
    Generates content using Gemini, with retry logic for transient errors like rate limits.
    """
    try:
        response = model.generate_content(prompt_parts, generation_config=generation_config)
        # Check for empty response or safety blocks
        if not response.text:
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                print(f"Content generation blocked: {response.prompt_feedback.block_reason}")
            if response.candidates:
                for candidate in response.candidates:
                    if candidate.finish_reason == genai.protos.Candidate.FinishReason.SAFETY:
                        print(f"Candidate blocked due to safety settings.")
            raise ValueError("Empty response or blocked content from Gemini API.")
        return response
    except Exception as e:
        print(f"Error during content generation: {e}. Retrying...")
        raise

def fetch_article_content(url):
    """
    Fetches the main text content from a given URL.
    Uses BeautifulSoup to parse HTML and extract paragraphs.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Attempt to find common article content containers
        article_text = []
        for tag in ['p', 'h1', 'h2', 'h3', 'li']:
            for element in soup.find_all(tag):
                if element.get_text(strip=True):
                    article_text.append(element.get_text(strip=True))

        full_text = "\n".join(article_text)

        # Try to find a more specific article body if available
        article_body = soup.find('article') or soup.find(class_=re.compile(r'article|content|body|main'))
        if article_body:
            paragraphs = article_body.find_all(['p', 'h1', 'h2', 'h3', 'li'])
            clean_text = "\n".join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
            if len(clean_text) > len(full_text) * 0.5:
                full_text = clean_text

        if not full_text or len(full_text) < 200:
            print(f"Warning: Could not extract sufficient content from {url}. Raw text length: {len(full_text)}")
            return None

        return full_text[:900000]

    except requests.exceptions.RequestException as e:
        print(f"Error fetching article from {url}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while parsing {url}: {e}")
        return None

def summarize_article(article_content):
    """
    Summarizes the given article content into key points.
    """
    prompt = f"""
    Read the following news article and extract the most important summary points.
    Focus on the core facts and events. Present them as a concise bulleted list.

    Article:
    {article_content}

    Summary Points:
    """
    print("Summarizing article...")
    response = generate_content_with_retry([prompt])
    return response.text.strip()

def extract_search_queries(summary_points):
    """
    Extracts relevant search queries from summary points to gather real-time information
    """
    prompt = f"""
    Based on the following summary points from a news article, generate 3-5 specific search queries 
    that would help find the most recent and relevant information to enrich the content.
    
    Focus on:
    - Recent developments or updates
    - Statistical data and facts
    - Expert opinions and quotes
    - Background context and causes
    - Related events or impacts
    
    Return only the search queries, one per line, without numbering or bullets.
    
    Summary Points:
    {summary_points}
    
    Search Queries:
    """
    
    response = generate_content_with_retry([prompt])
    queries = [q.strip() for q in response.text.strip().split('\n') if q.strip()]
    return queries[:5]  # Limit to 5 queries to avoid rate limits

def enrich_summary_points(summary_points_str):
    """
    Enriches the summary points using real-time search for latest information,
    data, stats, and expert opinions.
    """
    print("\nExtracting search queries from summary points...")
    search_queries = extract_search_queries(summary_points_str)
    print(f"Generated search queries: {search_queries}")
    
    # Perform searches
    all_search_results = []
    for query in search_queries:
        print(f"Searching for: {query}")
        
        # Search both general results and news
        general_results = search_google(query, 3)
        news_results = search_news(query, 2)
        
        if general_results or news_results:
            all_search_results.append({
                "query": query,
                "general_results": general_results,
                "news_results": news_results
            })
        
        # Add small delay to respect rate limits
        time.sleep(1)
    
    # Format search results for AI processing
    search_context = "\n\n--- REAL-TIME SEARCH RESULTS ---\n"
    sources_used = []
    
    for search_data in all_search_results:
        search_context += f"\nQuery: {search_data['query']}\n"
        
        if search_data['general_results']:
            search_context += "General Results:\n"
            for i, result in enumerate(search_data['general_results'], 1):
                search_context += f"{i}. {result['title']}\n"
                search_context += f"   {result['snippet']}\n"
                search_context += f"   Source: {result['link']}\n"
                if result['link'] not in sources_used:
                    sources_used.append(result['link'])
        
        if search_data['news_results']:
            search_context += "Recent News:\n"
            for i, result in enumerate(search_data['news_results'], 1):
                search_context += f"{i}. {result['title']}\n"
                search_context += f"   {result['snippet']}\n"
                search_context += f"   Source: {result['source']} - {result['link']}\n"
                search_context += f"   Date: {result.get('date', 'Recent')}\n"
                if result['link'] not in sources_used:
                    sources_used.append(result['link'])
    
    # Now enrich the summary points with real-time data
    prompt = f"""
    You are an expert journalist and researcher. I will provide you with summary points from a news article 
    along with real-time search results. Your task is to enrich each summary point by incorporating the 
    latest information from the search results.

    Add the following details using the search results:
    - **What:** More specific details and recent developments
    - **Why:** The significance, causes, or underlying reasons with latest context
    - **When:** Precise dates, timelines, or recent updates from search results
    - **Where:** Specific locations or contexts with current information
    - **How:** The process, methods, or mechanisms with latest insights
    - **Data/Stats/Quotes:** Recent factual data, statistics, or expert quotes from search results

    IMPORTANT: 
    - Always cite sources when using information from search results
    - Use format: "According to [Source Name]" or "Recent data from [Source]"
    - Prioritize the most recent and credible information
    - Integrate search results naturally into the enriched points

    Current Date: {time.strftime('%Y-%m-%d')}

    Original Summary Points:
    {summary_points_str}

    {search_context}

    Enriched Summary Points (with real-time data and citations):
    """
    
    print("\nEnriching summary points with real-time search data...")
    response = generate_content_with_retry([prompt])
    
    return response.text.strip(), sources_used

def generate_blog_article(enriched_summary_points_str, article_title_suggestion="", sources_used=[]):
    """
    Generates a comprehensive, SEO-optimized blog article from enriched summary points.
    """
    sources_text = ""
    if sources_used:
        sources_text = f"\n\nSources consulted during research:\n" + "\n".join([f"- {source}" for source in sources_used[:10]])
    
    prompt = f"""
    You are a professional SEO content writer for a top-ranking blog.
    Your goal is to write a comprehensive, engaging, and highly SEO-optimized blog article
    based on the following enriched summary points that include real-time data and research.

    **Instructions for SEO and Readability:**
    1. **Headline:** Create a compelling, keyword-rich headline (H1) that attracts clicks
    2. **Introduction:** Strong hook with clear value proposition
    3. **Subheadings (H2, H3):** Descriptive and keyword-rich subheadings
    4. **Content Structure:**
        * Expand each enriched point into informative paragraphs
        * Use bullet points for lists or key takeaways
        * Keep paragraphs short (3-5 sentences)
        * Maintain conversational tone
    5. **Data/Stats/Quotes:** Integrate citations naturally and prominently
    6. **Call to Action:** Soft CTA at the end
    7. **Featured Image Idea:** Suggest compelling visual concept

    **Enriched Summary Points (with real-time data):**
    {enriched_summary_points_str}

    **Original Article Title (for inspiration):**
    "{article_title_suggestion}"
    {sources_text}

    **Output Format (Markdown):**

    # [SEO-Optimized Headline]

    [Introduction - 2-3 paragraphs]

    ## [Subheading 1]
    [Content with integrated citations]

    ## [Subheading 2]
    [Content continuing the story]

    ## Conclusion
    [Summary and soft CTA]

    ---

    **Featured Image Idea:** [Description]

    **Key Sources:**
    [List main sources cited in article]
    """
    
    print("\nGenerating comprehensive blog article with real-time data...")
    generation_config = {
        "temperature": 0.7,
        "max_output_tokens": 4096
    }
    response = generate_content_with_retry([prompt], generation_config=generation_config)
    return response.text.strip()

# --- Main Execution Flow ---
def main():
    print("=== AI-Powered Article Enrichment with Real-Time Search ===\n")
    
    if not SERPAPI_KEY:
        print("⚠️  For real-time search functionality, you need a SerpAPI key:")
        print("1. Go to https://serpapi.com/")
        print("2. Sign up for a free account (100 searches/month)")
        print("3. Add SERPAPI_KEY=your_key_here to your creds.env file")
        print("4. Re-run this script\n")
        
        proceed = input("Continue without real-time search? (y/n): ").lower()
        if proceed != 'y':
            return
    
    article_url = input("Please enter the URL of the news article: ").strip()
    if not article_url:
        print("No URL provided. Exiting.")
        return

    original_article_title = input("Enter the original article title (optional): ").strip()

    print(f"\nFetching content from: {article_url}")
    article_content = fetch_article_content(article_url)

    if not article_content:
        print("Failed to retrieve article content. Cannot proceed.")
        return

    # Step 1: Summarize
    print("\n--- Step 1: Summarizing Article ---")
    summary_points = summarize_article(article_content)
    print("\nGenerated Summary Points:")
    print(summary_points)

    # Step 2: Enrich with Real-time Search
    print("\n--- Step 2: Enriching with Real-Time Search Data ---")
    enriched_points, sources_used = enrich_summary_points(summary_points)
    print("\nEnriched Summary Points:")
    print(enriched_points)

    if sources_used:
        print(f"\nSources Used ({len(sources_used)} total):")
        for i, source in enumerate(sources_used[:5], 1):  # Show first 5
            print(f"{i}. {source}")
        if len(sources_used) > 5:
            print(f"... and {len(sources_used) - 5} more sources")

    # Step 3: Generate Blog Article
    print("\n--- Step 3: Generating SEO-Optimized Blog Article ---")
    blog_article_markdown = generate_blog_article(enriched_points, original_article_title, sources_used)
    print("\n--- Generated Blog Article (Markdown) ---")
    print(blog_article_markdown)

    # Save to file
    timestamp = int(time.time())
    file_name = f"enriched_blog_article_{timestamp}.md"
    with open(file_name, "w", encoding="utf-8") as f:
        f.write(blog_article_markdown)
        if sources_used:
            f.write(f"\n\n## Research Sources\n")
            for source in sources_used:
                f.write(f"- {source}\n")
    
    print(f"\n✅ Blog article saved to {file_name}")
    print(f"📊 Used {len(sources_used)} real-time sources for enrichment")

    print("\n--- 🚀 Next Steps for TOP Ranking ---")
    print("1. **Review & Edit:** Verify all facts and add your unique insights")
    print("2. **Keyword Optimization:** Use SEO tools to refine keyword targeting")
    print("3. **Add Visuals:** Create compelling images, charts, or infographics")
    print("4. **Internal Linking:** Add relevant internal links to your site")
    print("5. **Promotion:** Share on social media and build quality backlinks")
    print("6. **Monitor Performance:** Track rankings and user engagement")

if __name__ == "__main__":
    main()

# This enhanced script uses real-time search to enrich news articles with the latest information,
# statistics, expert opinions, and recent developments before generating SEO-optimized blog content.