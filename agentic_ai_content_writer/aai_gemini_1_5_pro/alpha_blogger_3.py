import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities import GoogleSearchAPIWrapper
from newspaper import Article # For scraping initial article
from bs4 import BeautifulSoup
import requests
from urllib.parse import urlparse
from keybert import KeyBERT # For keyword extraction

# Load environment variables from .env file
load_dotenv(dotenv_path="../../creds.env")

# --- Configuration ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")
GOOGLE_CSE_API_KEY = os.getenv("GOOGLE_CSE_API_KEY")


if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found. Please set it in your .env file.")

if not SEARCH_ENGINE_ID:
    raise ValueError("SEARCH_ENGINE_ID not found. Please set it in your .env file.")

if not GOOGLE_CSE_API_KEY:
    raise ValueError("GOOGLE_CSE_API_KEY not found. Please set it in your .env file.")

# --- Helper Functions (Moved/Modified from previous version) ---

class CustomGoogleSearchTool:
    """
    A wrapper for Google Custom Search to perform both web and news searches.
    """
    def __init__(self, api_key: str, search_engine_id: str):
        self.search_wrapper = GoogleSearchAPIWrapper(
            google_api_key=api_key,
            google_cse_id=search_engine_id
        )

    def search_web(self, query: str, num_results: int = 10) -> list:
        """Performs a standard Google web search."""
        return self.search_wrapper.results(query, num_results)

    def search_news(self, query: str, num_results: int = 10) -> list:
        """Performs a Google News search."""
        # Google CSE supports site: search, so we can simulate news search
        return self.search_wrapper.results(f"{query} site:news.google.com", num_results)


class SEOBloggerAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-latest", # Use latest stable or 1.5 pro for better long-context handling
            google_api_key=GOOGLE_API_KEY,
            temperature=0.7
        )
        self.Google_Search_tool = CustomGoogleSearchTool(GOOGLE_CSE_API_KEY, SEARCH_ENGINE_ID)
        self.kb = KeyBERT() # Initialize KeyBERT for keyword extraction

        # Define common chains
        self.article_summarizer_chain = self._setup_article_summarizer_chain()
        self.keyword_extractor_chain = self._setup_keyword_extractor_chain()
        self.blog_post_generator_chain = self._setup_blog_post_generator_chain()

    def _setup_article_summarizer_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert summarizer. Summarize the following news article in an in-depth manner, covering all key details and sub-points. Break the summary into logical paragraphs/sections."),
            ("user", "Article: {article_content}\n\nProvide an in-depth summary:")
        ])
        return prompt | self.llm | StrOutputParser()

    def _setup_keyword_extractor_chain(self):
        # Using KeyBERT directly as it's designed for this. If you prefer LLM for extraction,
        # you'd use a prompt here. KeyBERT is often better for raw extraction.
        # However, for extracting from "each summary point", an LLM might be more granular.
        # Let's keep a flexible design here.
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert in keyword and keyphrase extraction. For the following text, identify and list the most important keywords and keyphrases, separated by commas. Aim for 5-10 keywords/phrases per section."),
            ("user", "Text section: {text_section}\n\nExtract keywords and keyphrases:")
        ])
        return prompt | self.llm | StrOutputParser() # LLM for keyword extraction from sections

    def _setup_blog_post_generator_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a highly skilled SEO content writer. Your task is to create a comprehensive, engaging, and SEO-friendly blog post based on the provided core article summary and enriched research data.

            Your blog post must:
            - Have a compelling, SEO-optimized title.
            - Include a clear introduction.
            - Structure the content with clear headings (H2, H3).
            - Integrate keywords naturally throughout the text.
            - Provide in-depth information, referencing the research.
            - Maintain a professional yet engaging tone.
            - Conclude with a strong summary and a call to action.
            - Be at least 1500 words (aim for depth).

            Here is the original article summary:
            {original_summary}

            Here is the enriched research data from various sources (Google Search, Google News, etc.). Integrate this information seamlessly and intelligently to provide additional depth and diverse perspectives:
            {enriched_data}

            Focus on creating a well-structured and informative blog post."""),
            ("user", "Generate a detailed and SEO-friendly blog post:")
        ])
        return prompt | self.llm | StrOutputParser()

    def _scrape_article_content(self, url: str) -> str:
        """
        Scrapes the main text content from a news article URL using newspaper3k.
        Falls back to requests/BeautifulSoup if newspaper3k fails.
        """
        try:
            article = Article(url)
            article.download()
            article.parse()
            if article.text:
                return article.text
            else:
                raise ValueError("Newspaper3k failed to extract text.")
        except Exception as e:
            print(f"Newspaper3k failed for {url}: {e}. Falling back to BeautifulSoup.")
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status() # Raise an exception for HTTP errors
                soup = BeautifulSoup(response.content, 'html.parser')
                # Try to find common article content containers
                paragraphs = soup.find_all(['p', 'div'], class_=['article-content', 'story-body', 'body-text'])
                if not paragraphs:
                    paragraphs = soup.find_all('p') # General paragraph search
                content = "\n".join([p.get_text() for p in paragraphs if p.get_text().strip()])
                if content:
                    return content
                else:
                    raise ValueError("BeautifulSoup failed to extract meaningful text.")
            except Exception as bs_e:
                print(f"BeautifulSoup also failed for {url}: {bs_e}")
                return "Failed to retrieve article content."

    def _extract_keywords_from_text_section(self, text_section: str) -> list[str]:
        """
        Extracts keywords/keyphrases from a text section using KeyBERT and then
        optionally an LLM for refinement.
        """
        # KeyBERT for initial extraction
        kb_keywords = self.kb.extract_keywords(text_section, keyphrase_ngram_range=(1, 3), stop_words='english', top_n=10)
        keywords = [kw[0] for kw in kb_keywords]

        # Use LLM for further refinement/expansion based on the original summary point
        # This can help get more contextually relevant phrases
        llm_keywords_response = self.keyword_extractor_chain.invoke({"text_section": text_section})
        llm_keywords = [k.strip() for k in llm_keywords_response.split(',')]

        # Combine and deduplicate
        combined_keywords = list(set(keywords + llm_keywords))
        return combined_keywords[:10] # Limit to top 10 relevant terms

    def _perform_enrichment_research(self, keyword_or_phrase: str, num_results: int = 10) -> list[dict]:
        """
        Performs Google Search and Google News search for a given keyword/phrase
        and collects top N results.
        """
        print(f"  - Researching '{keyword_or_phrase}'...")
        web_results = self.Google_Search_tool.search_web(keyword_or_phrase, num_results)
        news_results = self.Google_Search_tool.search_news(keyword_or_phrase, num_results)

        all_results = []
        for res in web_results:
            all_results.append({"source": "Google Search", "title": res.get('title', 'N/A'), "link": res.get('link', 'N/A'), "snippet": res.get('snippet', 'N/A')})
        for res in news_results:
            all_results.append({"source": "Google News", "title": res.get('title', 'N/A'), "link": res.get('link', 'N/A'), "snippet": res.get('snippet', 'N/A')})

        return all_results

    def _read_and_understand_content(self, urls: list[str]) -> str:
        """
        Attempts to scrape and return the content from a list of URLs.
        Prioritizes relevant content.
        """
        print(f"  - Reading content from {len(urls)} external sources...")
        full_content_pieces = []
        for i, url in enumerate(urls):
            if "linkedin.com" in url or "youtube.com" in url or urlparse(url).netloc == '':
                # Skip social media, video, or invalid URLs often returned by search
                continue
            print(f"    - Scraping: {url} ({i+1}/{len(urls)})")
            try:
                # Use newspaper3k first as it's optimized for articles
                article = Article(url)
                article.download()
                article.parse()
                if article.text:
                    full_content_pieces.append(f"--- Source: {url} ---\nTitle: {article.title}\nContent:\n{article.text}\n")
                else:
                    # Fallback to requests/BeautifulSoup if newspaper3k fails to get text
                    response = requests.get(url, timeout=5)
                    response.raise_for_status()
                    soup = BeautifulSoup(response.content, 'html.parser')
                    main_content_div = soup.find('article') or soup.find(class_=lambda x: x and ('content' in x.lower() or 'body' in x.lower()))
                    if main_content_div:
                        text_content = main_content_div.get_text(separator='\n', strip=True)
                    else:
                        text_content = soup.get_text(separator='\n', strip=True) # Get all text
                    
                    # Basic cleaning to reduce noise (e.g., script, style tags)
                    lines = [line.strip() for line in text_content.splitlines() if line.strip()]
                    cleaned_text = '\n'.join(lines)
                    
                    # Heuristic to limit very long pages or pages with too much non-content
                    if len(cleaned_text) > 5000: # Arbitrary limit
                        cleaned_text = cleaned_text[:5000] + "...\n[Content truncated for brevity]"

                    if len(cleaned_text) > 200: # Ensure some meaningful content
                         full_content_pieces.append(f"--- Source: {url} ---\nTitle: {soup.title.string if soup.title else 'No Title'}\nContent:\n{cleaned_text}\n")
                    else:
                        print(f"      [Skipping: Too little content extracted from {url}]")

            except Exception as e:
                print(f"      [Error scraping {url}: {e}]")
                # Add snippet if available, otherwise just URL
                full_content_pieces.append(f"--- Source: {url} ---\nContent: [Scraping failed or limited content] - {e}\n")

            # Basic rate limiting for polite scraping
            import time
            time.sleep(1) # Be polite to websites, wait 1 second between requests

        return "\n\n".join(full_content_pieces)

    def create_seo_blog_post(self, news_article_url: str) -> str:
        """
        Orchestrates the entire process to create an SEO-friendly blog post
        from a news article URL.
        """
        print(f"\n--- Starting Blog Post Generation for URL: {news_article_url} ---")

        # 1. Scrape the initial news article
        print("\nStep 1: Scraping main news article...")
        original_article_content = self._scrape_article_content(news_article_url)
        if not original_article_content or original_article_content == "Failed to retrieve article content.":
            print("Error: Could not scrape content from the main news article URL.")
            return "Failed to create blog post: Could not retrieve main article content."

        # 2. Summarize the article in-depth
        print("\nStep 2: Summarizing the article in-depth...")
        original_summary = self.article_summarizer_chain.invoke({"article_content": original_article_content})
        print("\n--- Original Article Summary ---\n" + original_summary[:500] + "...\n------------------------------") # Print a snippet

        # 3. Extract keywords/keyphrases from each summary point
        print("\nStep 3: Extracting keywords/keyphrases from summary points...")
        summary_paragraphs = [p.strip() for p in original_summary.split('\n') if p.strip()]
        all_extracted_keywords = []
        for i, paragraph in enumerate(summary_paragraphs):
            print(f"  - Extracting keywords from summary paragraph {i+1}...")
            keywords_for_paragraph = self._extract_keywords_from_text_section(paragraph)
            all_extracted_keywords.extend(keywords_for_paragraph)
        all_extracted_keywords = list(set(all_extracted_keywords)) # Deduplicate
        print(f"Extracted Keywords/Keyphrases: {', '.join(all_extracted_keywords)}")

        # 4. & 5. Perform enrichment research and collect top results
        print("\nStep 4 & 5: Performing enrichment research for each keyword/keyphrase...")
        enriched_research_data = []
        for keyword in all_extracted_keywords:
            results = self._perform_enrichment_research(keyword, num_results=10) # Get 10 results from each (web/news)
            enriched_research_data.extend(results)

        # Deduplicate research results by link
        unique_enriched_links = {}
        for item in enriched_research_data:
            if item['link'] not in unique_enriched_links:
                unique_enriched_links[item['link']] = item
        final_enriched_results = list(unique_enriched_links.values())
        print(f"Collected {len(final_enriched_results)} unique external research links.")

        # 6. Read and understand content from all 20 results per keyword
        # This will be a lot of content, potentially too much for a single LLM call.
        # We need to be careful with context window limits of Gemini 1.5 Pro.
        # It has a large context window (1M tokens) but too much noise is bad.
        print("\nStep 6: Reading and understanding content from enriched sources...")
        all_external_urls = [res['link'] for res in final_enriched_results if res.get('link') and res.get('link') != 'N/A']
        
        # Limit the number of external articles to scrape to prevent overwhelming the system/API
        # and to manage processing time. Let's pick up to 20-30 most relevant looking links from snippets.
        # A more advanced approach would use RAG here.
        urls_to_scrape = all_external_urls[:30] # Adjust this limit as needed

        enriched_content_text = self._read_and_understand_content(urls_to_scrape)
        
        # If the enriched_content_text is still too long, summarize it.
        # For simplicity, we'll assume Gemini 1.5 Pro can handle the combined context,
        # but in production, you might need a summary step here for `enriched_content_text`.
        if len(enriched_content_text) > 800000: # Approximate a very large text
             print("Warning: Enriched content is very large, attempting to summarize it before final blog generation.")
             # You'd define another chain here to summarize the 'enriched_content_text'
             # For now, let's just truncate for demonstration if it's excessively long
             enriched_content_text = enriched_content_text[:800000] + "\n\n[...Additional research content truncated due to length...]"


        # 7. Generate SEO-friendly blog post
        print("\nStep 7: Generating SEO-friendly blog post...")
        blog_post = self.blog_post_generator_chain.invoke(
            {
                "original_summary": original_summary,
                "enriched_data": enriched_content_text
            }
        )

        print("\n--- Blog Post Generated Successfully! ---")
        return blog_post

# --- Main execution ---
def main():
    agent = SEOBloggerAgent()

    # Example Usage: Replace with a real news article URL
    news_url = input("Enter the URL of the news article to base the blog post on: ")
    if not news_url.startswith("http"):
        print("Invalid URL. Please enter a full URL starting with http:// or https://")
        return

    blog_content = agent.create_seo_blog_post(news_url)
    print("\n\n--- FINAL BLOG POST ---\n")
    print(blog_content)

    # Optional: Save to file
    with open("seo_blog_post.md", "w", encoding="utf-8") as f:
        f.write(blog_content)
    print("\nBlog post saved to seo_blog_post.md")

if __name__ == "__main__":
    main()