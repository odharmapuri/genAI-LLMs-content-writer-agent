# This script uses Google Gemini 1.5 Pro to enrich news articles with real-time search data, generating SEO-optimized blog content.
# It fetches the article content, summarizes it, enriches it with real-time search results  

import google.generativeai as genai
import os
import requests
from bs4 import BeautifulSoup
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
import time
from dotenv import load_dotenv
import re

# Load environment variables from .env file
load_dotenv(dotenv_path="../../creds.env")

# --- Configuration ---
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found. Please set it in your .env file.")

genai.configure(api_key=API_KEY)

# Choose your Gemini model
# gemini-1.5-flash is more cost-effective and faster for many tasks.
# gemini-1.5-pro offers higher reasoning for very complex articles.
GEMINI_MODEL = 'gemini-1.5-flash' # or 'gemini-1.5-pro'

# Initialize the Gemini model
# Note: Google Search is not available as a built-in tool in the current API version
# The model will work with the provided content and generate enriched responses based on training data
model = genai.GenerativeModel(model_name=GEMINI_MODEL)

# --- Helper Functions ---

@retry(wait=wait_exponential(multiplier=1, min=4, max=60), stop=stop_after_attempt(5),
       retry=retry_if_exception_type(Exception)) # Catches general exceptions, including rate limits
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
        raise # Re-raise to trigger tenacity retry


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
        response.raise_for_status()  # Raise an exception for HTTP errors
        soup = BeautifulSoup(response.text, 'html.parser')

        # Attempt to find common article content containers
        article_text = []
        for tag in ['p', 'h1', 'h2', 'h3', 'li']: # Include headings and list items
            for element in soup.find_all(tag):
                if element.get_text(strip=True):
                    article_text.append(element.get_text(strip=True))

        full_text = "\n".join(article_text)

        # Basic filtering for boilerplate text (e.g., footers, navs)
        # This is a simple heuristic and might need refinement for specific sites.
        min_len = 100 # Minimum length for a section to be considered meaningful
        filtered_sections = [s for s in full_text.split('\n\n') if len(s.split()) > 10] # filter short lines

        # Try to find a more specific article body if available
        article_body = soup.find('article') or soup.find(class_=re.compile(r'article|content|body|main'))
        if article_body:
            paragraphs = article_body.find_all(['p', 'h1', 'h2', 'h3', 'li'])
            clean_text = "\n".join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
            if len(clean_text) > len(full_text) * 0.5: # If specific body is significantly larger
                full_text = clean_text

        if not full_text or len(full_text) < 200: # Ensure we got enough content
            print(f"Warning: Could not extract sufficient content from {url}. Raw text length: {len(full_text)}")
            return None

        # Return a truncated version if it's extremely long, to fit into context window
        # Max context window for Gemini 1.5 Flash is 1M tokens, 1.5 Pro is 1M (2M in preview)
        # This is a safe guard. Roughly 4 chars = 1 token.
        return full_text[:900000] # Adjust based on model and expected article length

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

def enrich_summary_points(summary_points_str):
    """
    Enriches the summary points by adding context, analysis, and relevant details.
    Note: Real-time search is not available in the current API version,
    but the model can still provide valuable enrichment based on its training data.
    """
    prompt = f"""
    You are an expert journalist and researcher. I will provide you with a list of summary points from a news article.
    Your task is to enrich each point by adding the following details based on your knowledge:
    - **What:** More specific details about the event or topic.
    - **Why:** The significance, causes, or underlying reasons.
    - **When:** Historical context, typical timelines, or patterns.
    - **Where:** Geographic context, regional implications.
    - **How:** The process, methods, or mechanisms involved.
    - **Context/Analysis:** Relevant background information, implications, and expert perspectives.

    Provide detailed, coherent, and expanded analysis for each point.
    Present the enriched points as a well-structured, informative list.

    Current Date: {time.strftime('%Y-%m-%d')}

    Summary Points to Enrich:
    {summary_points_str}

    Enriched Summary Points:
    """
    print("\nEnriching summary points with detailed analysis...")
    response = generate_content_with_retry([prompt])

    # Since real-time search isn't available, we won't have citations
    citations_info = []

    return response.text.strip(), citations_info

def generate_blog_article(enriched_summary_points_str, article_title_suggestion=""):
    """
    Generates a comprehensive, SEO-optimized blog article from enriched summary points.
    Includes headline, subheadings, bullet points, short paragraphs, and link suggestions.
    """
    prompt = f"""
    You are a professional SEO content writer for a top-ranking blog.
    Your goal is to write a comprehensive, engaging, and highly SEO-optimized blog article
    based on the following enriched summary points.

    **Instructions for SEO and Readability:**
    1.  **Headline:** Create a compelling, keyword-rich headline (H1) that attracts clicks and is optimized for search engines.
    2.  **Introduction:** A strong hook that clearly states the article's purpose and what the reader will gain.
    3.  **Subheadings (H2, H3):** Use descriptive and keyword-rich subheadings to break down the content, improving readability and SEO.
    4.  **Content Structure:**
        * Expand each enriched point into concise, informative paragraphs.
        * Use bullet points within sections where appropriate for lists or key takeaways.
        * Keep paragraphs relatively short (3-5 sentences).
        * Maintain a natural, conversational tone.
    5.  **Data/Stats/Quotes:** Integrate the provided data, statistics, and quotes naturally within the text, crediting sources clearly.
    6.  **Internal/External Link Suggestions:**
        * **External:** Suggest specific anchor text and URLs for the sources you used during enrichment (if provided in citations) and other highly authoritative external resources that would add value.
        * **Internal:** Suggest generic internal link opportunities (e.g., "Learn more about [related topic] in our dedicated guide").
    7.  **Call to Action (Optional):** A soft call to action at the end (e.g., "What are your thoughts? Share in the comments!").
    8.  **Featured Image Idea:** Suggest a concept for a compelling featured image.

    **Enriched Summary Points:**
    {enriched_summary_points_str}

    **Suggested Original Article Title (for context, use for inspiration, but create new SEO-friendly one):**
    "{article_title_suggestion}"

    **Output Format (use Markdown for headings, bold, italics, lists):**

    # [Compelling, Keyword-Rich Headline]

    [Catchy Introduction - ~2-3 paragraphs]

    ## [Descriptive Subheading 1]
    [Paragraph(s) expanding on enriched point 1, incorporating data/stats/quotes]
    - [Bullet point example]
    - [Another bullet point]

    ## [Descriptive Subheading 2]
    [Paragraph(s) expanding on enriched point 2, incorporating data/stats/quotes]

    ### [More Specific Subheading for Detail]
    [Further detail]

    ... (continue for all enriched points) ...

    ## Conclusion
    [Summary of key takeaways, future outlook, soft CTA]

    ---

    **Featured Image Idea:** [Brief description for a visual]

    **External Link Opportunities:**
    - [Anchor text]: [URL]
    - [Anchor text]: [URL]

    **Internal Link Opportunities:**
    - [Anchor text for a relevant internal article]
    - [Another anchor text for a relevant internal article]
    """
    print("\nGenerating comprehensive blog article...")
    generation_config = {
        "temperature": 0.7,  # Adjust for creativity vs. factual accuracy (0.0 for factual, 1.0 for creative)
        "max_output_tokens": 4096 # Increase if your articles are very long
    }
    response = generate_content_with_retry([prompt], generation_config=generation_config)
    return response.text.strip()

# --- Main Execution Flow ---
def main():
    article_url = input("Please enter the URL of the news article: ").strip()
    if not article_url:
        print("No URL provided. Exiting.")
        return

    original_article_title = input("Enter the original article title (optional, for context): ").strip()

    print(f"Fetching content from: {article_url}")
    article_content = fetch_article_content(article_url)

    if not article_content:
        print("Failed to retrieve article content. Cannot proceed.")
        return

    # Step 1: Summarize
    print("\n--- Step 1: Summarizing Article ---")
    summary_points = summarize_article(article_content)
    print("\nGenerated Summary Points:")
    print(summary_points)

    # Step 2: Enrich Summary Points
    print("\n--- Step 2: Enriching Summary Points with Real-time Data ---")
    enriched_points, citations = enrich_summary_points(summary_points)
    print("\nEnriched Summary Points:")
    print(enriched_points)

    if citations:
        print("\nCitations Used:")
        for citation in citations:
            print(citation)

    # Step 3: Generate Blog Article
    print("\n--- Step 3: Generating SEO-Optimized Blog Article ---")
    blog_article_markdown = generate_blog_article(enriched_points, original_article_title)
    print("\n--- Generated Blog Article (Markdown) ---")
    print(blog_article_markdown)

    # Optional: Save to a file
    file_name = f"blog_article_{int(time.time())}.md"
    with open(file_name, "w", encoding="utf-8") as f:
        f.write(blog_article_markdown)
    print(f"\nBlog article saved to {file_name}")

    # Process and suggest final links based on citations
    # Clean up citations for better presentation in the blog post
    final_external_links = []
    if citations:
        print("\n--- Suggested Final External Links from Citations ---")
        for citation_uri in citations:
            # Basic cleanup: remove (Title: N/A) and just get the URI
            clean_uri = citation_uri.split(" (Title:")[0].replace("- ", "").strip()
            # Try to get a more readable anchor text from the URI or by reverse-lookup (if possible, though complex)
            # For simplicity, we'll just use the domain or a generic text
            domain = clean_uri.replace("https://", "").replace("http://", "").split("/")[0]
            final_external_links.append(f"- [{domain}]({clean_uri})")
        print("\n".join(final_external_links))

    print("\n--- Remember for TOP Ranking ---")
    print("1. **Human Review & Refinement:** Always review and edit the generated content for factual accuracy, tone, and unique human insights.")
    print("2. **Manual Research:** Since real-time search isn't available, supplement with manual research for current data and statistics.")
    print("3. **Dedicated Keyword Research:** Use keyword research tools to optimize your content.")
    print("4. **Technical SEO:** Ensure your website's technical SEO (speed, mobile, schema) is flawless.")
    print("5. **Promote & Build Backlinks:** High-quality content needs promotion and backlinks to rank.")
    print("6. **Monitor & Iterate:** Use Google Search Console to track performance and adapt your strategy.")


if __name__ == "__main__":
    main()
# This script uses Google Gemini to summarize and enrich news articles, generating SEO-optimized blog content.
# It fetches article content, summarizes it, enriches the summary with real-time data, and generates a comprehensive blog article.
# The final output is saved as a Markdown file, ready for publication.