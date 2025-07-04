import os
import json
import re
import time
import requests
import feedparser
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from urllib.parse import quote_plus
from dotenv import load_dotenv

# LangChain imports
from langchain.agents import initialize_agent, Tool, AgentType
# from langchain.memory import ConversationBufferMemory # You might not need this direct import for simple chat history
from langchain.schema import BaseOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain # Deprecated
from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain_core.runnables import RunnablePassthrough # For LCEL

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

# Data structures
@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    headings: List[str] = None

@dataclass
class NewsResult:
    title: str
    url: str
    published: str
    summary: str

@dataclass
class BlogContent:
    title: str
    meta_title: str
    meta_description: str
    og_tags: Dict[str, str]
    outline: List[Dict[str, str]]
    content: str
    keywords: List[str]
    lsi_terms: List[str]
    schema_markup: str
    cover_image_prompt: str

class GoogleSearchTool(BaseTool):
    name: str = "Google Search"
    description: str = "Search Google using Custom Search Engine API"
    api_key: str
    search_engine_id: str

    def __init__(self, api_key: str, search_engine_id: str, **kwargs: Any):
        super().__init__(api_key=api_key, search_engine_id=search_engine_id, **kwargs)

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> List[SearchResult]:
        """Search Google and return structured results"""
        try:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                'key': self.api_key,
                'cx': self.search_engine_id,
                'q': query,
                'num': 10
            }

            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            results = []
            for item in data.get('items', []):
                result = SearchResult(
                    title=item.get('title', ''),
                    url=item.get('link', ''),
                    snippet=item.get('snippet', ''),
                    headings=self._extract_headings(item.get('snippet', ''))
                )
                results.append(result)

            return results
        except Exception as e:
            print(f"Error in Google search: {e}")
            return []

    def _extract_headings(self, text: str) -> List[str]:
        """Extract potential headings from text"""
        # Simple heading extraction - can be enhanced
        sentences = text.split('.')
        headings = [s.strip() for s in sentences if len(s.strip()) > 10 and len(s.strip()) < 100]
        return headings[:5]

class GoogleNewsTool(BaseTool):
    name: str = "google_news"
    description: str = "Search Google News using RSS feeds"

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> List[NewsResult]:
        """Search Google News and return structured results"""
        try:
            encoded_query = quote_plus(query)
            rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"

            feed = feedparser.parse(rss_url)
            results = []

            for entry in feed.entries[:10]:
                result = NewsResult(
                    title=entry.get('title', ''),
                    url=entry.get('link', ''),
                    published=entry.get('published', ''),
                    summary=entry.get('summary', '')
                )
                results.append(result)

            return results
        except Exception as e:
            print(f"Error in Google News search: {e}")
            return []

class SEOContentAgent:
    def __init__(self, google_api_key: str, google_cse_api_key: str, search_engine_id: str):
        # Initialize Gemini LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=google_api_key,
            temperature=0.7
        )

        # Initialize tools
        self.Google_Search = GoogleSearchTool(google_cse_api_key, search_engine_id)
        self.google_news = GoogleNewsTool()

        # Initialize memory (kept for now, but be mindful of future migrations)
        # self.memory = ConversationBufferMemory(memory_key="chat_history") # Deprecated warning still applies

        # Setup prompts
        self._setup_prompts()

        # Setup chains
        self._setup_chains()

    def _setup_prompts(self):
        """Setup all prompts for different stages"""

        # Outline generation prompt
        self.outline_prompt = PromptTemplate(
            input_variables=["topic", "search_results", "news_results", "target_keywords"],
            template="""
            Create a comprehensive blog post outline for the topic: "{topic}"

            Based on the following research data:

            TOP SEARCH RESULTS:
            {search_results}

            RECENT NEWS:
            {news_results}

            TARGET KEYWORDS: {target_keywords}

            Create a detailed outline with:
            1. Compelling headline (H1)
            2. Introduction hook
            3. 5-8 main sections (H2) with subsections (H3)
            4. Conclusion with call-to-action
            5. FAQ section

            Focus on:
            - Search intent satisfaction
            - E-A-T (Expertise, Authoritativeness, Trustworthiness)
            - User engagement
            - SEO optimization

            Return as structured JSON with sections and subsections.
            """
        )

        # Content expansion prompt
        self.content_prompt = PromptTemplate(
            input_variables=["section_title", "section_outline", "context", "keywords", "lsi_terms"],
            template="""
            Write a comprehensive section for: "{section_title}"

            Section Outline: {section_outline}
            Context: {context}
            Primary Keywords: {keywords}
            LSI Terms: {lsi_terms}

            Requirements:
            - 300-500 words per section
            - Include statistics, facts, and data points
            - Answer Who, What, Where, When, Why, How
            - Use active voice and clear language
            - Include relevant examples
            - Naturally integrate keywords (2-3% density)
            - Add internal linking opportunities [LINK: anchor text]
            - Use bullet points and lists where appropriate
            - Maintain authoritative tone

            Format in clean HTML with proper heading tags.
            """
        )

        # Meta data generation prompt
        self.meta_prompt = PromptTemplate(
            input_variables=["title", "content_summary", "keywords"],
            template="""
            Generate SEO-optimized meta data for:

            Title: {title}
            Content Summary: {content_summary}
            Keywords: {keywords}

            Generate:
            1. Meta Title (50-60 characters, include primary keyword)
            2. Meta Description (150-160 characters, compelling with CTA)
            3. Open Graph Title
            4. Open Graph Description
            5. Focus Keyword
            6. 5 LSI Keywords
            7. Schema Markup (Article type)
            8. Cover Image Prompt (detailed description for AI image generation)

            Return as structured JSON.
            """
        )

    def _setup_chains(self):
        """Setup LangChain chains using LCEL"""
        # For simple Prompt | LLM chains, direct piping is used.
        # .invoke() is used to run the chain, and .content to get the string output.
        self.outline_chain = self.outline_prompt | self.llm
        self.content_chain = self.content_prompt | self.llm
        self.meta_chain = self.meta_prompt | self.llm

    def research_topic(self, topic: str) -> Dict[str, Any]:
        """Research topic using Google Search and News"""
        print(f"🔍 Researching topic: {topic}")

        # Search Google
        search_results = self.Google_Search._run(topic)
        print(f"Found {len(search_results)} search results")

        # Search Google News
        news_results = self.google_news._run(topic)
        print(f"Found {len(news_results)} news results")

        return {
            'search_results': search_results,
            'news_results': news_results
        }

    def extract_keywords(self, topic: str, research_data: Dict) -> List[str]:
        """Extract and generate relevant keywords"""
        # Combine all text from research
        all_text = topic + " "

        for result in research_data['search_results']:
            all_text += f"{result.title} {result.snippet} "

        for news in research_data['news_results']:
            all_text += f"{news.title} {news.summary} "

        # Use LLM to extract keywords
        keyword_prompt = f"""
        Extract 10-15 relevant SEO keywords from this text about "{topic}":

        {all_text[:2000]}

        Include:
        - Primary keywords (high volume, competitive)
        - Long-tail keywords (specific, lower competition)
        - LSI keywords (semantically related)

        Return as comma-separated list.
        """

        try:
            response = self.llm.invoke(keyword_prompt) # Use .invoke() for direct LLM calls
            keywords = [k.strip() for k in response.content.split(',')]
            return keywords[:15]
        except Exception as e:
            print(f"Error extracting keywords: {e}")
            return [topic, f"{topic} guide", f"how to {topic}", f"{topic} tips"]

    def generate_outline(self, topic: str, research_data: Dict, keywords: List[str]) -> Dict:
        """Generate comprehensive blog outline"""
        print("📝 Generating blog outline...")

        # Format research data for prompt
        search_summary = "\n".join([
            f"- {r.title}: {r.snippet}" for r in research_data['search_results'][:5]
        ])

        news_summary = "\n".join([
            f"- {n.title} ({n.published}): {n.summary}" for n in research_data['news_results'][:5]
        ])

        # Generate outline
        try:
            # Use .invoke() with LCEL chains
            response = self.outline_chain.invoke(
                {
                    "topic": topic,
                    "search_results": search_summary,
                    "news_results": news_summary,
                    "target_keywords": ", ".join(keywords[:5])
                }
            )
            # Access content via .content for the string output
            outline = json.loads(response.content)
            return outline
        except Exception as e:
            print(f"Error generating outline: {e}")
            return self._fallback_outline(topic)

    def _fallback_outline(self, topic: str) -> Dict:
        """Fallback outline structure"""
        return {
            "title": f"Complete Guide to {topic}",
            "introduction": f"Introduction to {topic}",
            "sections": [
                {"title": f"What is {topic}?", "subsections": ["Definition", "Key Features"]},
                {"title": f"Benefits of {topic}", "subsections": ["Advantages", "Use Cases"]},
                {"title": f"How to Get Started with {topic}", "subsections": ["Step-by-Step Guide", "Best Practices"]},
                {"title": f"Common Challenges and Solutions", "subsections": ["Obstacles", "Solutions"]},
                {"title": f"Future of {topic}", "subsections": ["Trends", "Predictions"]}
            ],
            "conclusion": "Conclusion and next steps",
            "faq": ["What is " + topic + "?", "How does " + topic + " work?"]
        }

    def generate_content(self, topic: str, outline: Dict, keywords: List[str], research_data: Dict) -> str:
        """Generate full blog content"""
        print("✍️ Generating blog content...")

        content_parts = []

        # Generate introduction
        intro_context = f"Research data: {research_data['search_results'][:3]}"
        intro_response = self.content_chain.invoke( # Use .invoke()
            {
                "section_title": "Introduction",
                "section_outline": outline.get('introduction', ''),
                "context": intro_context,
                "keywords": ", ".join(keywords[:3]),
                "lsi_terms": ", ".join(keywords[3:8])
            }
        )
        intro_content = intro_response.content # Access content via .content
        content_parts.append(f"<h1>{outline.get('title', topic)}</h1>")
        content_parts.append(intro_content)

        # Generate main sections
        for section in outline.get('sections', []):
            section_title = section.get('title', '')
            subsections = section.get('subsections', [])

            section_context = f"Subsections: {', '.join(subsections)}"
            section_response = self.content_chain.invoke( # Use .invoke()
                {
                    "section_title": section_title,
                    "section_outline": str(subsections),
                    "context": section_context,
                    "keywords": ", ".join(keywords[:5]),
                    "lsi_terms": ", ".join(keywords[5:10])
                }
            )
            section_content = section_response.content # Access content via .content
            content_parts.append(section_content)

        # Generate conclusion
        conclusion_response = self.content_chain.invoke( # Use .invoke()
            {
                "section_title": "Conclusion",
                "section_outline": outline.get('conclusion', ''),
                "context": "Wrap up the article with key takeaways",
                "keywords": ", ".join(keywords[:3]),
                "lsi_terms": ", ".join(keywords[3:6])
            }
        )
        conclusion_content = conclusion_response.content # Access content via .content
        content_parts.append(conclusion_content)

        # Generate FAQ
        if outline.get('faq'):
            faq_content = "<h2>Frequently Asked Questions</h2>\n"
            for faq in outline.get('faq', []):
                faq_answer_response = self.content_chain.invoke( # Use .invoke()
                    {
                        "section_title": f"FAQ: {faq}",
                        "section_outline": "Provide a concise, helpful answer",
                        "context": f"Answer the question: {faq}",
                        "keywords": ", ".join(keywords[:3]),
                        "lsi_terms": ""
                    }
                )
                faq_answer = faq_answer_response.content # Access content via .content
                faq_content += f"<h3>{faq}</h3>\n{faq_answer}\n"
            content_parts.append(faq_content)

        return "\n\n".join(content_parts)

    def generate_meta_data(self, title: str, content: str, keywords: List[str]) -> Dict:
        """Generate SEO meta data"""
        print("🏷️ Generating meta data...")

        # Create content summary
        content_summary = content[:500] + "..." if len(content) > 500 else content
        content_summary = re.sub(r'<[^>]+>', '', content_summary)  # Remove HTML tags

        try:
            meta_response = self.meta_chain.invoke( # Use .invoke()
                {
                    "title": title,
                    "content_summary": content_summary,
                    "keywords": ", ".join(keywords[:10])
                }
            )
            meta_data = json.loads(meta_response.content) # Access content via .content
            return meta_data
        except Exception as e:
            print(f"Error generating meta data: {e}")
            return self._fallback_meta_data(title, keywords)

    def _fallback_meta_data(self, title: str, keywords: List[str]) -> Dict:
        """Fallback meta data"""
        return {
            "meta_title": title[:60],
            "meta_description": f"Complete guide to {title}. Learn everything you need to know about {keywords[0] if keywords else title}.",
            "og_title": title,
            "og_description": f"Comprehensive guide to {title}",
            "focus_keyword": keywords[0] if keywords else title.split()[0],
            "lsi_keywords": keywords[1:6] if len(keywords) > 1 else [title],
            "schema_markup": self._generate_schema_markup(title, keywords),
            "cover_image_prompt": f"Professional, high-quality image representing {title}, modern design, clean background"
        }

    def _generate_schema_markup(self, title: str, keywords: List[str]) -> str:
        """Generate JSON-LD schema markup"""
        schema = {
            "@context": "https://schema.org",
            "@type": "Article",
            "headline": title,
            "description": f"Comprehensive guide about {title}",
            "keywords": keywords[:5],
            "author": {
                "@type": "Organization",
                "name": "Your Website Name"
            },
            "publisher": {
                "@type": "Organization",
                "name": "Your Website Name"
            },
            "datePublished": datetime.now().isoformat(),
            "dateModified": datetime.now().isoformat()
        }
        return json.dumps(schema, indent=2)

    def create_blog_post(self, topic: str) -> BlogContent:
        """Main method to create complete blog post"""
        print(f"🚀 Starting blog post creation for: {topic}")

        # Step 1: Research
        research_data = self.research_topic(topic)

        # Step 2: Extract keywords
        keywords = self.extract_keywords(topic, research_data)
        print(f"🔑 Extracted keywords: {keywords[:5]}")

        # Step 3: Generate outline
        outline = self.generate_outline(topic, research_data, keywords)

        # Step 4: Generate content
        content = self.generate_content(topic, outline, keywords, research_data)

        # Step 5: Generate meta data
        meta_data = self.generate_meta_data(outline.get('title', topic), content, keywords)

        # Create BlogContent object
        blog_content = BlogContent(
            title=outline.get('title', topic),
            meta_title=meta_data.get('meta_title', ''),
            meta_description=meta_data.get('meta_description', ''),
            og_tags={
                'og:title': meta_data.get('og_title', ''),
                'og:description': meta_data.get('og_description', ''),
                'og:type': 'article'
            },
            outline=outline.get('sections', []),
            content=content,
            keywords=keywords,
            lsi_terms=meta_data.get('lsi_keywords', []),
            schema_markup=meta_data.get('schema_markup', ''),
            cover_image_prompt=meta_data.get('cover_image_prompt', '')
        )

        return blog_content

    def save_to_wordpress(self, blog_content: BlogContent, wp_url: str, wp_username: str, wp_password: str, publish: bool = False):
        """Save blog post to WordPress via REST API"""
        print("📤 Publishing to WordPress...")

        try:
            # WordPress REST API endpoint
            api_url = f"{wp_url}/wp-json/wp/v2/posts"

            # Authentication
            auth = (wp_username, wp_password)

            # Prepare post data
            post_data = {
                'title': blog_content.title,
                'content': blog_content.content,
                'status': 'publish' if publish else 'draft',
                'meta': {
                    '_yoast_wpseo_title': blog_content.meta_title,
                    '_yoast_wpseo_metadesc': blog_content.meta_description,
                    '_yoast_wpseo_focuskw': blog_content.keywords[0] if blog_content.keywords else '',
                }
            }

            # Send request
            response = requests.post(api_url, json=post_data, auth=auth)
            response.raise_for_status()

            post_info = response.json()
            print(f"✅ Post created successfully! ID: {post_info.get('id')}")
            print(f"📝 Post URL: {post_info.get('link')}")

            return post_info

        except Exception as e:
            print(f"❌ Error publishing to WordPress: {e}")
            return None

    def save_to_file(self, blog_content: BlogContent, filename: str = None):
        """Save blog content to local files"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"blog_post_{timestamp}"

        # Save HTML version
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{blog_content.meta_title}</title>
    <meta charset="UTF-8">
    <meta name="description" content="{blog_content.meta_description}">
    <meta property="og:title" content="{blog_content.og_tags.get('og:title', '')}">
    <meta property="og:description" content="{blog_content.og_tags.get('og:description', '')}">
    <meta property="og:type" content="{blog_content.og_tags.get('og:type', 'article')}">
    <script type="application/ld+json">
    {blog_content.schema_markup}
    </script>
</head>
<body>
    {blog_content.content}
</body>
</html>
        """

        with open(f"{filename}.html", 'w', encoding='utf-8') as f:
            f.write(html_content)

        # Save metadata as JSON
        metadata = {
            'title': blog_content.title,
            'meta_title': blog_content.meta_title,
            'meta_description': blog_content.meta_description,
            'keywords': blog_content.keywords,
            'lsi_terms': blog_content.lsi_terms,
            'cover_image_prompt': blog_content.cover_image_prompt,
            'og_tags': blog_content.og_tags,
            'schema_markup': blog_content.schema_markup
        }

        with open(f"{filename}_metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        print(f"💾 Files saved: {filename}.html and {filename}_metadata.json")

# Usage Example
def main():
    # Configuration
    # Ensure these are loaded from .env or provided securely
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GOOGLE_CSE_API_KEY = os.getenv("GOOGLE_CSE_API_KEY")
    SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")

    # WordPress configuration (optional)
    WP_URL = os.getenv("WP_URL")
    WP_USERNAME = os.getenv("WP_USERNAME")
    WP_PASSWORD = os.getenv("WP_PASSWORD")

    if not all([GOOGLE_API_KEY, GOOGLE_CSE_API_KEY, SEARCH_ENGINE_ID]):
        print("Error: Missing one or more Google API/CSE keys or Search Engine ID in .env file.")
        print("Please set GOOGLE_API_KEY, GOOGLE_CSE_API_KEY, and SEARCH_ENGINE_ID.")
        return # Exit if keys are not set

    # Initialize agent
    agent = SEOContentAgent(
        google_api_key=GOOGLE_API_KEY,
        google_cse_api_key=GOOGLE_CSE_API_KEY,
        search_engine_id=SEARCH_ENGINE_ID
    )

    # Generate blog post
    topic = input("Enter your topic/keyword: ")
    blog_content = agent.create_blog_post(topic)

    # Save to file
    agent.save_to_file(blog_content)

    # Optional: Publish to WordPress
    publish_choice = input("Publish to WordPress? (y/n): ").lower() == 'y'
    if publish_choice:
        if not all([WP_URL, WP_USERNAME, WP_PASSWORD]):
            print("Error: WordPress credentials not found in .env file. Skipping WordPress publish.")
        else:
            agent.save_to_wordpress(
                blog_content,
                WP_URL,
                WP_USERNAME,
                WP_PASSWORD,
                publish=False  # Set to True to publish immediately
            )

    print("\n📊 Blog Post Summary:")
    print(f"Title: {blog_content.title}")
    print(f"Word Count: {len(blog_content.content.split())}")
    print(f"Keywords: {', '.join(blog_content.keywords[:5])}")
    print(f"Cover Image Prompt: {blog_content.cover_image_prompt}")

if __name__ == "__main__":
    main()