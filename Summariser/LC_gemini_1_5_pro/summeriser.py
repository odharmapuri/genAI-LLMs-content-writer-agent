import os
# Set the USER_AGENT environment variable
os.environ['USER_AGENT'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'

from langchain.chat_models import init_chat_model
# from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import re

# Load environment variables from .env file
load_dotenv(dotenv_path="../../creds.env")

# --- Configuration ---
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found. Please set it in your .env file.")

# models initialisation
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=API_KEY)
model = init_chat_model(model = "gemini-2.0-flash-001", model_provider="google_genai", google_api_key=API_KEY, temperature=0.1, max_output_tokens=1024)
vectorstore = InMemoryVectorStore(embeddings)

# web scrapping
loader = WebBaseLoader(
    web_path=[
        "https://www.yahoo.com/news/first-time-webb-telescope-discovers-151013150.html"
    ]
)
docs = loader.load()


# content preprocessing
textsplitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
content = textsplitter.split_documents(docs)

# store the content in vector store
documentids = vectorstore.add_documents(documents=content)

# # creating prompt
# prompt = """
# You are an assistant for generating seo friendly content for a blog post on crypto, stock, financial markets, trading, and investing niche. Use the following pieces of retrieved context and add fact based additional value added data from the internet based on latest news updates to generate blog article.
# You must use the retrieved context to generate the blog article. The blog article should be concise, informative, and engaging. It should be written in a way that is easy to understand for a general audience. The blog article should also be optimized for search engines, with relevant keywords and phrases included throughout the text.
# """
# Be precise and straightforward: Avoid unnecessary detail or filler text.
# Avoid opinion or bias: Present only the factual content of the source.
prompt = """
You are an AI model tasked with summarizing news articles or blog posts into useful data. Your summary must meet the following criteria:

break down point wise

Highlight useful data points: Focus on facts, figures, names, dates, actions taken, outcomes, and implications.

Include context: When needed, add a brief explanation what, why and how to make each point understandable on its own.
"""

query = "Now, summarize the following article/blog post accordingly:"

# retrieval
context = vectorstore.similarity_search(query, k=5)

# Combine prompt, context, and question into a single string
message = f"{prompt}\nQuestion: {query}\nContext: {context}"

# # augmenation
# message = prompt.invoke({"question": query, "context": context})

# generation
response = model.invoke(message)

def remove_asterisk_content(line):
    # Remove content from first * to last * (inclusive)
    return re.sub(r"\*.*\*", "", line).strip()

# Clean and print each line, and write to a txt file
output_lines = []
for line in response.content.splitlines():
    cleaned = remove_asterisk_content(line)
    if cleaned:
        print(cleaned)
        output_lines.append(cleaned)

with open("article_summary.txt", "w", encoding="utf-8") as f:
    for line in output_lines:
        f.write(line + "\n")
