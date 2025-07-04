# this is a custom query script that uses langchain to scrape a website, 
# preprocess the content, store it in a vector store, and then 
# use a language model to generate a response based on a query.

from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

OPENAI_API_KEY="your-openai-api-key"  # Replace with your actual OpenAI API key


# models initialisation
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
model = init_chat_model(model = "gpt-4.1-nano", model_provider="openai")
vectorstore = InMemoryVectorStore(embeddings)


# web scrapping
loader = WebBaseLoader(web_path="url-for-context")  # Replace with the actual URL you want to scrape
docs = loader.load()


# content preprocessing
textsplitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
content = textsplitter.split_documents(docs)

# store the content in vector store
documentids = vectorstore.add_documents(documents=content)

# creating prompt
prompt = """
You are an assistant for generating seo friendly content for a blog post on crypto, stock, financial markets, trading, and investing niche. Use the following pieces of retrieved context and add fact based additional value added data from the internet based on latest news updates to generate blog article.
You must use the retrieved context to generate the blog article. The blog article should be concise, informative, and engaging. It should be written in a way that is easy to understand for a general audience. The blog article should also be optimized for search engines, with relevant keywords and phrases included throughout the text.
"""

query = "your-custom-query-here"  # Replace with your actual query

# retrieval
context = vectorstore.similarity_search(query, k=5)

# augmenation
message = prompt.invoke({"question": query, "context": context})

# generation
response = model.invoke(message)
print(response.content)