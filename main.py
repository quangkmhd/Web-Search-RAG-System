import streamlit as st
import requests
from typing import Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get Hugging Face token from environment variable
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

if not HUGGINGFACE_TOKEN:
    st.error("Hugging Face API token not found. Please add it to your .env file.")
    st.stop()

if "client" not in st.session_state:
    st.session_state.client = None
if "collection_name" not in st.session_state:
    st.session_state.collection_name = None

# Hugging Face API settings
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2" 
GENERATION_MODEL = "HuggingFaceH4/zephyr-7b-beta"

# Get embeddings using Hugging Face API
def get_embeddings(texts, hf_token):
    api_url = HUGGINGFACE_API_URL + EMBEDDING_MODEL
    headers = {"Authorization": f"Bearer {hf_token}"}
    
    all_embeddings = []
    
    for text in texts:
        payload = {"inputs": text}
        response = requests.post(api_url, headers=headers, json=payload)
        
        if response.status_code == 200:
            embedding = response.json()
            all_embeddings.append({"embedding": embedding})
        else:
            st.error(f"Error getting embeddings: {response.text}")
            return None
        
        time.sleep(0.5)
    
    return all_embeddings

# Generate text using Hugging Face API
def generate_text(prompt, hf_token, max_tokens=1000):
    api_url = HUGGINGFACE_API_URL + GENERATION_MODEL
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": 0.7,
            "do_sample": True
        }
    }
    
    response = requests.post(api_url, headers=headers, json=payload)
    
    if response.status_code == 200:
        return response.json()[0]["generated_text"].replace(prompt, "").strip()
    else:
        st.error(f"Error generating text: {response.text}")
        return "I couldn't generate a response. Please check your API key and try again."

def get_all_urls(base_url):
    urls = set()
    try:
        response = requests.get(base_url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            for link in soup.find_all("a", href=True):
                url = link["href"]
                full_url = urljoin(base_url, url)
                parsed_url = urlparse(full_url)
                if parsed_url.netloc == urlparse(base_url).netloc:
                    urls.add(parsed_url.scheme + "://" + parsed_url.netloc + parsed_url.path)
    except Exception as e:
        st.error(f"An error occurred while crawling {base_url}: {e}")
    return urls

def extract_text_from_url(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            return " ".join(chunk for chunk in chunks if chunk)
        else:
            st.warning(f"Failed to fetch content from {url}: Status code {response.status_code}")
            return None
    except Exception as e:
        st.warning(f"Error extracting text from {url}: {e}")
        return None

def fetch_url_content(url: str) -> Optional[str]:
    try:
        return extract_text_from_url(url)
    except Exception as e:
        st.error(f"Error: Failed to fetch URL {url}. Exception: {e}")
        return None

def process_and_index_websites(web_urls, hf_token, chunk_size=150, crawl_website=False):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    all_chunks = []
    doc_metadata = []

    urls = [url.strip() for url in web_urls.split(",")]

    if crawl_website:
        all_urls = set()
        progress_bar = st.progress(0)
        progress_text = st.empty()
        for i, base_url in enumerate(urls):
            progress_text.text(f"Crawling website: {base_url}")
            site_urls = get_all_urls(base_url)
            all_urls.update(site_urls)
            progress_bar.progress((i + 1) / len(urls))
        urls = list(all_urls)
        progress_text.text(f"Found {len(urls)} unique URLs")

    progress_bar = st.progress(0)
    progress_text = st.empty()

    for i, url in enumerate(urls):
        progress_text.text(f"Processing URL {i+1}/{len(urls)}: {url}")
        content = fetch_url_content(url)
        if content:
            chunks = text_splitter.split_text(content)
            all_chunks.extend(chunks)
            doc_metadata.extend([{"url": url, "source": "web_content"} for _ in chunks])
        progress_bar.progress((i + 1) / len(urls))
        time.sleep(0.5)

    progress_text.empty()
    progress_bar.empty()

    if not all_chunks:
        st.error("No content to process. Please provide valid web URLs.")
        return None, None

    with st.spinner("Generating embeddings..."):
        embeddings_objects = get_embeddings(all_chunks, hf_token)
        if not embeddings_objects:
            return None, None
        embeddings = [obj["embedding"] for obj in embeddings_objects]

    client = QdrantClient("http://localhost:6333")
    collection_name = "agent_rag_index"
    VECTOR_SIZE = len(embeddings[0])

    with st.spinner("Creating vector database..."):
        try:
            client.delete_collection(collection_name)
        except:
            pass

        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )

        ids = list(range(len(all_chunks)))
        payload = [{"content": chunk, "metadata": metadata} for chunk, metadata in zip(all_chunks, doc_metadata)]

        client.upload_collection(
            collection_name=collection_name,
            vectors=embeddings,
            payload=payload,
            ids=ids,
            batch_size=256,
        )

    st.success(f"Indexed {len(all_chunks)} chunks from {len(urls)} different URLs")
    return client, collection_name

def search_web_for_question(question):
    results = DDGS().text(question, max_results=5)
    web_texts, urls = [], []

    for doc in results:
        urls.append(doc["href"])
        web_texts.append(doc["body"])

    for url in urls[:2]:
        text = fetch_url_content(url)
        if text:
            web_texts.append(text)

    return "\n\n".join(web_texts)

def answer_question(question, hf_token, client=None, collection_name=None, top_k=3):
    if not question.strip():
        st.warning("Please enter a question.")
        return

    with st.spinner("Searching the web for information..."):
        web_context = search_web_for_question(question)

    rag_context = ""
    if client and collection_name:
        with st.spinner("Searching indexed websites..."):
            query_embedding = get_embeddings([question], hf_token)[0]["embedding"]
            results = client.search(collection_name=collection_name, query_vector=query_embedding, limit=top_k)

            formatted_chunks = []
            for doc in results:
                source_info = f"\nSource: {doc.payload['metadata']['url']}"
                formatted_chunks.append(doc.payload["content"] + source_info)

            rag_context = "\n\n".join(formatted_chunks)

    combined_context = rag_context + "\n\n" + web_context

    system_prompt = """You are an expert in answering questions. Provide answers based on the given context. 
If the information is not in the context, try to give a reasonable answer based on general knowledge, but make it clear that it's not from the provided sources.
Format your response in Markdown.
Context: """

    user_prompt = f"""
Question: {question}
Answer:"""

    with st.spinner("Generating your answer..."):
        final_prompt = system_prompt + combined_context + user_prompt
        response = generate_text(final_prompt, hf_token)
        st.markdown(response)

# ---------- Streamlit UI ----------
st.title("Web Search RAG System")

st.subheader("Website Input (optional)")
web_urls = st.text_input(
    "Enter website URLs (comma-separated):", placeholder="https://example.com"
)
crawl_website = st.checkbox("Crawl entire website(s)", help="Enable this to extract content from all pages of the specified website(s)")

if st.button("Index Websites", key="index_button"):
    if not web_urls:
        st.error("Please enter at least one website URL.")
    else:
        st.session_state.client, st.session_state.collection_name = process_and_index_websites(
            web_urls, HUGGINGFACE_TOKEN, crawl_website=crawl_website
        )

st.subheader("Ask a Question")
question = st.text_input("Enter your question:")

if st.button("Get Answer", key="get_answer_button"):
    answer_question(
        question,
        HUGGINGFACE_TOKEN,
        st.session_state.client,
        st.session_state.collection_name
    )

auto_answer = st.checkbox("Auto-answer as I type", value=False)
if auto_answer and question and question != st.session_state.get("last_question", ""):
    st.session_state.last_question = question
    answer_question(
        question,
        HUGGINGFACE_TOKEN,
        st.session_state.client,
        st.session_state.collection_name
    )
