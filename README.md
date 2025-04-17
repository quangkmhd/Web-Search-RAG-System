# Web Search RAG System

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)

## 🧠 Overview

**Web Search RAG System** is an intelligent Retrieval-Augmented Generation (RAG) system that combines:

* Searching and collecting content from websites or the Internet (using DuckDuckGo)
* Text chunking and embedding using Hugging Face
* Semantic storage and retrieval with vector databases (Qdrant)
* Answer generation using Hugging Face large language models

## 🚀 Key Features

* 🕸 **Web Crawler (optional):** Collect content from all pages of specified websites
* 🔍 **Semantic Search:** Intelligent meaning-based search using embeddings
* 💬 **Question Answering (QA):** Generate high-quality answers from crawled information
* 🔌 **Hugging Face API Integration:** Using `sentence-transformers/all-MiniLM-L6-v2` and `HuggingFaceH4/zephyr-7b-beta` models
* 📦 **Vector Storage with Qdrant:** Fast storage and retrieval of embedding vectors

## 🧱 Overall Architecture

```mermaid
graph TD
  A[User inputs URL & Question] --> B{Crawl website?}
  B -- Yes --> C[Collect entire website]
  B -- No --> D[Only fetch main page content]
  C --> E[Split text & create embeddings]
  D --> E
  E --> F[Store embeddings in Qdrant]
  A --> G[Search Internet with DuckDuckGo]
  A --> H[Vector search from Qdrant]
  G --> I[Compile context]
  H --> I
  I --> J[Prompt LLM to generate answer]
```

## 📦 Installation

### System Requirements
* Python 3.8+
* Libraries: `streamlit`, `requests`, `beautifulsoup4`, `duckduckgo_search`, `qdrant-client`, `langchain_text_splitters`, `python-dotenv`

### Install Libraries

```bash
pip install -r requirements.txt
```

### Create `.env` file:

```
HUGGINGFACE_TOKEN=your_huggingface_api_token
```

### Launch the Application

```bash
streamlit run main.py
```

## 🛠 How to Use

1. **Enter website URL(s)**: You can enter multiple URLs separated by commas.
2. **Select crawl entire website** if you want the system to collect all sub-pages.
3. Click **Index Websites** to process and store the content.
4. Enter the **question** you want to ask based on website content + Internet search.
5. Click **Get Answer** to see the response.
6. You can also enable the **Auto-answer** feature to receive answers as you type.

## 🧠 Technologies Used

| Technology | Role |
|-----------|---------|
| Streamlit | User interface |
| Hugging Face | Text embedding and generation |
| Qdrant | Vector store for embedding storage and search |
| DuckDuckGo Search | Real-time Internet search |
| BeautifulSoup | HTML content extraction from web |

## 📎 Example Prompt

```
You are an expert in answering questions. Provide answers based on the given context. If the information is not in the context, try to give a reasonable answer based on general knowledge, but make it clear that it's not from the provided sources. Format your response in Markdown.

Context: <content extracted from web and search>

Question: Youtuber nào có nhiều follower nhất trên youtube tính đến năm 2025

Answer:
```

![Ingredient Identification](img/screenshot-1743952379350.png)
![Ingredient Identification](img/screenshot-1743952401225.png)

## 📌 Notes

* The application requires a **Hugging Face Inference API token**.
* Ensure that the **Qdrant Server** is running at `localhost:6333`.

## 📮 Contributions

Thanks for your interest in contributing. Please create a Pull Request or Issue to help develop this project.

## 🪪 License

MIT License. Free for personal and commercial use.
