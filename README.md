# Unified PDF & Web Chatbot

A unified Streamlit application that allows users to chat with content extracted seamlessly from both URLs and PDF documents. The application leverages advanced LangChain configurations, HuggingFace embeddings, and the Groq API to provide extremely fast and highly relevant answers based entirely on the provided context.

## Features
- **Dual Support**: Chat with content extracted from websites, PDFs, or both simultaneously.
- **Smart Chunking**: Uses `pymupdf4llm` to extract PDF data page-by-page to retain metadata, coupled with an advanced `RecursiveCharacterTextSplitter`.
- **High-Speed Inference**: Powered by the highly performant `llama-3.1-8b-instant` model on Groq.
- **FAISS Vector Database**: Fast local embeddings storage cached effectively for instant retrieval.

## Prerequisites
- Python 3.9+
- [uv](https://github.com/astral-sh/uv) (Extremely fast Python package installer and resolver)

## Setup Instructions

### 1. Create and Activate Virtual Environment
You can create a virtual environment using standard python or `uv`.

**Using standard Python:**
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

**Using uv:**
```bash
uv venv
# On Windows:
.venv\Scripts\activate
# On Mac/Linux:
source .venv/bin/activate
```

### 2. Install Dependencies
Once activated, use `uv` to install the required packages rapidly:
```bash
uv pip install -r requirements.txt
```

### 3. Environment Variables
To interact with the chatbot, you must configure your API keys.

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Open `.env` and fill in your details:
   - **Free Version (Default)**: Use the `GROQ_API_KEY` for free, fast inference via Groq.
   - **Paid Version (Optional)**: If you decide to modify the code later to use OpenAI or Anthropic models, their respective placeholders exist in the `.env.example`.

### 4. Running the Application
Execute the application using `uv run`:
```bash
uv run streamlit run app.py
```