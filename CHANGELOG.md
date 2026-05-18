# Changelog

## [Unreleased]
### Added
- Created `.env.example` to provide templates for both free (Groq) and paid (OpenAI/Anthropic) environment configurations.
- Created `app.py`, a unified Streamlit application merging both URL scraping and PDF extraction capabilities. Users can now process and chat with a PDF, a URL, or both simultaneously.
- Created `pdf_chatbot.py` to handle interactions with PDF documents.
  - Added PDF file upload support via Streamlit.
  - Implemented `pymupdf4llm` for extracting Markdown from PDF files.
  - Configured `RecursiveCharacterTextSplitter` with `chunk_size=200` and `chunk_overlap=100`.
  - Added FAISS vector store indexing and MMR retrieval.

### Fixed
- Fixed `NameError: name 'content' is not defined` in `url_chatbot.py` by initializing `content = None` and ensuring document ingestion logic only runs if content extraction succeeds.
- Fixed `Received status code 403` error in `url_chatbot.py` by adding a standard `User-Agent` header to the HTTP request to bypass basic anti-bot mechanisms.
### Changed
- Massively updated `README.md` to include a full project overview, feature list, and comprehensive setup instructions (including `venv` creation and `uv` execution commands).
- Improved `pdf_chatbot.py` extraction by using `pymupdf4llm` page-wise chunking to retain metadata.
- Optimized text splitting in `pdf_chatbot.py` using `RecursiveCharacterTextSplitter` with `chunk_size=1000` and Markdown-aware separators.
- Optimized performance in `pdf_chatbot.py` by caching FAISS vector store creation using `@st.cache_resource`.
