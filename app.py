import os
import tempfile
import requests
import streamlit as st
import pymupdf4llm
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variables
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set")

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.1) # temp=0.1 for less randomness and more accurate responses

@st.cache_resource(show_spinner=False)
def build_vectorstore(documents):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.from_documents(documents, embeddings)

st.title("Unified Content Chatbot")
st.info("Upload a PDF and/or provide a URL to interact with the combined content.")

# Sidebar for inputs
with st.sidebar:
    st.header("Content Sources")
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    url = st.text_input("Enter a URL to scrape:")
    process_button = st.button("Process Content")

if process_button:
    if not uploaded_file and not url:
        st.warning("Please provide at least one source (PDF or URL).")
    else:
        langchain_docs = []
        
        # Process URL
        if url:
            with st.spinner("Scraping URL..."):
                try:
                    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
                    response = requests.get(url, headers=headers, timeout=10)  
                    if response.status_code != 200:
                        st.error(f"Error scraping URL: Received status code {response.status_code}")
                    else:
                        soup = BeautifulSoup(response.text, "html.parser")
                        content = " ".join([tag.get_text() for tag in soup.find_all(["p", "h1", "h2", "h3", "li", "ol", "ul", "td"])])
                        langchain_docs.append(Document(page_content=content, metadata={"source": url}))
                        st.success("Website content extracted successfully!")
                except Exception as e:
                    st.error(f"Error scraping website: {e}")

        # Process PDF
        if uploaded_file:
            with st.spinner("Extracting content from PDF..."):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name

                    page_docs = pymupdf4llm.to_markdown(tmp_file_path, page_chunks=True)
                    os.remove(tmp_file_path)
                    
                    if page_docs:
                        for doc in page_docs:
                            metadata = doc.get("metadata", {})
                            metadata["source"] = uploaded_file.name
                            langchain_docs.append(Document(page_content=doc["text"], metadata=metadata))
                        st.success("PDF content extracted successfully!")
                except Exception as e:
                    st.error(f"Error reading PDF: {e}")
                    if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
                        os.remove(tmp_file_path)

        if langchain_docs:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=200,
                separators=[
                    "\n# ",
                    "\n## ",
                    "\n### ",
                    "\n\n",
                    "\n",
                    ". ",
                    " "
                ]
            )
            documents = text_splitter.split_documents(langchain_docs)
            st.info(f"Number of combined chunks created: {len(documents)}")

            with st.spinner("Creating FAISS vector store..."):
                # Clear cache to force rebuild if content changes
                st.cache_resource.clear() 
                vector_store = build_vectorstore(documents)
            
            # Store in session state so we don't lose it on rerun when querying
            st.session_state.vector_store = vector_store
            st.success("FAISS vector store ready for querying.")
        else:
            st.error("No valid content was extracted from the provided sources.")

# Query interface
if "vector_store" in st.session_state:
    st.divider()
    user_query = st.text_input("Ask a question based on the extracted content:")
    
    if user_query:
        retriever = st.session_state.vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 6})
        relevant_docs = retriever.invoke(user_query)

        if relevant_docs:
            st.info(f"Retrieved {len(relevant_docs)} relevant document(s).")

            for i, doc in enumerate(relevant_docs, 1):
                with st.expander(f"View Chunk {i} (Source: {doc.metadata.get('source', 'Unknown')})"):
                    st.write(f"Metadata: {doc.metadata}")
                    st.write(f"Content: {doc.page_content[:4500]}...")

            combined_content = "\n\n".join([doc.page_content for doc in relevant_docs])
            combined_content = combined_content[:3000]

            prompt = ChatPromptTemplate.from_messages([
                ("system", "Based on the content retrieved from the provided documents, answer the following question:"
                           "If the information is not explicitly mentioned, respond with:"
                           " 'Sorry! The related content is not available in the provided sources.'"),
                ("system", "{content}"),
                ("user", "Question: {question}")
            ])

            prompt_input = {
                "content": combined_content,
                "question": user_query
            }

            try:
                result = llm.invoke(prompt.format_prompt(**prompt_input).to_string()).content
                st.write("**Answer:**")
                st.write(result)
            except Exception as e:
                st.error(f"Error generating response: {e}")
        else:
            st.warning("No relevant documents found. Please refine your query.")
