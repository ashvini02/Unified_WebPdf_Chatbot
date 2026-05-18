import os
import streamlit as st
from bs4 import BeautifulSoup
import requests
import warnings
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Suppress warnings
warnings.filterwarnings("ignore")

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variables
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set")

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)

st.title("Chatbot with Scraped Content (URL)")
st.info("Provide a website URL to scrape and interact with its content.")

url = st.text_input("Enter the URL to scrape:")
if url:
    content = None
    try:
        #scraping content from the URL
        # response = requests.get(url, timeout=10) 
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
        response = requests.get(url, headers=headers, timeout=10)   
        if response.status_code != 200:
            st.error(f"Error: Received status code {response.status_code}")
        else:
            soup = BeautifulSoup(response.text, "html.parser")
            content = " ".join([tag.get_text() for tag in soup.find_all(["p", "h1", "h2", "h3", "li", "ol", "ul", "td"])])
            st.success("Website content extracted successfully!")
    except requests.exceptions.RequestException as e:
        st.error(f"Error scraping website: {e}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")

    if content:
        #langChain document from scraped content
        document = Document(page_content=content, metadata={"source": url})

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1400, chunk_overlap=100)
        documents = text_splitter.split_documents([document])
        st.info(f"Number of chunks created: {len(documents)}")

        #FAISS with HuggingFace embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(documents, embeddings)
        st.success("FAISS vector store created.")

        #retriever using MMR (Maximal Marginal Relevance)
        retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 6})

        user_query = st.text_input("Ask a question based on the content:")
        if user_query:
            #retrieve relevant documents for the user's query
            relevant_docs = retriever.invoke(user_query)

            if relevant_docs:
                st.info(f"Retrieved {len(relevant_docs)} relevant document(s).")

                #display metadata and content preview for retrieved chunks
                for i, doc in enumerate(relevant_docs, 1):
                    with st.expander(f"View Chunk {i}"):
                        st.write(f"Metadata: {doc.metadata}")
                        st.write(f"Content: {doc.page_content[:4500]}...")

                #combine relevant chunks into one string
                combined_content = "\n\n".join([doc.page_content for doc in relevant_docs])

                #limiting the combined content length to the model's token capacity
                max_combined_length = 3000
                combined_content = combined_content[:max_combined_length]

                #creating a ChatPromptTemplate
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "Based on the content retrieved from the provided URL, answer the following question:"
                               "If the information is not explicitly mentioned, respond with:"
                               " 'Sorry! The related content is not available in the provided URL.'"),
                    ("system", "{content}"),
                    ("user", "Question: {question}")
                ])

                #input values for the prompt
                prompt_input = {
                    "content": combined_content,
                    "question": user_query
                }

                try:
                    #generate the response 
                    result = llm.invoke(prompt.format_prompt(**prompt_input).to_string()).content
                    st.write("**Answer:**")
                    st.write(result)
                except Exception as e:
                    st.error(f"Error generating response: {e}")
            else:
                st.warning("No relevant documents found. Please refine your query.")