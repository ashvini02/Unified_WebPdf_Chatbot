import os
import streamlit as st
import tempfile
import warnings
import pymupdf4llm
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

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.1)

@st.cache_resource(show_spinner=False)
def build_vectorstore(documents):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.from_documents(
        documents,
        embeddings
    )

st.title("PDF Chatbot")
st.info("Upload a PDF to interact with its content.")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    page_docs = None
    try:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        # Extract markdown from PDF using pymupdf4llm
        with st.spinner("Extracting content from PDF..."):
            page_docs = pymupdf4llm.to_markdown(tmp_file_path, page_chunks=True)
            
        # Clean up temporary file
        os.remove(tmp_file_path)
        st.success("PDF content extracted successfully!")
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        # Make sure to clean up if an error occurs
        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

    if page_docs:
        # Create langchain documents from page chunks
        langchain_docs = []
        for doc in page_docs:
            metadata = doc.get("metadata", {})
            metadata["source"] = uploaded_file.name
            langchain_docs.append(Document(page_content=doc["text"], metadata=metadata))

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
        st.info(f"Number of chunks created: {len(documents)}")

        #FAISS with HuggingFace embeddings
        with st.spinner("Creating FAISS vector store..."):
            vector_store = build_vectorstore(documents)
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
                    ("system", "Based on the content retrieved from the provided PDF, answer the following question:"
                               "If the information is not explicitly mentioned, respond with:"
                               " 'Sorry! The related content is not available in the provided PDF.'"),
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
