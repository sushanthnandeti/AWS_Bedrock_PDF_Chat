
"""
AWS PDF PROCESSOR AND VECTOR STORE CREATOR
------------------------------------------
This application processes PDF documents, splits them into chunks, 
creates vector embeddings using AWS Bedrock, and stores them in FAISS.
The vectors are then saved to S3 for later retrieval and use.
"""

import boto3
import streamlit as st
import os
import uuid
from typing import List, Any

# Set AWS region as environment variable (must be done before creating boto3 clients)
os.environ["AWS_REGION"] = "eu-central-1"

# Initialize S3 client for storing vector databases
s3_client = boto3.client("s3")
BUCKET_NAME = os.getenv("BUCKET_NAME")

# Import necessary components for document processing and embeddings
from langchain_community.embeddings import BedrockEmbeddings  # AWS Bedrock embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter  # For chunking documents
from langchain_community.document_loaders import PyPDFLoader  # PDF handling
from langchain_community.vectorstores import FAISS  # Vector storage

# Initialize AWS Bedrock client with appropriate region
bedrock_client = boto3.client(service_name="bedrock-runtime", region_name="eu-central-1")

# Configure embeddings model using AWS Titan embeddings model
bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1", 
    client=bedrock_client
)

def get_unique_id() -> str:
    return str(uuid.uuid4())

def split_text(pages: List[Any], chunk_size: int, chunk_overlap: int) -> List[Any]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(pages)

def create_vector_store(request_id: str, documents: List[Any]) -> bool:
    vectorstore_faiss = FAISS.from_documents(documents, bedrock_embeddings)
    file_name = f"{request_id}.bin"
    folder_path = "/tmp/"
    vectorstore_faiss.save_local(index_name=file_name, folder_path=folder_path)
    try:
        s3_client.upload_file(
            Filename=folder_path + "/" + file_name + ".faiss", 
            Bucket=BUCKET_NAME, 
            Key="my_faiss.faiss"
        )
        s3_client.upload_file(
            Filename=folder_path + "/" + file_name + ".pkl", 
            Bucket=BUCKET_NAME, 
            Key="my_faiss.pkl"
        )
        return True
    except Exception as e:
        st.error(f"Error uploading to S3: {str(e)}")
        return False

def main():
    st.title("PDF Vector Processor")
    st.subheader("Backend Interface for Chat with PDF Demo")
    with st.sidebar:
        st.header("Upload Document")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        request_id = get_unique_id()
        st.info(f"Processing Request ID: {request_id}")
        saved_file_name = f"{request_id}.pdf"
        with open(saved_file_name, "wb") as w:
            w.write(uploaded_file.getvalue())
        with st.spinner("Loading PDF document..."):
            loader = PyPDFLoader(saved_file_name)
            pages = loader.load_and_split()
            st.success(f"Successfully loaded {len(pages)} pages from PDF")
        with st.spinner("Splitting document into chunks..."):
            splitted_docs = split_text(pages, chunk_size=1000, chunk_overlap=200)
            st.success(f"Document split into {len(splitted_docs)} chunks")
        with st.expander("Preview document chunks"):
            st.subheader("Sample Chunk 1"); st.write(splitted_docs[0])
            st.subheader("Sample Chunk 2"); st.write(splitted_docs[1])
        with st.spinner("Creating vector embeddings and storing in FAISS..."):
            st.text("This might take a few moments depending on document size")
            result = create_vector_store(request_id, splitted_docs)
        if result:
            st.success("PDF processed successfully! Vector store created and uploaded to S3.")
            st.balloons()
        else:
            st.error("Error processing document. Please check application logs.")

if __name__ == "__main__":
    st.set_page_config(
        page_title="PDF Vector Processor",
        page_icon="📚",
        layout="wide"
    )
    main()
