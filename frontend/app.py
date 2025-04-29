
"""
PDF CHAT CLIENT APPLICATION
--------------------------
An elegant RAG (Retrieval Augmented Generation) system that allows users to chat with
their PDF documents using AWS Bedrock and FAISS vector database.
"""

import boto3
import streamlit as st
import os
import uuid
import time
from typing import Any

os.environ["AWS_REGION"] = "eu-central-1"

st.set_page_config(
    page_title="PDF Chat Assistant",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    \"\"\"
    <style>
    .main-header { font-size: 2.5rem; color: #1E3A8A; margin-bottom: 1rem; }
    .subheader   { font-size: 1.5rem; color: #3B82F6; margin-bottom: 2rem; }
    .stButton>button { background-color: #2563EB; color: white; border-radius: 6px; padding: 0.5rem 1rem; font-weight: bold; }
    .stTextInput>div>div>input { border-radius: 6px; }
    </style>
\"\"\", unsafe_allow_html=True)

s3_client = boto3.client("s3")
BUCKET_NAME = os.getenv("BUCKET_NAME")

from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.llms import Bedrock as BedrockLLM
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS

bedrock_client = boto3.client(service_name="bedrock-runtime", region_name="eu-central-1")
bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1", 
    client=bedrock_client
)
folder_path = "/tmp/"

def get_unique_id() -> str:
    return str(uuid.uuid4())

def load_index() -> None:
    try:
        with st.status("Downloading vector index from S3...") as status:
            s3_client.download_file(Bucket=BUCKET_NAME, Key="my_faiss.faiss", Filename=f"{folder_path}my_faiss.faiss")
            status.update(label="Downloaded FAISS index", state="running", expanded=True)
            s3_client.download_file(Bucket=BUCKET_NAME, Key="my_faiss.pkl", Filename=f"{folder_path}my_faiss.pkl")
            status.update(label="Vector index ready!", state="complete")
    except Exception as e:
        st.error(f"Error downloading index: {str(e)}")
        st.stop()

def get_llm() -> BedrockLLM:
    return BedrockLLM(
        model_id="eu.meta.llama3-2-1b-instruct-v1:0",
        client=bedrock_client,
        model_kwargs={
            "temperature": 0.2,
            "maxTokens": 512,
            "topP": 0.9,
        }
    )

def get_response(llm: BedrockLLM, vectorstore: FAISS, question: str) -> str:
    prompt_template = \"\"\"\
Instructions: You are an expert document assistant. Please use only the information in the provided context 
to answer the question accurately and concisely.

If the context doesn't contain the information needed to answer the question, respond with:
"I don't have enough information in the document to answer this question."

Context:
{context}

Question: {question}

Answer:\"\"\"
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa.invoke({"query": question})
    return answer["result"]

def main():
    st.markdown('<div class="main-header">PDF Chat Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="subheader">Powered by AWS Bedrock & RAG Architecture</div>', unsafe_allow_html=True)
    with st.sidebar:
        st.image("https://placeholder.pics/svg/300x100/DEDEDE/555555/AWS%20Bedrock", width=300)
        st.subheader("About this app")
        st.info(\"\"\"
        This application uses Retrieval Augmented Generation (RAG) to chat with your PDF documents.
        The system:
        1. Loads pre-indexed document vectors from S3
        2. Matches your question with relevant document sections
        3. Uses Amazon Nova to generate precise answers based on document content
        \"\"\")
        st.subheader("Session Info")
        session_id = get_unique_id()
        st.code(f"Session ID: {session_id}")
    with st.spinner("Preparing document index..."):
        load_index()
        with st.expander("System Debug Info", expanded=False):
            dir_list = os.listdir(folder_path)
            st.write(f"Files in {folder_path}:"); st.code("\n".join(dir_list))
    try:
        faiss_index = FAISS.load_local(
            index_name="my_faiss",
            folder_path=folder_path,
            embeddings=bedrock_embeddings,
            allow_dangerous_deserialization=True
        )
        st.success("📚 Document knowledge base successfully loaded!")
    except Exception as e:
        st.error(f"Failed to load index: {str(e)}"); st.stop()
    st.subheader("Ask me anything about your document")
    question = st.text_input("Your question:", placeholder="E.g., What are the main points discussed in the document?")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if st.button("Ask 🔍") and question:
        st.session_state.chat_history.append({"question": question, "answer": None})
        with st.spinner("Analyzing document and generating answer..."):
            llm = get_llm()
            start_time = time.time()
            response = get_response(llm, faiss_index, question)
            response_time = time.time() - start_time
            st.session_state.chat_history[-1].update(answer=response, time=f"{response_time:.2f}s")
    if st.session_state.chat_history:
        st.markdown("### Conversation")
        for ex in st.session_state.chat_history:
            st.markdown(f"**Q: {ex['question']}**")
            if ex["answer"]:
                st.markdown(ex["answer"])
                st.caption(f"Response time: {ex['time']}")
            st.divider()
    st.markdown("---")
    st.caption("PDF Chat Assistant © 2025")
if __name__ == "__main__":
    main()
