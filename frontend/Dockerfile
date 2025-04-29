
FROM python:3.12-slim
EXPOSE 8084
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir streamlit boto3 langchain langchain-community faiss-cpu
COPY . ./
ENTRYPOINT ["python", "-m", "streamlit", "run", "app.py", "--server.port=8084", "--server.address=0.0.0.0"]
