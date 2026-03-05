# Vision & Memory app with FAISS persistence
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py streamlit_app.py ./

# Persistent FAISS index (mount volume at /data)
ENV FAISS_INDEX_DIR=/data/faiss_index
RUN mkdir -p /data/faiss_index

EXPOSE 8501
ENV STREAMLIT_SERVER_PORT=8501
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
