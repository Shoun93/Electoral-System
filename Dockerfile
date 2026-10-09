# Build từ thư mục gốc repo: docker build -t ballot-ai .
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app/modules/ballot-validation-ai
COPY dataset /app/dataset
COPY modules/ballot-validation-ai/ ./
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
CMD ["python", "main.py"]