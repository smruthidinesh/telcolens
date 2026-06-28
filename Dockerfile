FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TELCOLENS_DEMO=1

# tesseract — OCR engine for scanned/image PDF pages (used by pytesseract)
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY data/sample/ ./data/sample/

EXPOSE 8000

# bind to the platform-provided $PORT (Render/Fly inject it); default 8000 locally / HF
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
