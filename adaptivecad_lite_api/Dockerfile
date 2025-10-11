FROM python:3.11-slim

# System prep
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# Hugging Face Spaces expect the app to listen on $PORT
ENV PORT=7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
