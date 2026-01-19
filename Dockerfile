# ---------- Base Image ----------
FROM python:3.10-slim

# ---------- System Dependencies ----------
RUN apt-get update && apt-get install -y \
    build-essential \
    poppler-utils \
    tesseract-ocr \
    libgl1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# ---------- Environment Variables ----------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ---------- Working Directory ----------
WORKDIR /app

# ---------- Install Python Dependencies ----------
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---------- Copy App Code ----------
COPY . .

# ---------- Streamlit Configuration ----------
EXPOSE 8501

# ---------- Run Streamlit App ----------
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0"]
