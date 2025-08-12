# Use slim base with Python 3.11 (fast + smaller)
FROM python:3.11-slim

# System deps for lxml, pandas, pillow, etc.
RUN apt-get update && apt-get install -y \
    build-essential \
    libxml2-dev libxslt1-dev \
    libjpeg-dev zlib1g-dev \
    libatlas-base-dev gfortran \
    && rm -rf /var/lib/apt/lists/*

# Workdir
WORKDIR /app

# Copy requirements and install
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY new_git_3.py /app/main.py

# Vercel sets PORT; Uvicorn must bind 0.0.0.0
ENV PORT=8000
EXPOSE 8000

# Start FastAPI (lifespan already handled in your file)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "${PORT}"]
