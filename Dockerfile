# Use Python 3.10 slim base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (for Pillow, etc.)
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire app
COPY . .

# Create upload dir (in case static/uploads doesn't exist in repo)
RUN mkdir -p static/uploads

# Expose port
EXPOSE 5000

# Run app
CMD ["python", "app.py"]