# Use official lightweight Python image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Leverage Docker layer caching for dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir sentence-transformers==3.4.1

# Copy all source code
COPY . .

# Expose port 80 for Azure App Service
EXPOSE 80

# Azure expects the app to run on 0.0.0.0 and port 80
CMD ["streamlit", "run", "app.py", "--server.port=80", "--server.address=0.0.0.0"]
