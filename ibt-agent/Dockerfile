# Build stage
FROM artifactory.fepoc.com:443/docker-virtual/ironbank/opensource/python:v3.12

# Set working directory in container
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port for FastAPI
EXPOSE 8000

# Set working directory to app
WORKDIR /app
ENTRYPOINT []

## Run the FastAPI application using the new main
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
