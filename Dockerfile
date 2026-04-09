# Use Python 3.9 slim image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies (useful for some python packages like networkx, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Expose port 7860 (This is the standard port for Hugging Face Spaces)
EXPOSE 7860

# Command to run the Streamlit application
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
