# Use official Python slim image (smaller, faster builds)
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy and install dependencies first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the port Hugging Face expects (7860)
EXPOSE 7860

# Use our start script
CMD ["bash", "start.sh"]