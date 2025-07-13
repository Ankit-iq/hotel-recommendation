# Use official Python image as base
FROM python:3.12-slim

# Set working directory inside the container
WORKDIR /app

# Copy local files to container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port (Flask default)
EXPOSE 5000

# Command to run the app
CMD ["python", "app.py"]
