
# Using a lightweight Python base image
FROM python:3.12-slim

# Setting the working directory inside the container
WORKDIR /app

# Installing the required Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copying the application source code and trained model
COPY src/ ./src/
COPY models/ ./models/

# Creating a directory for application logs
RUN mkdir -p logs

# Exposing the API port
EXPOSE 8000

# Starting the FastAPI application
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
