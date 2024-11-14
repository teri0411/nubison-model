# Use Python 3.12 as the base image
FROM python:3.12-slim

# Set the timezone to KST (Korea Standard Time)
RUN apt-get update && \
    apt-get install -y --no-install-recommends tzdata && \
    ln -snf /usr/share/zoneinfo/Asia/Seoul /etc/localtime && \
    echo "Asia/Seoul" > /etc/timezone && \
    dpkg-reconfigure -f noninteractive tzdata && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the requirements.txt file and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port specified by the PORT environment variable
EXPOSE ${PORT}

# Set environment variables with default values (optional)
ENV MLFLOW_TRACKING_URI=""
ENV MLFLOW_MODEL_URI=""
ENV PORT=5000
ENV DEBUG=False

# Define the command to run your application
CMD ["sh", "-c", "bentoml serve --port ${PORT} --debug ${DEBUG}"]
