# Use a lightweight Python image
FROM python:3.9-slim

# Install Java (Required for Spark) and procps (for Spark monitoring)
RUN apt-get update && apt-get install -y \
    default-jdk-headless \
    procps \
    && apt-get clean

# Set Environment Variables for Spark
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV SPARK_HOME=/usr/local/lib/python3.9/site-packages/pyspark
ENV PATH=$PATH:$JAVA_HOME/bin

# Set work directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Expose ports: 8501 (Streamlit), 4040 (Spark UI)
EXPOSE 8501 4040

# Command to keep container running (we will override this in docker-compose)
CMD ["bash"]