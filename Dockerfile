# Use a lightweight Python image pinned to Debian bookworm
# so OpenJDK 17 package names are available.
FROM python:3.9-slim-bookworm

# Install Java 17 (compatible with Spark 3.4.x) and procps (for Spark monitoring)
RUN apt-get update && apt-get install -y \
    openjdk-17-jdk-headless \
    procps \
    && apt-get clean

# Set environment variables for Spark.
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
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