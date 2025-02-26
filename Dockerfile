# Use an official Python runtime as the base image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code to the working directory
COPY src/ ./src/
COPY models/ ./models/
COPY Makefile .

# Expose the necessary ports
EXPOSE 8000
EXPOSE 8501

# Set environment variables if needed
# ENV VARIABLE_NAME=value

# Run the application
CMD ["make", "run-api", "&&", "make", "run-streamlit"]
