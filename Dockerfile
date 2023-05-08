# Start with Python 3.9 slim buster base image
FROM python:3.9-slim-buster

# Copy the prerequisites shell script to the container
COPY prereq.sh /

# Run the prerequisites script
RUN chmod +x /prereq.sh && /prereq.sh

# Set the working directory to /app
WORKDIR /app

# Copy the project files to the container
COPY . .

# Install the required Python packages
# RUN pip install --no-cache-dir -r requirements.txt

# Expose port 10000
EXPOSE 10000

# Launch the project with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
