# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Copy requirements.txt and install dependencies

# Copy the application code
COPY . .

# Expose the port for the FastAPI app
EXPOSE 8000

# Default command for the video-scheduler service
CMD ["uvicorn", "services.video_scheduler.server:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
