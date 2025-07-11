# Dockerfile for spam-api
# Place this file alongside app.py, requirements.txt, best_model.keras, and data/

FROM python:3.9-slim

WORKDIR /app

# Copy and install dependencies, including gunicorn and prometheus client
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir gunicorn prometheus_client

# Copy application code and model artifacts
COPY app.py ./app.py
COPY best_model.keras ./best_model.keras
COPY tokenizer.pickle ./tokenizer.pickle

# Expose the ports the app and metrics serve on
EXPOSE 5000
EXPOSE 8000

# Launch with Gunicorn for production
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000", "--workers", "4"]
