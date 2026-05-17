FROM python:3.11-slim

# Keep Python logs visible in Docker and prevent bytecode files from being
# written into the container filesystem.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/app/models/trained

WORKDIR /app

# Install API dependencies before copying source code so Docker can reuse the
# dependency layer when only application code changes.
COPY src/api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/api/ .

# The image includes artifacts as a fallback. docker-compose mounts
# ./models/trained over this directory so retrained local models are served
# without rebuilding the image during development.
COPY models/trained/*.pkl models/trained/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).read()"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
