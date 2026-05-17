## Streamlit interface

Build and run the bank marketing prediction UI as a standalone container:

```bash
docker build -t bank-marketing-streamlit:dev .
docker run -p 8501:8501 -e API_URL=http://host.docker.internal:8000 bank-marketing-streamlit:dev
```

When using the root `docker-compose.yaml`, the `API_URL` is set to the FastAPI service automatically.
The app also checks the FastAPI `/health` endpoint and shows API readiness in the sidebar.
