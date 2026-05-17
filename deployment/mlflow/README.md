## MLflow tracking server

This Compose file starts an optional local MLflow server for experiment tracking.
The server is configured to proxy artifact uploads, which means training scripts
send model artifacts through the tracking URI instead of trying to write directly
to a container-local path.

```bash
docker compose -f deployment/mlflow/docker-compose.yaml up -d
```

Training and model-comparison scripts can log to it with:

```bash
--mlflow-tracking-uri http://localhost:5555
```

The Compose setup stores the SQLite backend database and proxied artifact files
in the `mlflow-data` Docker volume, so runs are preserved when the container is
recreated.
