# Bank Marketing - Predicting Term Deposit Subscription

This project adapts the original MLOps template into an end-to-end bank marketing classifier. It predicts whether a customer is likely to subscribe to a term deposit using the Bank Marketing dataset.

Here is the production feature set:

- Target: `deposit`
- Task: binary classification
- Excluded from prediction: `duration`, because it is only known after the call
- Excluded from prediction: `pdays`, because the `-1` value dominates and was treated as unreliable
- Default model: Gradient Boosting classifier
- Evaluation metrics: accuracy, precision, recall, specificity, balanced accuracy, F1, and ROC AUC
- Preprocessing: `default`, `housing`, and `loan` are label-encoded; other categorical fields use full-rank one-hot encoding
- Comparison: Random Forest, GBM, Logistic Regression, and LDA are evaluated both with and without `duration`

## Project Structure

```
.
├── configs/                 # Model configuration
├── data/
│   ├── raw/                 # bank.csv
│   └── processed/           # cleaned and feature-engineered data
├── deployment/              # MLflow and Kubernetes examples
├── models/trained/          # Model, preprocessor, and metrics artifacts
├── notebooks/               # Data science notebooks
├── reports/figures/         # Python-generated versions of the R EDA plots
├── src/
│   ├── api/                 # FastAPI model service
│   ├── data/                # Data cleaning pipeline
│   ├── features/            # Feature engineering pipeline
│   ├── models/              # Training and comparison pipelines
│   └── visualization/       # EDA plot generation
├── streamlit_app/           # Prediction UI
├── Dockerfile               # FastAPI container
└── docker-compose.yaml      # FastAPI + Streamlit
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On macOS/Linux, activate with:

```bash
source .venv/bin/activate
```

## Pipeline

Run the project from raw data to trained model:

```bash
python src/data/run_processing.py `
  --input data/raw/bank.csv `
  --output data/processed/cleaned_bank_data.csv
```

Generate the EDA figures used in the analysis:

```bash
python src/visualization/create_plots.py `
  --input data/raw/bank.csv `
  --output-dir reports/figures
```

```bash
python src/features/engineer.py `
  --input data/processed/cleaned_bank_data.csv `
  --output data/processed/featured_bank_data.csv `
  --preprocessor models/trained/preprocessor.pkl
```

```bash
python src/models/train_model.py `
  --config configs/model_config.yaml `
  --data data/processed/featured_bank_data.csv `
  --models-dir models
```

To log to MLflow, start the tracking server and add `--mlflow-tracking-uri http://localhost:5555` to the training command.

```bash
docker compose -f deployment/mlflow/docker-compose.yaml up -d
```

## Model Comparison

Compare models both without `duration` and with `duration`.

```bash
python src/models/compare_models.py `
  --raw-data data/raw/bank.csv `
  --output models/trained/bank_model_comparison.yaml `
  --mlflow-tracking-uri http://localhost:5555
```

By default this uses 100 Monte Carlo CV iterations for Logistic Regression and LDA. The GBM command uses the best GBM configuration unless `--full-gbm-grid` is provided.

For the with-duration feature matrix:

```bash
python src/data/run_processing.py `
  --input data/raw/bank.csv `
  --output data/processed/cleaned_bank_data_with_duration.csv `
  --keep-duration

python src/features/engineer.py `
  --input data/processed/cleaned_bank_data_with_duration.csv `
  --output data/processed/featured_bank_data_with_duration.csv `
  --preprocessor models/trained/preprocessor_with_duration.pkl
```

## FastAPI

Run locally:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Example prediction:

```bash
curl -X POST "http://localhost:8000/predict" `
  -H "Content-Type: application/json" `
  -d "{\"age\":41,\"job\":\"technician\",\"marital\":\"married\",\"education\":\"secondary\",\"default\":\"no\",\"balance\":1270,\"housing\":\"yes\",\"loan\":\"no\",\"contact\":\"cellular\",\"day\":15,\"month\":\"may\",\"campaign\":2,\"previous\":0,\"poutcome\":\"unknown\"}"
```

## Streamlit

Run locally:

```bash
streamlit run streamlit_app/app.py
```

The app reads `API_URL`; if it is unset, it uses `http://localhost:8000`.

## Docker Compose

Build and run FastAPI plus Streamlit:

```bash
docker compose up --build
```

Open:

- FastAPI docs: http://localhost:8000/docs
- Streamlit UI: http://localhost:8501

## GitHub Actions

The workflows in `.github/workflows/` run the same pipeline:

1. Clean `data/raw/bank.csv`
2. Generate the EDA figures
3. Engineer production and with-duration comparison features
4. Run model comparison and log it to MLflow
5. Train `bank_deposit_model.pkl`
6. Build and publish the FastAPI container
