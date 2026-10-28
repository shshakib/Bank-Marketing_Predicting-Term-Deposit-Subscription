## FastAPI model service

This app serves the trained bank term-deposit classifier.
It loads `bank_deposit_model.pkl` and `preprocessor.pkl` from `MODEL_DIR`
when that environment variable is set, otherwise it searches the local
`models/trained/` artifact directory.

Container layout:

```
/app
  main.py
  schemas.py
  inference.py
  requirements.txt
  /models
    /trained
      bank_deposit_model.pkl
      preprocessor.pkl
```

Build and run from the project root:

```bash
docker build -t bank-marketing-fastapi:dev -f Dockerfile .
docker run -p 8000:8000 bank-marketing-fastapi:dev
```

The `/health` endpoint reports the loaded artifact paths, artifact timestamps,
and model readiness. If either artifact is missing or cannot be loaded, the
endpoint returns HTTP 503.

Example request:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 41,
    "job": "technician",
    "marital": "married",
    "education": "secondary",
    "default": "no",
    "balance": 1270,
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "day": 15,
    "month": "may",
    "campaign": 2,
    "previous": 0,
    "poutcome": "unknown"
  }'
```

The response returns `subscription_probability`, `probability_range`, and
`top_model_factors`. The probability range is a simple display range around the
score and should not be interpreted as a statistical confidence interval.

`explanation` contains probability-space TreeSHAP values for this customer:
`base_probability`, unrounded `prediction_probability`, signed `contributions`
grouped by original input, and `background_size`. Sum the baseline and all
contributions to reconstruct the score. Multiply contributions by 100 to display
percentage points. `top_model_factors` still describes global model importance.

Training produces `shap_background.pkl`; deploy it with the matching model and
preprocessor, and restart the API. `/health` reports `explanations_available`
separately from model readiness. Missing or stale backgrounds keep predictions
available but return `explanation: null` and an `explanation_error` message.
GradientBoosting and RandomForest are supported; other configured estimators
can score without explanations. `/batch-predict` returns the same contract for
each customer. SHAP values explain model scores, not causal effects.
