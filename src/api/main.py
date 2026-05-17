"""FastAPI application that exposes the trained bank marketing classifier."""

from fastapi import FastAPI, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware

try:
    from .inference import batch_predict, get_model_health, predict_subscription
    from .schemas import BankPredictionRequest, PredictionResponse
except ImportError:
    from inference import batch_predict, get_model_health, predict_subscription
    from schemas import BankPredictionRequest, PredictionResponse

app = FastAPI(
    title="Bank Term Deposit Prediction API",
    description=(
        "Predicts whether a bank customer is likely to subscribe to a term "
        "deposit after a direct marketing campaign."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=dict)
async def health_check(response: Response):
    """Report whether the serving artifacts are loaded and which files are used."""
    health = get_model_health()
    if not health["model_loaded"]:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return health


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: BankPredictionRequest):
    """Score one customer record."""
    try:
        return predict_subscription(request)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/batch-predict", response_model=list[PredictionResponse])
async def batch_predict_endpoint(requests: list[BankPredictionRequest]):
    """Score multiple customer records using the same validation contract."""
    try:
        return batch_predict(requests)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
