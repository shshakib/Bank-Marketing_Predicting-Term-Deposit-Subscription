"""Request and response contracts for the FastAPI scoring service.

The API accepts only the variables available before or during campaign
planning. Post-call fields such as duration are intentionally not part of the
serving contract because they would create leakage for pre-call scoring.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, validator


CATEGORY_VALUES: dict[str, set[str]] = {
    "job": {
        "admin.",
        "blue-collar",
        "entrepreneur",
        "housemaid",
        "management",
        "retired",
        "self-employed",
        "services",
        "student",
        "technician",
        "unemployed",
        "unknown",
    },
    "marital": {"divorced", "married", "single"},
    "education": {"primary", "secondary", "tertiary", "unknown"},
    "default": {"no", "yes"},
    "housing": {"no", "yes"},
    "loan": {"no", "yes"},
    "contact": {"cellular", "telephone", "unknown"},
    "month": {"jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"},
    "poutcome": {"failure", "other", "success", "unknown"},
}


class BankPredictionRequest(BaseModel):
    """Input payload for scoring one customer.

    The schema mirrors the production feature set used during training. It also
    normalizes categorical strings so the API can accept values like
    ``" Technician "`` while still rejecting categories that were not part of
    the modeling pipeline.
    """

    age: int = Field(..., ge=18, le=100, description="Customer age")
    job: str = Field(..., description="Customer job category")
    marital: str = Field(..., description="Marital status")
    education: str = Field(..., description="Education level")
    default: str = Field(..., description="Has credit in default: yes/no")
    balance: float = Field(..., description="Average yearly balance in euros")
    housing: str = Field(..., description="Has housing loan: yes/no")
    loan: str = Field(..., description="Has personal loan: yes/no")
    contact: str = Field(..., description="Contact communication type")
    day: int = Field(..., ge=1, le=31, description="Last contact day of month")
    month: str = Field(..., description="Last contact month")
    campaign: int = Field(..., ge=1, description="Number of contacts in this campaign")
    previous: int = Field(..., ge=0, description="Number of contacts before this campaign")
    poutcome: str = Field(..., description="Outcome of previous marketing campaign")

    @validator(*CATEGORY_VALUES.keys(), pre=True)
    def normalize_and_validate_categories(cls, value: object, field: Any) -> str:
        """Normalize string categories and reject values unseen by the pipeline."""
        if not isinstance(value, str):
            raise TypeError(f"{field.name} must be provided as a string")

        normalized = value.strip().lower()
        allowed_values = CATEGORY_VALUES[field.name]
        if normalized not in allowed_values:
            allowed_text = ", ".join(sorted(allowed_values))
            raise ValueError(f"{field.name} must be one of: {allowed_text}")
        return normalized

    class Config:
        schema_extra = {
            "example": {
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
                "poutcome": "unknown",
            }
        }


class PredictionResponse(BaseModel):
    """Prediction payload returned by the scoring service.

    ``top_model_factors`` contains global model importances, not a personalized
    SHAP-style explanation for a single customer.
    """

    predicted_deposit: str
    subscription_probability: float
    prediction_label: int
    probability_range: list[float] = Field(
        ...,
        description=(
            "Simple +/- 10 percentage point range around the model probability. "
            "This is a communication range, not a statistical confidence interval."
        ),
    )
    top_model_factors: dict[str, float] = Field(
        ...,
        description="Global model feature importances from the trained estimator.",
    )
    prediction_time: str
