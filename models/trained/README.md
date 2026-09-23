Trained bank marketing models are stored here.

Expected artifacts:

- `bank_deposit_model.pkl`
- `preprocessor.pkl`
- `bank_deposit_model_metrics.yaml`
- `bank_model_comparison.yaml`
- `bank_model_comparison.csv`
- `preprocessor_with_duration.pkl`
- `preprocessor_exploratory.pkl`

Pickle files are generated locally and ignored by Git. Production training
creates the model and `preprocessor.pkl` together from cleaned, unencoded data.
The feature-export command should use `preprocessor_exploratory.pkl` so it
cannot overwrite the preprocessor used by the API.

The committed comparison reports are historical results produced before the
preprocessing fixes. Their sensitivity/specificity labels are reversed relative
to the current convention (subscription is positive). Rerun compare_models.py
to replace them; do not treat the historical scores as validation of new code.
