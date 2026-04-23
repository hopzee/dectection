# Fake Signature Detection Streamlit App

This project provides a simple Streamlit interface for signature authenticity checks.

## Run

### Windows

Double-click `run_app.bat`.

If this is your first time, you can run `setup.bat` once, then use `run_app.bat` after that.

### Manual

```bash
streamlit run app.py
```

## Model Support

Place a trained model in `models/` with one of these names:

- `signature_model.pkl`
- `signature_model.joblib`
- `signature_model.h5`

If no model is present, the app runs in demo mode.
