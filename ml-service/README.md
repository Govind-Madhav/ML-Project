# ML Service - FastAPI

This is the ML microservice for CPU prediction.

## How to Run
- Create venv: `python -m venv venv && source venv/bin/activate`
- Install: `pip install -r requirements.txt`
- Run: `uvicorn main:app --reload`

## Endpoint
- `POST /predict` — Predicts CPU usage

## Loads trained model (.pkl)
