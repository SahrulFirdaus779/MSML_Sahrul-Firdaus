r"""FastAPI inference service with Prometheus metrics.

Run example:
  uvicorn inference:app --host 0.0.0.0 --port 8000
or
  python inference.py --model-path ..\Membangun_model\tuning_artifacts\rf_grid_model.joblib --preprocess-path ..\Membangun_model\data_clustering_preprocessing\preprocess_pipeline.joblib --host 0.0.0.0 --port 8000
"""

import argparse
import logging
import os
import time
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
import uvicorn

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR.parent / "Membangun_model" / "tuning_artifacts" / "rf_grid_model.joblib"
DEFAULT_PREPROCESS_PATH = BASE_DIR.parent / "Membangun_model" / "data_clustering_preprocessing" / "preprocess_pipeline.joblib"

logger = logging.getLogger("inference")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Prometheus metrics
REQUEST_COUNTER = Counter("inference_requests_total", "Total inference requests", ["endpoint", "status"])
REQUEST_LATENCY = Histogram(
    "inference_request_latency_seconds",
    "Inference request latency",
    buckets=(0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0),
)
PREPROCESS_DURATION = Histogram(
    "preprocess_duration_seconds",
    "Duration for preprocessing per request",
    buckets=(0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0),
)
INFERENCE_DURATION = Histogram(
    "inference_duration_seconds",
    "Duration for model inference per request",
    buckets=(0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0),
)
PREDICTION_CLASS_TOTAL = Counter("prediction_class_total", "Count per predicted class", ["class_label"])
PREDICTION_CONFIDENCE = Histogram(
    "prediction_confidence",
    "Distribution of top prediction confidence",
    buckets=(0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0),
)
ERROR_COUNTER = Counter("inference_errors_total", "Error count", ["endpoint", "reason"])
MODEL_LOADED_AT = Gauge("model_loaded_timestamp", "Unix timestamp when model loaded")
PIPELINE_LOADED_AT = Gauge("pipeline_loaded_timestamp", "Unix timestamp when preprocess pipeline loaded")

# Metrik tambahan
REQUEST_PER_USER = Counter("inference_requests_per_user", "Total inference requests per user", ["user_id"])
REQUEST_INPUT_SIZE = Histogram("inference_request_input_size", "Jumlah data per request", buckets=(1, 2, 5, 10, 20, 50, 100))
RESPONSE_STATUS_CODE = Counter("inference_response_status_code_total", "Total response per status code", ["status_code"])
PREDICTION_TIME_PER_CLASS = Histogram("prediction_time_per_class", "Prediction time per class", ["class_label"], buckets=(0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0))


class Transaction(BaseModel):
    TransactionAmount: float = Field(..., description="Amount of the transaction")
    TransactionType: float
    Location: float
    Channel: float
    CustomerAge: float
    CustomerOccupation: float
    TransactionDuration: float
    LoginAttempts: float
    AccountBalance: float
    AgeGroup: int
    TransactionSize: int


class Settings(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_path: Path = Field(default=DEFAULT_MODEL_PATH)
    preprocess_path: Path = Field(default=DEFAULT_PREPROCESS_PATH)
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)


def load_artifacts(model_path: Path, preprocess_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    if not preprocess_path.exists():
        raise FileNotFoundError(f"Preprocess pipeline not found at {preprocess_path}")

    model = joblib.load(model_path)
    preprocess = joblib.load(preprocess_path)

    MODEL_LOADED_AT.set_to_current_time()
    PIPELINE_LOADED_AT.set_to_current_time()
    logger.info("Loaded model from %s", model_path)
    logger.info("Loaded preprocess pipeline from %s", preprocess_path)
    return model, preprocess


def build_app(settings: Settings) -> FastAPI:
    model, preprocess = load_artifacts(settings.model_path, settings.preprocess_path)
    feature_names_out = preprocess.get_feature_names_out()

    app = FastAPI(title="Inference Service", version="1.0.0")

    @app.get("/health")
    def health():
        return {"status": "ok", "model": str(settings.model_path.name), "preprocess": str(settings.preprocess_path.name)}

    # Endpoint statistik metrik Prometheus
    from prometheus_client import REGISTRY
    @app.get("/stats")
    def stats():
        metrics = {}
        for metric in REGISTRY.collect():
            samples = []
            for s in metric.samples:
                samples.append({
                    "name": s.name,
                    "labels": s.labels,
                    "value": s.value
                })
            metrics[metric.name] = samples
        return metrics

    @app.post("/predict")
    def predict(payload: List[Transaction]):
        start_ts = time.perf_counter()
        endpoint = "/predict"
        # Contoh user_id, bisa diganti sesuai kebutuhan (misal dari payload, header, dsb)
        user_id = "anonymous"
        if not payload:
            ERROR_COUNTER.labels(endpoint=endpoint, reason="empty_payload").inc()
            REQUEST_COUNTER.labels(endpoint=endpoint, status="bad_request").inc()
            RESPONSE_STATUS_CODE.labels(status_code=400).inc()
            raise HTTPException(status_code=400, detail="Payload tidak boleh kosong")

        df = pd.DataFrame([row.model_dump() for row in payload])
        # Metrik input size
        REQUEST_INPUT_SIZE.observe(len(payload))

        try:
            t0 = time.perf_counter()
            transformed = preprocess.transform(df)
            PREPROCESS_DURATION.observe(time.perf_counter() - t0)

            t1 = time.perf_counter()
            proba = model.predict_proba(transformed)
            preds = np.argmax(proba, axis=1)
            inference_time = time.perf_counter() - t1
            INFERENCE_DURATION.observe(inference_time)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Error during inference")
            ERROR_COUNTER.labels(endpoint=endpoint, reason="processing_error").inc()
            REQUEST_COUNTER.labels(endpoint=endpoint, status="error").inc()
            RESPONSE_STATUS_CODE.labels(status_code=500).inc()
            raise HTTPException(status_code=500, detail=str(exc))

        results = []
        classes = getattr(model, "classes_", None)
        for idx, pred in enumerate(preds):
            label = classes[pred] if classes is not None else pred
            prob = float(proba[idx][pred])
            PREDICTION_CLASS_TOTAL.labels(class_label=str(label)).inc()
            PREDICTION_CONFIDENCE.observe(prob)
            # Update metrik waktu prediksi per kelas
            PREDICTION_TIME_PER_CLASS.labels(class_label=str(label)).observe(inference_time)
            results.append({"predicted_class": int(label) if isinstance(label, (np.integer, int)) else label, "confidence": prob})

        duration = time.perf_counter() - start_ts
        REQUEST_COUNTER.labels(endpoint=endpoint, status="ok").inc()
        REQUEST_LATENCY.observe(duration)
        # Update metrik request per user
        REQUEST_PER_USER.labels(user_id=user_id).inc()
        RESPONSE_STATUS_CODE.labels(status_code=200).inc()

        return {"count": len(results), "predictions": results, "features": list(feature_names_out)}

    @app.get("/metrics")
    def metrics():
        data = generate_latest()
        return Response(content=data, media_type=CONTENT_TYPE_LATEST)

    return app


def parse_args():
    parser = argparse.ArgumentParser(description="Serve model with FastAPI + Prometheus metrics")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="Path ke model joblib")
    parser.add_argument("--preprocess-path", type=Path, default=DEFAULT_PREPROCESS_PATH, help="Path ke preprocess pipeline joblib")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host bind")
    parser.add_argument("--port", type=int, default=8000, help="Port bind")
    return parser.parse_args()


args = parse_args() if __name__ == "__main__" else None
app = build_app(Settings())


if __name__ == "__main__":
    settings = Settings(model_path=args.model_path, preprocess_path=args.preprocess_path, host=args.host, port=args.port)
    app = build_app(settings)
    uvicorn.run(app, host=settings.host, port=settings.port)
