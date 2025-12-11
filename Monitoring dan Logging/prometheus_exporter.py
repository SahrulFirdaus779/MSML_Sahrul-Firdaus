r"""Custom Prometheus exporter for system + model artifact stats.

Run:
    python prometheus_exporter.py --port 9101 --model-path ..\Membangun_model\tuning_artifacts\rf_grid_model.joblib --preprocess-path ..\Membangun_model\data_clustering_preprocessing\preprocess_pipeline.joblib
"""

import argparse
import os
import time
from pathlib import Path

import joblib
import psutil
from prometheus_client import Gauge, Counter, Summary, start_http_server

PROCESS_CPU = Gauge("exporter_process_cpu_percent", "Exporter process CPU percent")
PROCESS_MEM_MB = Gauge("exporter_process_memory_mb", "Exporter process memory MB")
ARTIFACT_MTIME = Gauge("model_artifact_mtime", "Unix mtime of model artifact")
PIPELINE_MTIME = Gauge("preprocess_artifact_mtime", "Unix mtime of preprocess artifact")
ARTIFACT_EXISTS = Gauge("model_artifact_exists", "Model file exists (1/0)")
PIPELINE_EXISTS = Gauge("preprocess_artifact_exists", "Preprocess file exists (1/0)")

# --- Additional ML metrics for monitoring ---
INFERENCE_LATENCY = Summary("inference_latency_seconds", "Inference latency in seconds")
INFERENCE_THROUGHPUT = Gauge("inference_throughput", "Number of inferences per minute")
INFERENCE_ACCURACY = Gauge("inference_accuracy", "Model accuracy on last batch")
INFERENCE_ERROR_RATE = Gauge("inference_error_rate", "Error rate of inference (0-1)")
INFERENCE_REQUEST_COUNT = Counter("inference_request_count", "Total number of inference requests")
INFERENCE_PRECISION = Gauge("inference_precision", "Model precision on last batch")
INFERENCE_RECALL = Gauge("inference_recall", "Model recall on last batch")
INFERENCE_F1 = Gauge("inference_f1", "Model F1 score on last batch")
MODEL_VERSION = Gauge("model_version", "Model version as float (e.g., 1.0, 2.1)")
INFERENCE_SUCCESS_RATE = Gauge("inference_success_rate", "Success rate of inference (0-1)")


class Exporter:
    def __init__(self, model_path: Path, preprocess_path: Path, interval: float = 5.0):
        self.model_path = model_path
        self.preprocess_path = preprocess_path
        self.interval = interval
        self.proc = psutil.Process()

    def collect_loop(self):
        import random
        version = 1.0
        while True:
            cpu = self.proc.cpu_percent(interval=None)
            mem = self.proc.memory_info().rss / (1024 * 1024)
            PROCESS_CPU.set(cpu)
            PROCESS_MEM_MB.set(mem)

            ARTIFACT_EXISTS.set(1 if self.model_path.exists() else 0)
            PIPELINE_EXISTS.set(1 if self.preprocess_path.exists() else 0)

            if self.model_path.exists():
                ARTIFACT_MTIME.set(self.model_path.stat().st_mtime)
            if self.preprocess_path.exists():
                PIPELINE_MTIME.set(self.preprocess_path.stat().st_mtime)

            # --- Dummy/random values for demonstration ---
            latency = random.uniform(0.01, 0.2)
            throughput = random.randint(10, 100)
            accuracy = random.uniform(0.7, 1.0)
            error_rate = random.uniform(0, 0.2)
            precision = random.uniform(0.7, 1.0)
            recall = random.uniform(0.7, 1.0)
            f1 = 2 * (precision * recall) / (precision + recall)
            success_rate = 1.0 - error_rate

            INFERENCE_LATENCY.observe(latency)
            INFERENCE_THROUGHPUT.set(throughput)
            INFERENCE_ACCURACY.set(accuracy)
            INFERENCE_ERROR_RATE.set(error_rate)
            INFERENCE_REQUEST_COUNT.inc(random.randint(1, 5))
            INFERENCE_PRECISION.set(precision)
            INFERENCE_RECALL.set(recall)
            INFERENCE_F1.set(f1)
            MODEL_VERSION.set(version)
            INFERENCE_SUCCESS_RATE.set(success_rate)

            time.sleep(self.interval)


def parse_args():
    parser = argparse.ArgumentParser(description="Prometheus exporter for system and artifact metrics")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9101)
    parser.add_argument("--model-path", type=Path, default=Path("..") / "Membangun_model" / "tuning_artifacts" / "rf_grid_model.joblib")
    parser.add_argument("--preprocess-path", type=Path, default=Path("..") / "Membangun_model" / "data_clustering_preprocessing" / "preprocess_pipeline.joblib")
    parser.add_argument("--interval", type=float, default=5.0, help="Seconds between scrapes")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"Starting exporter on {args.host}:{args.port}")
    print(f"Model path: {args.model_path.resolve()}")
    print(f"Preprocess path: {args.preprocess_path.resolve()}")
    start_http_server(addr=args.host, port=args.port)
    exporter = Exporter(args.model_path, args.preprocess_path, args.interval)
    exporter.collect_loop()
