from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["service", "endpoint", "status"],
)
http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration",
    ["service", "endpoint"],
)
model_inference_duration_seconds = Histogram(
    "model_inference_duration_seconds",
    "Model inference duration",
)
feedback_votes_total = Counter(
    "feedback_votes_total",
    "Feedback votes",
    ["vote"],
)
drift_input_length_ks_pvalue = Gauge(
    "drift_input_length_ks_pvalue",
    "KS p-value for input length drift",
)
drift_oov_rate = Gauge("drift_oov_rate", "OOV token rate")
drift_pred_class = Gauge(
    "drift_pred_class",
    "Prediction class distribution",
    ["tone"],
)
drift_confidence_mean = Gauge("drift_confidence_mean", "Mean prediction confidence")
