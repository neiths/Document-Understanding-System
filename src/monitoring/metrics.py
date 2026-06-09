import time
import threading
from typing import Dict, Any, Optional
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
from prometheus_client.core import GaugeMetricFamily, CounterMetricFamily
import logging

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Central metrics collection service"""

    def __init__(self):
        # Create registry
        self.registry = CollectorRegistry()

        # Initialize metrics
        self._init_metrics()

        # Storage for dynamic metrics
        self._custom_metrics = {}

        # Thread-safe updates
        self._lock = threading.Lock()

    def _init_metrics(self):
        """Initialize Prometheus metrics"""
        # Request metrics
        self.request_total = Counter(
            'document_requests_total',
            'Total number of document processing requests',
            ['pipeline', 'document_type', 'status', 'model_version'],
            registry=self.registry
        )

        self.request_duration = Histogram(
            'document_processing_seconds',
            'Time spent processing documents',
            ['pipeline', 'document_type'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
            registry=self.registry
        )

        self.active_requests = Gauge(
            'active_requests',
            'Number of active requests',
            ['pipeline'],
            registry=self.registry
        )

        # Model metrics
        self.model_confidence = Gauge(
            'model_confidence',
            'Model confidence scores',
            ['pipeline', 'model_name', 'document_type'],
            registry=self.registry
        )

        self.model_errors = Counter(
            'model_errors_total',
            'Total model errors',
            ['pipeline', 'model_name', 'error_type'],
            registry=self.registry
        )

        # Pipeline metrics
        self.pipeline_requests = Counter(
            'pipeline_requests_total',
            'Total requests per pipeline',
            ['pipeline'],
            registry=self.registry
        )

        self.pipeline_latency = Histogram(
            'pipeline_latency_seconds',
            'Pipeline processing latency',
            ['pipeline'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            registry=self.registry
        )

        # Resource metrics
        self.gpu_utilization = Gauge(
            'gpu_utilization',
            'GPU utilization percentage',
            ['pipeline', 'gpu_id'],
            registry=self.registry
        )

        self.memory_usage = Gauge(
            'memory_usage_bytes',
            'Memory usage in bytes',
            ['pipeline'],
            registry=self.registry
        )

        # Custom metrics
        self.custom_metrics = {}

    def record_request(
        self,
        pipeline: str,
        document_type: str,
        status: str,
        model_version: str = "latest",
        duration: float = None
    ):
        """Record a request"""
        with self._lock:
            self.request_total.labels(
                pipeline=pipeline,
                document_type=document_type,
                status=status,
                model_version=model_version
            ).inc()

            if duration:
                self.request_duration.labels(
                    pipeline=pipeline,
                    document_type=document_type
                ).observe(duration)

    def record_pipeline_request(
        self,
        pipeline: str,
        duration: float,
        success: bool = True
    ):
        """Record pipeline-specific request"""
        with self._lock:
            self.pipeline_requests.labels(pipeline=pipeline).inc()
            self.pipeline_latency.labels(pipeline=pipeline).observe(duration)

    def update_active_requests(self, pipeline: str, delta: int):
        """Update active request count"""
        with self._lock:
            self.active_requests.labels(pipeline=pipeline).set(
                self.active_requests.labels(pipeline=pipeline)._value._value + delta
            )

    def record_model_confidence(
        self,
        pipeline: str,
        model_name: str,
        document_type: str,
        confidence: float
    ):
        """Record model confidence"""
        with self._lock:
            self.model_confidence.labels(
                pipeline=pipeline,
                model_name=model_name,
                document_type=document_type
            ).set(confidence)

    def record_model_error(
        self,
        pipeline: str,
        model_name: str,
        error_type: str
    ):
        """Record model error"""
        with self._lock:
            self.model_errors.labels(
                pipeline=pipeline,
                model_name=model_name,
                error_type=error_type
            ).inc()

    def update_gpu_utilization(self, pipeline: str, gpu_id: str, utilization: float):
        """Update GPU utilization"""
        with self._lock:
            self.gpu_utilization.labels(
                pipeline=pipeline,
                gpu_id=gpu_id
            ).set(utilization)

    def update_memory_usage(self, pipeline: str, bytes_used: int):
        """Update memory usage"""
        with self._lock:
            self.memory_usage.labels(pipeline=pipeline).set(bytes_used)

    def add_custom_metric(
        self,
        name: str,
        documentation: str,
        labelnames: list = None,
        metric_type: str = "gauge"
    ):
        """Add a custom metric"""
        with self._lock:
            if name not in self.custom_metrics:
                if metric_type == "counter":
                    self.custom_metrics[name] = Counter(
                        name,
                        documentation,
                        labelnames=labelnames,
                        registry=self.registry
                    )
                elif metric_type == "gauge":
                    self.custom_metrics[name] = Gauge(
                        name,
                        documentation,
                        labelnames=labelnames,
                        registry=self.registry
                    )

    def record_custom_metric(
        self,
        name: str,
        value: float,
        labels: Dict[str, str] = None
    ):
        """Record a custom metric value"""
        with self._lock:
            if name in self.custom_metrics:
                if labels:
                    self.custom_metrics[name].labels(**labels).set(value)
                else:
                    self.custom_metrics[name].set(value)

    def get_metrics(self) -> str:
        """Get all metrics in Prometheus format"""
        return generate_latest(self.registry).decode('utf-8')

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of current metrics"""
        summary = {
            "total_requests": self.request_total._value._value,
            "active_requests": {
                "ocr": self.active_requests.labels(pipeline="ocr")._value._value,
                "vlm": self.active_requests.labels(pipeline="vlm")._value._value
            },
            "average_confidence": None,
            "error_rate": None
        }

        # Calculate average confidence
        with self._lock:
            confidences = []
            for metric in self.model_confidelity.collect():
                for sample in metric.samples:
                    if sample.name == "model_confidence":
                        confidences.append(sample.value)

            if confidences:
                summary["average_confidence"] = sum(confidences) / len(confidences)

        # Calculate error rate
        total_requests = self.request_total._value._value
        total_errors = self.model_errors._value._value

        if total_requests > 0:
            summary["error_rate"] = total_errors / total_requests

        return summary

    def reset_metrics(self):
        """Reset all metrics"""
        with self._lock:
            for metric in self.registry._collector_to_names:
                if hasattr(metric, '_value'):
                    metric._value._value = 0


class OpenTelemetryTracer:
    """OpenTelemetry tracing integration"""

    def __init__(self, service_name: str = "document-understanding-system"):
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.exporter.jaeger.thrift import JaegerExporter

            # Configure tracer
            self.tracer_provider = TracerProvider()
            trace.set_tracer_provider(self.tracer_provider)

            # Configure exporter
            jaeger_exporter = JaegerExporter(
                agent_host_name="localhost",
                agent_port=6831
            )

            # Add exporter
            self.tracer_provider.add_span_processor(
                trace.BatchSpanProcessor(jaeger_exporter)
            )

            self.tracer = trace.get_tracer(service_name)
            self.enabled = True

        except Exception as e:
            logger.warning(f"Failed to initialize OpenTelemetry: {str(e)}")
            self.enabled = False
            self.tracer = None

    def start_span(self, name: str, attributes: Dict[str, Any] = None):
        """Start a new span"""
        if not self.enabled:
            return None

        return self.tracer.start_span(
            name,
            attributes=attributes or {}
        )

    def end_span(self, span):
        """End a span"""
        if span and self.enabled:
            span.end()

    def create_span(
        self,
        name: str,
        func,
        attributes: Dict[str, Any] = None
    ):
        """Create and manage a span"""
        if not self.enabled:
            return func()

        with self.start_span(name, attributes):
            return func()