"""Monitoring and metrics setup"""

from prometheus_client import Counter, Histogram, Gauge
from fastapi import FastAPI, Request
import time
import logging

logger = logging.getLogger(__name__)

# Metrics
request_count = Counter(
    'codemind_requests_total',
    'Total request count',
    ['method', 'endpoint', 'status']
)

request_duration = Histogram(
    'codemind_request_duration_seconds',
    'Request duration in seconds',
    ['method', 'endpoint']
)

agent_execution_time = Histogram(
    'codemind_agent_execution_seconds',
    'Agent execution time in seconds',
    ['agent_type']
)

llm_token_usage = Counter(
    'codemind_llm_tokens_total',
    'Total LLM tokens used',
    ['model', 'type']
)

active_requests = Gauge(
    'codemind_active_requests',
    'Number of active requests'
)


def setup_monitoring(app: FastAPI):
    """Setup monitoring middleware"""
    
    @app.middleware("http")
    async def monitor_requests(request: Request, call_next):
        """Monitor all HTTP requests"""
        active_requests.inc()
        start_time = time.time()
        
        try:
            response = await call_next(request)
            duration = time.time() - start_time
            
            # Record metrics
            request_count.labels(
                method=request.method,
                endpoint=request.url.path,
                status=response.status_code
            ).inc()
            
            request_duration.labels(
                method=request.method,
                endpoint=request.url.path
            ).observe(duration)
            
            return response
        finally:
            active_requests.dec()
    
    logger.info("Monitoring middleware configured")