import asyncio
import json
import logging
import time
from datetime import datetime
from functools import wraps
from logging.handlers import RotatingFileHandler
from typing import Any, Callable, Dict, Optional, Type
import sys
import traceback

class CircuitBreaker:
    """
    A circuit breaker pattern implementation to prevent cascading failures.
    States:
      - CLOSED: Requests pass through freely.
      - OPEN: Requests fail immediately (cooldown).
      - HALF_OPEN: One request allowed to probe if the service is recovered.
    """
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"

    def __init__(self, failure_threshold: int = 5, cooldown_seconds: int = 30, name: str = "CircuitBreaker"):
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self.name = name
        self.state = self.CLOSED
        self.failures = 0
        self.last_failure_time = 0.0

    async def __aenter__(self):
        if self.state == self.OPEN:
            if time.time() - self.last_failure_time > self.cooldown_seconds:
                self.state = self.HALF_OPEN
                logging.info(f"[{self.name}] Circuit half-open, probing...")
            else:
                raise Exception(f"[{self.name}] Circuit is OPEN. Request rejected.")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.failures += 1
            self.last_failure_time = time.time()
            if self.failures >= self.failure_threshold and self.state != self.OPEN:
                self.state = self.OPEN
                logging.warning(f"[{self.name}] Circuit tripped to OPEN due to {exc_type.__name__}.")
        else:
            if self.state == self.HALF_OPEN:
                self.state = self.CLOSED
                logging.info(f"[{self.name}] Circuit recovered, closed.")
            self.failures = 0
        return False

    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            async with self:
                return await func(*args, **kwargs)
        return wrapper


def RetryWithBackoff(max_retries: int = 3, base_delay: float = 0.5, max_delay: float = 30.0, retryable_exceptions: tuple = (Exception,)):
    """
    Async decorator that retries a function with exponential backoff and jitter.
    """
    import random
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            delay = base_delay
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except retryable_exceptions as e:
                    if attempt == max_retries:
                        logging.error(f"[{func.__name__}] Failed after {max_retries} retries: {str(e)}")
                        raise e
                    
                    jitter = random.uniform(0, 0.1 * delay)
                    sleep_time = min(delay + jitter, max_delay)
                    logging.warning(f"[{func.__name__}] Attempt {attempt + 1} failed ({str(e)}). Retrying in {sleep_time:.2f}s...")
                    
                    await asyncio.sleep(sleep_time)
                    delay *= 2
        return wrapper
    return decorator


class ServiceHealthTracker:
    """
    Singleton tracker for health status of different backend subsystems.
    """
    _instance = None
    
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ServiceHealthTracker, cls).__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self):
        self.services = {
            "hailo": self._default_status(),
            "rtsp_cam_0": self._default_status(),
            "go2rtc": self._default_status(),
            "database": self._default_status(),
            "notification": self._default_status()
        }

    def _default_status(self) -> Dict[str, Any]:
        return {
            "status": self.HEALTHY,
            "last_error": None,
            "last_success_time": time.time(),
            "consecutive_failures": 0
        }

    def record_success(self, service_name: str):
        if service_name not in self.services:
            self.services[service_name] = self._default_status()
            
        s = self.services[service_name]
        s["status"] = self.HEALTHY
        s["last_success_time"] = time.time()
        s["consecutive_failures"] = 0
        s["last_error"] = None

    def record_failure(self, service_name: str, error: str):
        if service_name not in self.services:
            self.services[service_name] = self._default_status()
            
        s = self.services[service_name]
        s["consecutive_failures"] += 1
        s["last_error"] = str(error)
        
        if s["consecutive_failures"] >= 3:
            s["status"] = self.FAILED
        else:
            s["status"] = self.DEGRADED

    def get_system_health_report(self) -> Dict[str, Any]:
        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "services": self.services
        }


class StructuredJsonFormatter(logging.Formatter):
    """
    Format logs as JSON. Include timestamp, level, message, and optional extra fields.
    """
    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger_name": record.name,
            "message": record.getMessage(),
        }
        
        # Add standard optional fields if present in extra
        optional_fields = ["camera_id", "event_id", "service", "error_type", "latency_ms"]
        for field in optional_fields:
            if hasattr(record, field):
                log_obj[field] = getattr(record, field)
                
        if record.exc_info:
            log_obj["traceback"] = self.formatException(record.exc_info)
            
        return json.dumps(log_obj)

def setup_structured_logging():
    """
    Configures Python logging to output structured JSON to both stdout and a rotating file.
    """
    import os
    
    # Create logs directory
    os.makedirs("storage/logs", exist_ok=True)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        
    formatter = StructuredJsonFormatter()
    
    # Rotating File Handler (10MB, 5 backups)
    file_handler = RotatingFileHandler(
        "storage/logs/edge_cctv.log", maxBytes=10*1024*1024, backupCount=5
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Stdout handler
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    
    logging.info("Structured logging initialized.")
