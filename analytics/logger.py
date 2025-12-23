"""Enhanced Analytics Logger Module - Production Ready"""

import logging
import json
from datetime import datetime
from threading import Lock
from typing import Dict, Any, Optional, List, Iterable
from collections import deque
import os

# Operational logging (distinct from analytics events)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AnalyticsLogger:
    """
    Thread-safe analytics event logger with optional persistence.
    Designed for trading systems: events are immutable once logged.
    """
Analytics Logger Module
"""
import logging
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class AnalyticsLogger:
    """Logger for analytics events and metrics"""
    
    def __init__(self):
        """Initialize the analytics logger"""
        self.events = []
        logger.info("AnalyticsLogger initialized")
    
    def log_event(self, event_type, data=None):
        """
        Log an analytics event
        
        Args:
            event_type: Type of event (e.g., 'trade', 'signal', 'error')
            data: Event data dictionary
        """
        event = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': event_type,
            'data': data or {}
        }
        self.events.append(event)
        logger.info(f"Event logged: {event_type} - {data}")
    
    def log_trade(self, symbol, action, quantity, price):
        """Log a trade event"""
        self.log_event('trade', {
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'price': price
        })
    
    def log_signal(self, symbol, signal_type, strength):
        """Log a trading signal"""
        self.log_event('signal', {
            'symbol': symbol,
            'signal_type': signal_type,
            'strength': strength
        })
    
    def log_error(self, error_type, message, details=None):
        """Log an error"""
        self.log_event('error', {
            'error_type': error_type,
            'message': message,
            'details': details
        })
    
    def get_events(self, event_type=None, limit=100):
        """
        Get logged events
        
        Args:
            event_type: Filter by event type (optional)
            limit: Maximum number of events to return
        """
        if event_type:
            filtered = [e for e in self.events if e['event_type'] == event_type]
            return filtered[-limit:]
        return self.events[-limit:]
    
    def clear_events(self):
        """Clear all logged events"""
        self.events = []
        logger.info("Events cleared")


# Global analytics logger instance
analytics_logger = AnalyticsLogger()
    """

    def __init__(
        self,
        max_events: int = 10_000,
        persist_path: Optional[str] = None,
        persist_batch_size: int = 50
    ):
        """
        Initialize logger.

        Args:
            max_events: Maximum in-memory events (circular buffer if exceeded)
            persist_path: If provided, append events to JSONL file
            persist_batch_size: Flush to disk every N events
        """
        self._lock = Lock()
        self._events: deque[Dict[str, Any]] = deque(maxlen=max_events)
        self._persist_path = persist_path
        self._persist_batch_size = persist_batch_size
        self._persist_counter = 0
        self._file_handle: Optional[Any] = None

        if persist_path:
            os.makedirs(os.path.dirname(persist_path), exist_ok=True) if os.path.dirname(persist_path) else None
            self._file_handle = open(persist_path, 'a', encoding='utf-8')

        logger.info(f"AnalyticsLogger initialized - persist={'yes' if persist_path else 'no'}")

    def _persist_event(self, event: Dict[str, Any]) -> None:
        if not self._file_handle:
            return
        json.dump(event, self._file_handle)
        self._file_handle.write('\n')
        self._persist_counter += 1
        if self._persist_counter >= self._persist_batch_size:
            self._file_handle.flush()
            self._persist_counter = 0

    def log_event(self, event_type: str, data: Optional[Dict[str, Any]] = None) -> None:
        """Core method - log arbitrary analytics event"""
        event = {
            'timestamp': datetime.utcnow().isoformat(timespec='milliseconds') + 'Z',
            'event_type': event_type,
            'data': data or {}
        }

        with self._lock:
            self._events.append(event)
            self._persist_event(event)

        logger.debug(f"Analytics event: {event_type}")

    def log_trade(self, symbol: str, action: str, quantity: float, price: float, **extra) -> None:
        """Log executed trade"""
        self.log_event('trade', {
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'price': price,
            **extra
        })

    def log_signal(self, symbol: str, signal_type: str, strength: float, **extra) -> None:
        """Log generated signal"""
        self.log_event('signal', {
            'symbol': symbol,
            'signal_type': signal_type,
            'strength': strength,
            **extra
        })

    def log_error(self, error_type: str, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Log error event"""
        self.log_event('error', {
            'error_type': error_type,
            'message': message,
            'details': details or {}
        })

    def get_events(
        self,
        event_type: Optional[str] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """Retrieve recent events (thread-safe snapshot)"""
        with self._lock:
            events = list(self._events)

        if event_type:
            events = [e for e in events if e['event_type'] == event_type]

        return events[-limit:]

    def export_json(self) -> str:
        """Export current in-memory events as JSON string"""
        with self._lock:
            return json.dumps(list(self._events), indent=2)

    def clear_events(self) -> None:
        """Clear in-memory buffer (persisted events remain on disk)"""
        with self._lock:
            self._events.clear()
        logger.info("In-memory analytics events cleared")

    def close(self) -> None:
        """Flush and close file handle (call on shutdown)"""
        if self._file_handle:
            self._file_handle.flush()
            self._file_handle.close()
            self._file_handle = None


# Recommended usage: dependency injection instead of global
# For simple scripts, controlled singleton:
class _AnalyticsLoggerSingleton:
    _instance: Optional[AnalyticsLogger] = None

    @classmethod
    def get_instance(
        cls,
        persist_path: Optional[str] = "logs/analytics/events.jsonl",
        **kwargs
    ) -> AnalyticsLogger:
        if cls._instance is None:
            cls._instance = AnalyticsLogger(persist_path=persist_path, **kwargs)
        return cls._instance
analytics_logger(persist_path="data/analytics_$(date +%Y%m%d).jsonl")

# Alias for convenience in small projects
analytics_logger = _AnalyticsLoggerSingleton.get_instance
