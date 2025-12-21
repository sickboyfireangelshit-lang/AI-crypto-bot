"""Analytics Logger Module"""
import logging
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
