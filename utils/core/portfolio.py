"""Portfolio Management Module"""
import logging

logger = logging.getLogger(__name__)

class Portfolio:
    """Portfolio manager for tracking positions and performance"""
    
    def __init__(self):
        self.positions = {}
        self.balance = 0.0
        logger.info("Portfolio initialized")
    
    def add_position(self, symbol, quantity, price):
        if symbol not in self.positions:
            self.positions[symbol] = {'quantity': 0, 'avg_price': 0}
        
        current = self.positions[symbol]
        total_quantity = current['quantity'] + quantity
        
        if total_quantity > 0:
            avg_price = (current['quantity'] * current['avg_price'] + quantity * price) / total_quantity
            self.positions[symbol] = {'quantity': total_quantity, 'avg_price': avg_price}
        else:
            del self.positions[symbol]
        
        logger.info(f"Position updated: {symbol} - {quantity} @ {price}")
    
    def get_position(self, symbol):
        return self.positions.get(symbol, {'quantity': 0, 'avg_price': 0})
    
    def get_all_positions(self):
        return self.positions
    
    def update_balance(self, amount):
        self.balance += amount
        logger.info(f"Balance updated: {self.balance}")
