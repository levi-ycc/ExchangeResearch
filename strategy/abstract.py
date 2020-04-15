from abc import ABC, abstractmethod

class StrategyAbstract(ABC):
    
    @abstractmethod
    def __init__(self, journal_df):
        return NotImplemented
    
    @abstractmethod
    def parse_data(self):
        return NotImplemented
    
    @abstractmethod
    def generate_signals(self):
        return NotImplemented
    
    @abstractmethod
    def run_backtest(self):
        return NotImplemented
    
    @abstractmethod
    def stats(self):
        return NotImplemented
    
    @abstractmethod
    def out_journal(self):
        return NotImplemented
    
    