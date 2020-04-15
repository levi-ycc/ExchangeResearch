import numpy as np
from abc import abstractmethod
from abc import ABC

class AgentAbstract(ABC):
    
    @abstractmethod
    def __init__(self, config, save_name=None):
        return NotImplemented
        
    @abstractmethod
    def fit(self):
        return NotImplemented
        
    @abstractmethod
    def save_model(self, save_name=None):
        return NotImplemented
        
    @abstractmethod
    def load_model(self):
        return NotImplemented

    @abstractmethod
    def test(self):
        return NotImplemented

    @abstractmethod
    def init_env(self):
        return NotImplemented
    
    @abstractmethod
    def create_agent(self):   
        return NotImplemented
            
    @abstractmethod
    def predict(self, df):
        return NotImplemented
        
    @abstractmethod
    def backtest(self):
        return NotImplemented

