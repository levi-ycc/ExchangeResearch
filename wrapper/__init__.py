from gym import Wrapper
import numpy as np

class MyWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        
    def step(self, action):
        return self.env.step(action)