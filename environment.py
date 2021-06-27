from abc import ABC, abstractmethod

class Environment(ABC):
    
    @abstractmethod
    def make_move(self):
        pass 

    @abstractmethod
    def get_reward(self):
        pass 

    @abstractmethod
    def display(self):
        pass 

    @abstractmethod
    def actions(self):
        pass 

    @abstractmethod
    def finished(self):
        pass 

    @abstractmethod
    def state_tensor(self):
        pass 