from abc import ABC, abstractmethod
from environment import core


class AbstractModel(ABC):
    def __init__(self):
        pass

    def load(self, filename):
        """ Load model from file. """
        pass

    def save(self, filename):
        """ Save model to file. """
        pass

    def train(self, stop_at_convergence=False, **kwargs):
        """ Train model. """
        pass

    @abstractmethod
    def q(self, state):
        pass

    @abstractmethod
    def predict(self, state):
        pass


class TestModel(AbstractModel):
    """
    A testing model, gambler sticks on any sum of 20 or greater, and hits otherwise.
    # TODO: make the model trainable! Input gambler index and record the (state, action, reward) for them.
    """
    def q(self, state):
        pass

    def predict(self, state: list):
        if state[0] < 20:
            return core.Action.HIT
        return core.Action.STICK

