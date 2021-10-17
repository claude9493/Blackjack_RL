from abc import ABC, abstractmethod
from environment import core


class AbstractModel(ABC):
    check_convergence_every = 5
    discount = 1

    def __init__(self, game):
        self.game = game
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
    Not trainable, because policy is fixed, the evaluation is given in MonteCarloLearning.py
    """

    def __init__(self):
        super().__init__(game=None)

    def q(self, state):
        pass

    def predict(self, state: tuple):
        if state[0] < 20:
            return core.Action.HIT
        return core.Action.STICK

