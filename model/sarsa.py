from model.abstractmodel import AbstractModel
from environment import core  # import Action, PlayerState
from loguru import logger


class SARSA(AbstractModel):
    def __init__(self):
        super().__init__()
        self.Q = dict()

    def q(self, state):
        pass

    def predict(self, state):
        pass
