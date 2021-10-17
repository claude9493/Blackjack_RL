import random
import numpy as np
from environment import core, blackjack
from model.abstract_model import AbstractModel
from loguru import logger


class SARSA(AbstractModel):
    def __init__(self, game, **kwargs):
        super().__init__(game)
        self.exploration_rate = kwargs.get("exploration_rate", 0.10)
        self.Q = np.zeros((10, 10, 2, 2))  # PlayerSum, DealerShow, UsableAce, Action

    def save(self, filename):
        np.savez(filename, **{"Q": self.Q})

    def load(self, filename):
        loader = np.load(file=filename)
        self.Q = loader.get("Q")

    def train(self):
        pass

    def q(self, obs: tuple):
        obs_index = tuple(np.array(obs) - np.array([12, 1, 0]))  # "PlayerSum", "DealerShow", "UsableAce"
        return self.Q[obs_index]

    def predict(self, state: tuple):
        if np.random.random() < self.exploration_rate:
            action = random.choice(self.game.actions)
            return action
        else:
            q = self.q(state)
            actions = np.nonzero(q == np.max(q))[0]
            return random.choice(actions)

