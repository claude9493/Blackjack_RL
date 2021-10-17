import random
from collections import deque
import numpy as np
from environment import core, blackjack
from model.abstract_model import AbstractModel
from loguru import logger


class MonteCarloControl(AbstractModel):

    def __init__(self, game, **kwargs):
        super().__init__(game)
        self.exploration_rate = kwargs.get("exploration_rate", 0.10)
        self.Q = np.zeros((10, 10, 2, 2))  # PlayerSum, DealerShow, UsableAce, Action

    def save(self, filename):
        np.savez(filename, **{"Q": self.Q})

    def load(self, filename):
        loader = np.load(file=filename)
        self.Q = loader.get("Q")

    def updateQ(self):
        pass

    def train(self, stop_at_convergence=True, **kwargs):
        # discount = kwargs.get("discount", 1)  # 0.90)
        self.exploration_rate = kwargs.get("exploration_rate", 0.10)
        # GLIE Monte Carlo Control
        exploration_decay = kwargs.get("exploration_decay", 0.995)  # % reduction per step = 100 - exploration decay
        learning_rate = kwargs.get("learning_rate", 0.10)
        episodes = max(kwargs.get("episodes", 1000), 1)

        check_convergence_every = kwargs.get("check_convergence_every", self.check_convergence_every)
        report_every = kwargs.get("report_every", int(episodes/10))

        last_100_updates = deque(maxlen=100)  # np.zeros((100, 1))

        for episode in range(1, episodes + 1):
            status, rewards, player_trajectory = self.game.play(self)
            for reward, trajectory in zip(rewards, player_trajectory):
                for obs, action in trajectory:
                    obs_index = tuple(np.array(obs) - np.array([12, 1, 0]))
                    self.Q[obs_index][action] += learning_rate * (reward - self.Q[obs_index][action])

                    last_100_updates.append(learning_rate * (reward - self.Q[obs_index][action]))

            if episode % report_every == 0:
                logger.info("Episode {:4d}/{}: epsilon = {:.4e}| {:3d} unseen behaviors".format(episode, episodes, self.exploration_rate, np.count_nonzero(self.Q == 0)))

            if episode % check_convergence_every == 0:
                if np.mean(np.abs(last_100_updates)) <= 1e-3 and stop_at_convergence==True:
                    logger.info("CONVERGENCE: Average updates of last 100 iterates is smaller than 1e-3.")
                    break

            self.exploration_rate *= exploration_decay

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


if __name__ == '__main__':
    logger.disable("environment.blackjack")
    table = blackjack.Table(m=0, n=2)  # Infinity deck of cards and 2 players
    game = blackjack.Blackjack(table=table)
    model = MonteCarloControl(game=game)
    model.train(episodes=5000000, exploration_decay=1-1e-5, learning_rate=0.5)
    print(model.Q[:, :, 0, 0])
    print(model.Q[:, :, 0, 1])
    model.save("MC_Control_5m.npz")

