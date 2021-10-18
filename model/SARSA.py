import random
import numpy as np
from collections import deque
from environment import core, blackjack
from model.abstract_model import AbstractModel, q_obs_index
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

    def train(self, stop_at_convergence=True, **kwargs):
        self.exploration_rate = kwargs.get("exploration_rate", 0.10)  # $\epsilon$
        self.gamma = kwargs.get("gamma", 1)  # $\gamma$
        exploration_decay = kwargs.get("exploration_decay", 0.995)  # $epsilon_{t+1} = epsilon_{t} * exploration_decay$
        eps_min = kwargs.get("eps_min", 0.05)  # min epsilon
        episodes = max(kwargs.get("episodes", 1000), 1)
        learning_rate = kwargs.get("learning_rate", 0.01)
        check_convergence_every = kwargs.get("check_convergence_every", self.check_convergence_every)
        report_every = kwargs.get("report_every", int(episodes/10))

        last_100_updates = deque(maxlen=100)  # np.zeros((100, 1))
        alpha = learning_rate

        for episode in range(1, episodes + 1):
            self.exploration_rate = min(1.0 / episode, eps_min)
            state_action_cnt = np.zeros((10, 10, 2))
            status, rewards, player_trajectory = self.game.play(self)
            for reward, trajectory in zip(rewards, player_trajectory):
                for j, (obs, action) in enumerate(trajectory):
                    obs_index = q_obs_index(obs)
                    state_action_cnt[obs_index] += 1
                    # alpha = 1/state_action_cnt[obs_index]
                    # Gt = np.sum(reward * np.power(self.gamma, range(len(trajectory) - j)))
                    if j < len(trajectory) - 1:
                        # action value function of next observation and action pair $Q(S^\prime, A^\prime)
                        Q_next = self.Q[q_obs_index(trajectory[j+1][0])][trajectory[j+1][1]]
                    else:
                        Q_next = 0
                    # =========================== Update Step =======================================
                    update = alpha*(reward + self.gamma * Q_next - self.Q[obs_index][action])
                    self.Q[obs_index][action] += update
                    # ===============================================================================
                    # last_100_updates.append(learning_rate * (reward - self.Q[obs_index][action]))
                    last_100_updates.append(update)

            if episode % report_every == 0:
                logger.info(
                    "Episode {:4d}/{}: epsilon = {:.4e} | {:3d} unseen pairs | average last 100 updates = {:.5f}"
                    .format(episode, episodes, self.exploration_rate, np.count_nonzero(self.Q == 0),
                            np.mean(np.abs(last_100_updates)))
                )

            if episode % check_convergence_every == 0:
                if np.mean(np.abs(last_100_updates)) <= 1e-3 and stop_at_convergence:
                    logger.info("CONVERGENCE: Average updates of last 100 iterates is smaller than 1e-3.")
                    break

            # self.exploration_rate = min(self.exploration_rate*exploration_decay, eps_min)


    def q(self, obs: tuple):
        # obs_index = tuple(np.array(obs) - np.array([12, 1, 0]))  # "PlayerSum", "DealerShow", "UsableAce"
        return self.Q[q_obs_index(obs)]

    def predict(self, state: tuple):
        if np.random.random() < self.exploration_rate:
            action = random.choice(self.game.actions)
            return action
        else:
            q = self.q(state)
            actions = np.nonzero(q == np.max(q))[0]
            return random.choice(actions)

