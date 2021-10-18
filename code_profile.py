from environment import blackjack
from loguru import logger
from model.MonteCarloControl import MonteCarloControl

if __name__ == '__main__':
    logger.disable("environment.blackjack")
    episodes = 1000
    table = blackjack.Table(m=0, n=2)
    game = blackjack.Blackjack(table=table)
    policy = MonteCarloControl(game)
    policy.train(episodes, exploration_decay=1-1e-5)

