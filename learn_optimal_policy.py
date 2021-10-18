from itertools import product
import pytest

from loguru import logger
from environment import blackjack
from model.abstract_model import TestModel
from model import model_visualization
from model.MonteCarloControl import MonteCarloControl
from model.SARSA import SARSA
from model.QLearning import QLearning

# logger.add("test.log", enqueue=True)

name_record = "./record/npz/{}_m{}_n{}_e{:.0e}.npz"
name_figure = "./record/{}_m{}_n{}_e{:.0e}.png"

m = [6, 3, 1]
n = [3, 4, 6]
episodes = 500000

models = {
    "Test": TestModel,
    "MC": MonteCarloControl,
    "TD": SARSA,
    "QL": QLearning
}


@pytest.fixture
def game(request):  # Does the parameter of fixture function must be named request?
    table = blackjack.Table(*request.param)
    game = blackjack.Blackjack(table=table)
    return game


# Question 1, 2
@pytest.mark.parametrize("game, model", product([(0, 2)], ["MC"]), indirect=["game"], ids=str)
def test_q1(game, model):
    logger.disable("environment.blackjack")
    policy = models[model](game)
    policy.train(episodes=episodes, exploration_rate=1.0, exploration_decay=1-1e-5, gamma=1.0)
    policy.save(name_record.format(model, game.table.m, game.table.n, episodes))
    model_visualization.draw_policy(policy.Q, filename=name_figure.format(model, game.table.m, game.table.n, episodes))


# Question 3
@pytest.mark.parametrize("game, model", product(product(m, n), ["MC"]), indirect=["game"], ids=str)
def test_combinations(game, model):
    logger.disable("environment.blackjack")
    policy = models[model](game)
    policy.train(episodes=episodes, exploration_rate=1, exploration_decay=1-1e-5)  # , learning_rate = 0.5)
    policy.save(name_record.format(model, game.table.m, game.table.n, episodes))
    model_visualization.draw_policy(policy.Q, filename=name_figure.format(model, game.table.m, game.table.n, episodes))

