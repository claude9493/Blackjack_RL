from itertools import product
import pytest

from loguru import logger
from environment import blackjack
from model.abstractmodel import TestModel

logger.add("test.log", enqueue=True)

m = [6, 3, 1]
n = [3, 4, 6]

models = {
    "Test": TestModel(),
    "MC": None,
    "TD": None,
    "QL": None
}


@pytest.fixture
def game(request):  # Does the parameter of fixture function must be named request?
    table = blackjack.Table(*request.param)
    game = blackjack.Blackjack(table=table)
    return game


@pytest.mark.parametrize("game, model", [((0, 2), "Test")], indirect=["game"], ids=str)
def test_two(game, model):
    game.play(models[model])


@pytest.mark.parametrize("game, model", product(product(m, n), ["Test"]), indirect=["game"], ids=str)
def test_combinations(game, model):
    print([game.table.m, game.table.n])
    game.play(models[model])
