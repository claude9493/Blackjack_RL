import itertools
import pytest

from environment import blackjack
from model.abstractmodel import TestModel

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


@pytest.fixture
def model(request):
    return models[request.param]


@pytest.mark.parametrize("game", [(0, 2)], indirect=True, ids=str)
@pytest.mark.parametrize("model", ["Test"], indirect=True)
def test_two(game, model):
    game.play(model)


@pytest.mark.parametrize("game", itertools.product(m, n), indirect=True, ids=str)
@pytest.mark.parametrize("model", ["Test"], indirect=True)
def test_combinations(game, model):
    print([game.table.m, game.table.n])
    game.play(model)
