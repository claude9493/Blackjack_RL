import unittest
from loguru import logger
from environment import blackjack
from model.abstract_model import TestModel
from model.dealer_policy import DealerPolicy


def setup(m=2, n=0):
    table = blackjack.Table(m, n)
    # policies = [TestModel() for i in range(n-1)]
    # policies.append(DealerPolicy())
    # table.set_policy(policies)
    game = blackjack.Blackjack(table=table)
    return game


class MyTestCase(unittest.TestCase):
    def test_2_players(self):
        game = setup(0, 2)
        print(game.table.players)
        game.play(TestModel())

    def test_3_players(self):
        game = setup(m=0, n=3)
        print(game.table.players)
        game.play(TestModel())

    def test_combination(self):
        model = TestModel()
        for m in [6, 3, 1]:
            for n in [3, 4, 6]:
                logger.info("Now goes the combination m = {}, n = {}.".format(m, n))
                game = setup(m, n)
                game.play(model)

if __name__ == '__main__':
    unittest.main()
