from model.abstract_model import AbstractModel
from environment import core


class DealerPolicy(AbstractModel):
    def __init__(self):
        super().__init__(game=None)

    def q(self, state):
        pass

    @classmethod
    def predict(cls, state):
        """
        The dealer sticks on any sum of 17 or greater, and hits otherwise.
        :param state: Current state of the dealer
        :type state: PlayerState
        :return: Action, stick or hit
        :rtype: Action
        """
        # logger.debug("Got observation: {}. Dealer's point: {}".format(state, core.Card.sum(state[:-1])))
        if state[0] >= 17:
            return core.Action.STICK
        else:
            return core.Action.HIT
