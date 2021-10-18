import numpy as np
from loguru import logger
from enum import IntEnum
import itertools
from typing import Union, List


class Action(IntEnum):
    '''Available actions for players.
    '''
    HIT = 0
    STICK = 1


class GameStatus(IntEnum):
    '''Possible statuses for an episode of Blackjack game.
    '''
    END = 0  #
    PLAYING = 1
    NATURAL = 2  # Someone has 21 immediately (Ace + 10-card) after first dealing
    DRAW = 3  # Some player and dealer have natural at same time


class PlayerStatus(IntEnum):
    '''Possible statuses for a player in an episode of Blackjack game.
    '''
    WIN = 0
    LOSE = 1
    LOSE_BUST = 2
    PLAYING = 3
    STICK = 4
    NATURAL = 5


class UsableAce(IntEnum):
    NO_USABLE = 0
    USABLE = 1


class Card(object):
    """A basic Card class with sum method for the convenience of Blackjack game.
    """
    suits = ["spades", "hearts", "diamonds", "clubs"]
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    # one_deck = np.array(list(itertools.product(values, suits)),
    #                          dtype=[('value', int), ('suit', object)])  # numpy structured arrays
    # one_deck = [Card(v, s) for v, s in itertools.product(values, suits)]
    
    def __init__(self, v, s):
        self.value = v
        self.suit = s

    @classmethod
    def sum(cls, cards, discard=True):
        """
        When summing up cards, regard those greater than 10 as 10 (except usable_ace).
        """
        if not cards:
            return 0
        else:
            # return sum([11 if card.suit == 'usble_ace' else 10 if card.value > 10 else card.value for card in cards])
            return sum([card.value if card.value <= 10 else 10 for card in cards])

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            raise Exception("{} and {} objects are not comparable!".format(self.__class__.__name__,
                                                                           other.__class__.__name__))
        return self.value == other.value and self.suits == other.suits

    def __hash__(self):
        # Redefine the hash function, to hash the representation of object
        # So that in the Player.draw() method, we can check if the player holds aces.
        return hash(self.__repr__())

    def __repr__(self):
        return "({}, {})".format(self.value, self.suit)


class Decks(object):
    one_deck = [Card(v, s) for v, s in itertools.product(Card.values, Card.suits)]
    aces = set([Card(1, s) for s in Card.suits])
    def __init__(self, m: int):
        assert m >= 0, "Negative decks of cards is invalid!"
        self.m = m
        self.reset(self.m)

    def reset(self, m: int):
        if m == 0:
            self.cardset = Decks.one_deck.copy()
        else:
            self.cardset = np.repeat(Decks.one_deck, repeats=m, axis=0).tolist()
            # np.random.shuffle(self.cardset)  # No need to shuffle

    def dealCard(self, n=1) -> List[Card]:
        # Pop cards with/without replacement according to deck of cards (m)
        card = []
        for i in range(n):
            idx = np.random.choice(len(self.cardset))
            card.append(self.cardset[idx])
            if self.m > 0:
                self.cardset.pop(idx)

        return card

# dc = Decks(3)
# print(len(dc.cardset))
# dc.dealCard()
# print(len(dc.cardset))


class PlayerState(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.hand = []
        self.usable_ace = UsableAce.NO_USABLE
        self.update_points()

    def update_points(self):
        self.points = Card.sum(self.hand)
        if self.usable_ace == UsableAce.USABLE and self.points+10 <= 21:
            self.points += 10


class PlayerRecord(object):
    """
    !depreached!
    Player's record.
    The record is updated during continuous blackjack games.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.game_play = 0
        self.win = 0

    def __update(self, win=True):
        self.game_play += 1
        if win:
            self.win += 1


class Player(object):
    def __init__(self):
        self.name = ""
        self.state = PlayerState()
        # self.record = PlayerRecord()
        # self.policy = None
        self.reset()

    def draw(self, card: List[Card]):
        self.state.hand.extend(card)
        self.state.update_points()
        if self.points > 21:
            self.status = PlayerStatus.LOSE_BUST

        # TODO: usable ace: there is an Ace in self.state.hand and self.state.points+10 <= 21:
        if [hc for hc in self.state.hand if hc.value==1]: # Decks.aces & set(self.state.hand):  # If player has Ace
            hand_1st_ace = [hc for hc in self.state.hand if hc.value==1][0] # next(iter(Decks.aces & set(self.state.hand)))  # first ace in hand
            if self.points + 10 <= 21:
                # hand_ace.value = 11
                hand_1st_ace.suit = "usable_ace"
                self.state.usable_ace = UsableAce.USABLE
            else:
                hand_1st_ace.suit = "no_usable_ace"
                self.state.usable_ace = UsableAce.NO_USABLE
            self.state.update_points()

        if self.points + 10 == 21 and len(self.state.hand) == 2 and self.state.usable_ace == UsableAce.USABLE:
            # hand_1st_ace.value = 11
            self.state.update_points()
            self.status = PlayerStatus.NATURAL

    @property
    def points(self):
        return self.state.points

    def reset(self):
        """Reset a player, reset his state
        """
        self.state.reset()
        self.status = PlayerStatus.PLAYING


class Gambler(Player):
    def __init__(self, model=None):
        super(Gambler, self).__init__()
        # self.policy = model


class Dealer(Player):
    def __init__(self):
        super(Dealer, self).__init__()
        # self.policy = DealerPolicy()  # Dealer has its own policy
