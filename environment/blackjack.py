import numpy as np
from loguru import logger
from enum import IntEnum
from collections import Counter
from environment.core import *
from model.abstractmodel import  AbstractModel

# TODO: more logs.

class Table(object):
    """A Poker table, with decks of cards and a few players.
    1. Two ways to initialize a Table instance, by provide the deck of cards and number of players,
       or directly provide a list of players.
    """
    def __init__(self, m: int, n: int):
        """
        :param m: deck of cards, 0 indicates infinity, negative value is invalid.
        :type m: int
        :param n: number of players, must be greater or equal to 2.
        :type n: int
        """
        assert m >= 0, "Negative deck of cards is invalid!"
        self.m = m
        assert n >= 2, "There must be at least 2 players!"
        self.n = n

        self.players = []
        for i in range(n-1):
            self.players.append(Gambler())
        self.players.append(Dealer())
        self.decks = Decks(m)

        self.reset()

    """
    def __init__(self, m: int, players: List[Player]):
        assert m >= 0, "Negative deck of cards is invalid!"
        self.m = m

        self.n = len(players)
        assert n >= 2, "There must be at least 2 players!"

        # Make sure there is one and only one dealer.
        is_dealer = list(map(lambda x: isinstance(Dealer, x), player))
        n_dealer = sum(is_dealer)
        if n_dealer == 0:
            raise Exception("There is no dealer in the input player list!")
        elif n_dealer > 1:
            raise Exception("There is {} dealers in the player list!".format(n_dealer))

        self.players = players

        # Move dealer to the end of the player list
        if not is_dealer[-1]:
            self.players.append(self.players.pop(is_dealer.index(True)))

        self.reset()
    """

    def reset(self):
        self.__reset_decks(self.m)
        self.__reset_players()

    def __reset_decks(self, m: int):
        # Reset the decks on table by creating a new Decks object
        self.decks.reset(m)

    def __reset_players(self):
        # Reset players' states after one eposide
        for player in self.players:
            player.reset()

    def set_policy(self, policies: List[AbstractModel]):
        """
        Assign policy to players
        :param policies: List of players' policies
        :type policies: List(AbstractModel)
        """
        for player, policy in zip(self.players, policies):
            player.policy = policy

    def deal_all(self, n=2):
        """
        Deal n cards to every player.
        """
        for player in self.players:
            if player.state.hand:
                raise Exception("Player {} already has {} cards in hand!".format(player.name, len(player.state.hand)))
            player.draw(self.decks.dealCard(n))

    @property
    def players_status(self):
        """
        Summary of all players' status.
        """
        return [player.status for player in self.players]


class Blackjack:
    actions = [Action.HIT, Action.STICK]
    reward_win = 1
    reward_lose = -1
    reward_draw = 0
    
    def __init__(self, table: Table) -> object:
        self.table = table
        # self.reset()
        self.player_reward = np.zeros(shape=(self.table.n - 1, 1))  # Does dealer need reward?

    def reset(self):
        self.table.reset()  # reset decks and players
        self.table.deal_all(n=2)
        self.dealer_face_up = self.table.players[-1].state.hand[0]
        logger.info("Every player has drawn two cards.\n{}\nDealer's facing up card is {}.".format([p.state.hand for p in self.table.players], self.dealer_face_up))
        self.act = 0  # index of current actor among players list

    def step(self, action):
        player = self.table.players[self.act]
        self.__execute(action)
        status = self.__status()

        logger.debug("Player {}: action: {:6s}, status: {}. {}".format(self.act,
                                                                    Action(action).name,
                                                                    PlayerStatus(player.status).name,
                                                                    player.state.hand))

        if (action == Action.STICK or player.status == PlayerStatus.LOSE_BUST) \
                and status == GameStatus.PLAYING:
            self.act += 1

        observation = self.__observe()  # observation before next action, active player may have changed.
        return observation, status # reward, status

    def __execute(self, action):
        player = self.table.players[self.act]
        if action == Action.STICK:
            if player.status != PlayerStatus.LOSE_BUST:
                player.status = PlayerStatus.STICK
            else:
                pass
        if action == Action.HIT:
            player.draw(self.table.decks.dealCard(n=1))

    def __possible_actions(self):
        player = self.table.players[self.act]
        possible_actions = Blackjack.actions.copy()
        if player.state.points >= 21:
            possible_actions.remove(Action.HIT)
        return possible_actions

    def __status(self):
        """
        Game status! Not player status.
        Note that player status are updated immediately after draw card.
        If there are at least 2 players still playing, the game continues.
        """
        # If some player get natural, game will end with natural or draw
        # But if only dealer get immediately 21, the game will not end.
        if self.table.players_status[:-1].count(PlayerStatus.NATURAL) >= 1:
            if self.table.players_status[-1] == PlayerStatus.NATURAL:  # If draw happens
                return GameStatus.DRAW
            else:  # else still natural
                return GameStatus.NATURAL
        elif self.table.players_status[-1] == PlayerStatus.NATURAL:
            logger.info("Dealer got natural!")
            return GameStatus.NATURAL

        if self.table.players_status.count(PlayerStatus.PLAYING) == 0:
            # No player continues playing, settle and end.
            return GameStatus.END

        return GameStatus.PLAYING


    def __observe(self):
        """
        Players' observe is their own state plus the dealer's face up card.
        What the player could observe?
            1. Their own sum up points
            2. Whether or not holds a usable ace
            3. the dealer's one showing card
        """
        # self.dealer_face_up = self.table.players[-1].state.hand[0]  # dealer's first card is face-up
        # player_cards = self.table.players[self.act].state.hand
        return [self.table.players[self.act].points,
                self.dealer_face_up,
                self.table.players[self.act].state.usable_ace,]

    def __is_over(self, status=None):
        if not status:
            status = self.__status()
        if status != GameStatus.PLAYING:
            return True
        return False

    def __settlement(self, status):
        logger.debug("Now come to the settlement section. Player points: {}".format(np.array([p.state.points for p in self.table.players])))
        if status == GameStatus.END:
            players_points = np.array([p.state.points for p in self.table.players])
            players_points[players_points > 21] = 0  # Eliminate busted points !
            gambler_status_cnt = Counter(self.table.players_status[:-1])

        for p in self.table.players:
            # Player got natural wins, other lose
            if p.status not in (PlayerStatus.NATURAL, PlayerStatus.LOSE_BUST):
                p.status = PlayerStatus.LOSE

        if status == GameStatus.DRAW:
            logger.info("MIRACLE! MIRACLE! MIRACLE!")
            reward = np.zeros(shape=(self.table.n - 1, 1))
        elif status == GameStatus.NATURAL:
            # Give gamblers win natural reward 1, others -1.
            # winner = np.array(self.table.players_status[:-1]).argmax()
            logger.info("Some lucky player got natural!")
            reward = (self.reward_win - self.reward_lose) * (np.array(self.table.players_status[:-1]) ==
                                                             PlayerStatus.NATURAL).reshape((self.table.n - 1, 1)) + \
                     self.reward_lose * np.ones((self.table.n - 1, 1))
            # reward = (self.reward_win - self.reward_lose) * np.eye(1, self.table.n - 1, winner).T + \
            #          self.reward_lose * np.ones((self.table.n - 1, 1))
            # Or give gamblers win natural reward 1, others 0.
            # self.player_reward += self.reward_win * (self.table.players_status[:-1] == PlayerState.NATURAL)
        elif status == GameStatus.END:
            if self.table.players_status[-1] == PlayerStatus.LOSE_BUST:  # If dealer goes bust.
                if gambler_status_cnt[PlayerStatus.STICK] >= 1:
                    winner = players_points.argmax()
                    logger.info("Dealer goes bust, gambler {} wins!".format(winner))
                    self.table.players[winner].status = PlayerStatus.WIN  # Reset winner's status
                    # Give winner self.reward_win, others self.reward_lose
                    reward = (self.reward_win - self.reward_lose) * np.eye(1, self.table.n - 1, winner).T + \
                             self.reward_lose * np.ones((self.table.n - 1, 1))
                    # TODO: What if there are ties in points?
                else:
                    logger.info("Everyone goes bust!")
                    reward = np.zeros((self.table.n - 1, 1))
            else:  # Dealer not bust
                dealer_point = players_points[-1]
                max_gambler_point = players_points[:-1].max()
                # Compare points of gamblers not busted and point of dealer
                if max_gambler_point < dealer_point:  # Gambler lose
                    logger.info("Dealer wins!")
                    self.table.players[-1].status = PlayerStatus.WIN
                    reward = self.reward_lose * np.ones((self.table.n - 1, 1))
                elif max_gambler_point == dealer_point:  # Tie
                    logger.info("The game ended in a tie...")
                    reward = np.zeros((self.table.n - 1, 1))
                else:  # Some gambler wins
                    winner = players_points.argmax()
                    logger.info("Gambler {}'s point is closer to 21, and he wins!".format(winner))
                    self.table.players[winner].status = PlayerStatus.WIN
                    reward = (self.reward_win - self.reward_lose) * np.eye(1, self.table.n - 1, winner).T + \
                             self.reward_lose * np.ones((self.table.n - 1, 1))

        return reward

    def play(self, model):
        logger.info("Game starts with {} decks of cards and {} players.".format(self.table.m, self.table.n))
        self.reset()
        status = self.__status()

        # if status in (GameStatus.NATURAL, GameStatus.DRAW):
        if self.__is_over(status):
            # Settlement
            reward = self.__settlement(status)
            # End game
            self.player_reward += reward
            logger.info("Finish with status {} and total reward {}.".format(GameStatus(status).name, self.player_reward.T))
            return status

        observation = self.__observe()
        while True:
            player = self.table.players[self.act]
            if self.act == self.table.n-1:
                action = DealerPolicy.predict(state=observation)
            else:
                action = model.predict(state=observation)
            # action = player.policy.predict(state=observation)
            observation, status = self.step(action)

            # if status == GameStatus.END:
            if self.__is_over(status):
                # Settlement
                reward = self.__settlement(status)
                # End game
                self.player_reward += reward

                logger.info("Game ends with points {}, result {}, total reward {}".format(
                    [p.state.points for p in self.table.players],
                    [PlayerStatus(ps).name for ps in self.table.players_status],
                    self.player_reward.T))
                return status

