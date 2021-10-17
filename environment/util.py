import numpy as np
import pandas as pd
from typing import Union, List
from environment.core import Gambler


class ExperienceRecorder(object):
    """
    Observe and record Blackjack game episodes.
    obs: dict of gamblers' playing history, e.g.
        {
            0: [(15, 4, 0), (20, 4, 0)],  #
            1: [(), (), ()],
            2: []
        }
    val: list of gamblers' rewards, e.g.
        [-1, -1, -1]  # assuming dealer wins
    """
    def __init__(self):
        self.__reset()

    def __reset(self):
        self.obs = {}
        self.val = None
        # self.N = 0

    def reset(self):
        self.__reset()

    def watch(self, gamblers: List[Gambler]) -> None:
        pass

