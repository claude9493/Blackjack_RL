import contextlib
import multiprocessing
from functools import partial

import pickle
import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from loguru import logger
from tqdm import tqdm

from environment import blackjack
from environment.core import GameStatus
from model.abstract_model import TestModel

logger.disable("environment.blackjack")

# TODO: Use Monte Carlo learning policy to evaluate the extreme policy and draw the figure in page 11 of Lecture 4
#  slide.

"""
Steps:
    1. Generate episodes of game experiences under the extreme policy $\pi$
    2. For each episode
        update value estimation of each state met incrementally $V(S_t) <- V(S_t) + 1/N(S_t)(G_t - V(S_t))$
    3. Draw the counter plot.
"""

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def indices_merged_arr_generic(arr, arr_pos="last"):
    n = arr.ndim
    grid = np.ogrid[tuple(map(slice, arr.shape))]
    out = np.empty(arr.shape + (n+1,), dtype=np.result_type(arr.dtype, int))

    if arr_pos=="first":
        offset = 1
    elif arr_pos=="last":
        offset = 0
    else:
        raise Exception("Invalid arr_pos")

    for i in range(n):
        out[...,i+offset] = grid[i]
    out[...,-1+offset] = arr
    out.shape = (-1,n+1)

    return out


def single_play(game, model, n=0):
    logger.disable("environment.blackjack")
    status, reward, player_trajectory = game.play(model)
    if status in (GameStatus.END, GameStatus.DRAW, GameStatus.NATURAL):
        return status, reward, player_trajectory


def parallel_play(game, model, R=1000, nprocs=0):
    if nprocs == 0:
        num_cores = multiprocessing.cpu_count()  # 8
    else:
        num_cores = min(multiprocessing.cpu_count(), abs(nprocs))
    f = partial(single_play, game, model)
    with tqdm_joblib(tqdm(desc="Simulation", total=R)) as progress_bar:
        res = Parallel(n_jobs=num_cores)(delayed(f)(i) for i in range(R))
    return res

if __name__ == '__main__':
    # Generate episodes
    logger.disable("environment.blackjack")
    R = 50000
    vf = np.zeros(shape=(10, 10, 2))

    table = blackjack.Table(m=0, n=2)  # Infinity deck of cards and 2 players
    game = blackjack.Blackjack(table=table)
    model = TestModel()
    # V = dict()

    states = np.zeros((10, 10, 2))
    states_count = np.zeros((10, 10, 2))

    res = parallel_play(game, model, R, nprocs=6)
    print(res)

    for episode in res:
        for reward, player_trajectory in zip(episode[1], episode[2]):
             for obs, _ in player_trajectory:
                 obs_index = tuple(np.array(obs) - np.array([12, 1, 0]))
                 states[obs_index] += reward
                 states_count[obs_index] += 1

    V = states / states_count

    V_df = pd.DataFrame(indices_merged_arr_generic(V), columns=["PlayerSum", "DealerShow", "UsableAce", "Value"])
    V_df.PlayerSum += 12
    V_df.DealerShow += 1

    print(V_df.Value.max())

    print(V_df.head())
    print(V_df.shape)

    # V_df.to_csv("MC_Learning_{}_episodes.csv".format(R), index=False)
