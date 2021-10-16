import contextlib
import multiprocessing
from functools import partial

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from loguru import logger
from tqdm import tqdm

from environment import blackjack
from environment.core import GameStatus
from environment.util import ExperienceRecorder
from model.abstractmodel import TestModel

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


def single_play(game, model, n=0):
    logger.disable("environment.blackjack")
    status = game.play(model)
    if status in (GameStatus.END, GameStatus.DRAW, GameStatus.NATURAL):
        return list(game.recorder.obs.values()), game.recorder.val[0].tolist()


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
    recorder = ExperienceRecorder()
    game.set_recorder(recorder)
    # game.play_n(model, recorder, N=N)
    V = dict()

    res = parallel_play(game, model, R, nprocs=6)
    # print(res)

    for episode in res:
        for observations, reward in zip(episode[0], episode[1]):
             for obs in observations:
                 obs_t = tuple(obs)
                 if obs_t not in V:
                     V[obs_t] = [0, 0]
                 V[obs_t][0] += reward
                 V[obs_t][1] += 1


    # for i in range(R):
    #     if i % (R/10) == 0:
    #         logger.info("Progress: {}/{}".format(i, R))
    #     status = game.play(model)
    #     if status in (GameStatus.END, GameStatus.DRAW, GameStatus.NATURAL):
    #         for observations, val in zip(recorder.obs.values(), recorder.val[0]):
    #             for obs in observations:
    #                 obs_t = tuple(obs)
    #                 if obs_t not in V:
    #                     V[obs_t] = [0, 0]
    #                 V[obs_t][0] += val
    #                 V[obs_t][1] += 1

    print(V)
    V = dict(map(lambda x: (x[0], x[1][0]/x[1][1]), V.items()))

    print(V)
    V_df = pd.Series(V).reset_index()
    V_df.columns = ['PlayerSum', 'DealerShow', "UsableAce", "Value"]
    print(V_df.head())
    print(V_df.shape)
    V_df.to_csv("MC_Learning_50000_episodes.csv", index=False)
