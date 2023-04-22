import matplotlib
matplotlib.use('Agg')

import numpy as np
from matplotlib import pyplot as plt
from joblib import Parallel, delayed

from mushroom_rl.algorithms.value import MaxminQLearning, QLearning
from mushroom_rl.core import Core, Logger
from mushroom_rl.environments import *
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.utils.callbacks import CollectDataset, CollectMaxQ
from mushroom_rl.utils.dataset import parse_dataset
from mushroom_rl.utils.parameters import ExponentialParameter
from sklearn.ensemble import ExtraTreesRegressor
import time


"""
This script aims to replicate the experiments on the Grid World MDP as
presented in:
"Double Q-Learning", Hasselt H. V.. 2010.
SARSA and many variants of Q-Learning are used. 
"""


def experiment(algorithm_class, exp, n):
    np.random.seed()

    TD_agents = [MaxminQLearning]

    # MDP
    mdp = GridWorldVanHasselt()

    # Policy
    epsilon = ExponentialParameter(value=1, exp=.5, size=mdp.info.observation_space.size)
    pi = EpsGreedy(epsilon=epsilon)

    # Agent
    learning_rate = ExponentialParameter(value=1, exp=exp, size=mdp.info.size)
    algorithm_params = dict(learning_rate=learning_rate)
    
    
    if algorithm_class in [MaxminQLearning]:
        algorithm_params['n_tables'] = n # We can tune n to balance over and under estimation! Would be something good to play with https://arxiv.org/pdf/2002.06487.pdf 


    if algorithm_class in TD_agents:
        agent = algorithm_class(mdp.info, pi, **algorithm_params)
    
    # Algorithm
    if algorithm_class in TD_agents:
        start = mdp.convert_to_int(mdp._start, mdp._width)
        collect_max_Q = CollectMaxQ(agent.Q, start)
        collect_dataset = CollectDataset()
        callbacks = [collect_dataset, collect_max_Q]
        core = Core(agent, mdp, callbacks)
    
        
    # Train
    if algorithm_class in TD_agents:
        core.learn(n_steps=10000, n_steps_per_fit=1, quiet=True) #fewer steps for debugging
    
    _, _, reward, _, _, _ = parse_dataset(collect_dataset.get())
    max_Qs = collect_max_Q.get()

    return reward, max_Qs


if __name__ == '__main__':
    n_experiment = 10000

    logger = Logger(QLearning.__name__, results_dir=None)
    logger.strong_line()
    logger.info('Experiment Algorithm: ' + QLearning.__name__)

    
    file = open('results_mm/times.txt', 'w')
    file.write('')
    file.close()

    ns = [1, 2, 3, 4, 6, 8, 10, 12] #how many to do?

    for e in [.8]:
        
        fig = plt.figure()
        
        legend_labels = []
        ticbig = time.perf_counter()
        a = MaxminQLearning
        for n in ns:

            tic = time.perf_counter()

            logger.info(f'Alg: {n}')
            out = Parallel(n_jobs=-1)(
                delayed(experiment)(a, e, n) for _ in range(n_experiment))
            r = np.array([o[0] for o in out])
            max_Qs = np.array([o[1] for o in out])

            r = np.convolve(np.mean(r, 0), np.ones(100) / 100., 'valid')
            max_Qs = np.mean(max_Qs, 0)

            toc = time.perf_counter()

            file = open('results_mm/times.txt', 'a')
            file.write('Method ' + str(n) + ' took ' + str((toc-tic)/60) + ' minutes.')
            file.write("\n")
            file.close()

            np.save('nps_mm/mm' + str(n) + '_r.npy', r)
            np.save('nps_mm/mm' + str(n) + '_maxQ.npy', max_Qs)

            print("r")
            print(r)

            print("Max Qs")
            print(max_Qs)

            plt.subplot(1, 1, 1)
            plt.plot(r)
            plt.title("r")
            plt.subplot(1, 2, 1)
            plt.plot(max_Qs)
            plt.title("Max Qs")
            legend_labels.append(str(n))
        plt.legend(legend_labels)
        fig.savefig('results_mm/test_mm.png')


        tocbig = time.perf_counter()
        file = open('results_mm/times.txt', 'a')
        file.write('Overall: ' + str((tocbig-ticbig)/60) + ' minutes.')
        file.close()