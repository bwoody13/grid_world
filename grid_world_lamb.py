import matplotlib
matplotlib.use('Agg')

import numpy as np
from matplotlib import pyplot as plt
from joblib import Parallel, delayed

from mushroom_rl.algorithms.value import QLearning, DoubleQLearning,\
    WeightedQLearning, SpeedyQLearning, SARSA,\
        SARSALambda, ExpectedSARSA, QLambda, RLearning, MaxminQLearning, RQLearning
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


def experiment(algorithm_class, exp, lambda_coeff=0.5, beta=0.5, n_tables=2):
    np.random.seed()

    TD_agents = [QLearning, DoubleQLearning, WeightedQLearning, SpeedyQLearning, SARSA, SARSALambda, ExpectedSARSA, QLambda, RLearning, MaxminQLearning, RQLearning]

    # MDP
    mdp = GridWorldVanHasselt()

    # Policy
    epsilon = ExponentialParameter(value=1, exp=.5, size=mdp.info.observation_space.size)
    pi = EpsGreedy(epsilon=epsilon)

    # Agent
    learning_rate = ExponentialParameter(value=1, exp=exp, size=mdp.info.size)
    algorithm_params = dict(learning_rate=learning_rate)
    
    # ADD EXTRA ALGORITHM PARAMS IF NEEDED 
    if algorithm_class in [SARSALambda, QLambda]:
        algorithm_params['lambda_coeff'] = lambda_coeff     # Unsure which is best
    if algorithm_class in [RLearning, RQLearning]:
        algorithm_params['beta'] = beta # Unsure which is best. Also RQ requires either beta or delta, not sure which one is best to provide
    if algorithm_class in [MaxminQLearning]:
        algorithm_params['n_tables'] = n_tables # Unsure which is best

    agent = algorithm_class(mdp.info, pi, **algorithm_params)

    # Algorithm
    start = mdp.convert_to_int(mdp._start, mdp._width)
    collect_max_Q = CollectMaxQ(agent.Q, start)
    collect_dataset = CollectDataset()
    callbacks = [collect_dataset, collect_max_Q]
    core = Core(agent, mdp, callbacks)
        
    # Train
    core.learn(n_steps=10000, n_steps_per_fit=1, quiet=True) #fewer steps for debugging

    _, _, reward, _, _, _ = parse_dataset(collect_dataset.get())
    max_Qs = collect_max_Q.get()

    return reward, max_Qs


if __name__ == '__main__':
    n_experiment = 10

    logger = Logger(QLearning.__name__, results_dir=None)
    logger.strong_line()
    logger.info('Experiment Algorithm: ' + QLearning.__name__)

    names = {1: '1', .8: '08', QLearning: 'Q', DoubleQLearning: 'DQ',
             WeightedQLearning: 'WQ', SpeedyQLearning: 'SPQ', SARSA: 'SARSA',
             SARSALambda: 'SARSAL', ExpectedSARSA: 'ESARSA', QLambda: 'QL', RLearning: 'RL', MaxminQLearning: 'MMQ', RQLearning: 'RQ'}
    e = 0.8
    logger.info(f'Exp: {e}')
    fig = plt.figure()
    plt.suptitle(names[e])
    legend_labels = []
    ticbig = time.perf_counter()
    for a in [SARSALambda, QLambda]:
        for lambda_coef in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
            tic = time.perf_counter()
            logger.info(f'Alg: {names[a]}-{lambda_coef}')
            out = Parallel(n_jobs=-1)(
                delayed(experiment)(a, e, lambda_coeff=lambda_coef) for _ in range(n_experiment))
            r = np.array([o[0] for o in out])
            max_Qs = np.array([o[1] for o in out])

            r = np.convolve(np.mean(r, 0), np.ones(100) / 100., 'valid')
            max_Qs = np.mean(max_Qs, 0)

            toc = time.perf_counter()
            file = open('results_lamb/times.txt', 'a')
            file.write(names[a] + ' with lambda = ' + str(lambda_coef) + ' took ' + str((toc-tic)/60) + ' minutes.')
            file.write("\n")
            file.close()

            np.save('nps_lamb/' + names[a] + str(int(lambda_coef*10)) + '_' + names[e] + '_r.npy', r)
            np.save('nps_lamb/' + names[a] + str(int(lambda_coef*10)) + '_' + names[e] + '_maxQ.npy', max_Qs)

            print("r")
            print(r)

            print("Max Qs")
            print(max_Qs)

            plt.subplot(2, 1, 1)
            plt.plot(r)
            plt.title("r")
            plt.subplot(2, 1, 2)
            plt.plot(max_Qs)
            plt.title("Max Qs")
            legend_labels.append(names[a]+f"-{lambda_coef}")
        plt.legend(legend_labels)
        fig.savefig('results_lamb/test_lamb.png')

        tocbig = time.perf_counter()
        file = open('results_lamb/times.txt', 'a')
        file.write('Overall: ' + str((tocbig - ticbig) / 60) + ' minutes.')
        file.close()
