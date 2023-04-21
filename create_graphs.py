import numpy as np
from matplotlib import pyplot as plt
from joblib import Parallel, delayed

from mushroom_rl.algorithms.value import QLearning, DoubleQLearning,\
    WeightedQLearning, SpeedyQLearning, SARSA,\
        SARSALambda, ExpectedSARSA, QLambda, RLearning, MaxminQLearning, RQLearning

names = {1: '1', .8: '08', QLearning: 'Q', DoubleQLearning: 'DQ',
             WeightedQLearning: 'WQ', SpeedyQLearning: 'SPQ', SARSA: 'SARSA',
             SARSALambda: 'SARSAL', ExpectedSARSA: 'ESARSA', QLambda: 'QL', RLearning: 'RL', MaxminQLearning: 'MMQ', RQLearning: 'RQ'}

names_legend = {1: '1', .8: '08', QLearning: 'Q Learning', DoubleQLearning: 'Double Q Learning',
             WeightedQLearning: 'Weighted Q Learning', SpeedyQLearning: 'Speed Q Learning', SARSA: 'SARSA',
             SARSALambda: 'SARSA Lambda', ExpectedSARSA: 'Expected SARSA', QLambda: 'Q Lambda', RLearning: 'R Learning', MaxminQLearning: 'Maxmin Q Learning', RQLearning: 'RQ Learning'}


legend_labels = []
e = .8

folder = 'nps_simple/'

plt.figure().set_figwidth(15)

for a in [QLearning, DoubleQLearning, WeightedQLearning,
                  SpeedyQLearning, SARSA, SARSALambda, ExpectedSARSA, QLambda, RLearning, MaxminQLearning, RQLearning,]:

    r = np.load(folder + names[a] + '_' + names[e] + '_r.npy')
    plt.plot(r)
    plt.title("r")
    legend_labels.append(names_legend[a])
plt.legend(legend_labels, bbox_to_anchor=(1.01, 1), loc="upper left")
plt.xlabel('Number of Time Steps')
plt.ylabel('Average Reward per Time Step')
plt.title('Rewards')
plt.subplots_adjust(right=0.7)
plt.savefig('results_simple/final_r' + names[e] + '.png', dpi=300)
print("rs plotted")
plt.clf()

for a in [QLearning, DoubleQLearning, WeightedQLearning,
                  SpeedyQLearning, SARSA, SARSALambda, ExpectedSARSA, QLambda, RLearning, MaxminQLearning, RQLearning,]:

    max_Qs = np.load(folder + names[a] + '_' + names[e] + '_maxQ.npy')
    plt.plot(max_Qs)
    plt.title("Max Qs")
plt.legend(legend_labels, bbox_to_anchor=(1.01, 1), loc="upper left")
plt.xlabel('Number of Time Steps')
plt.ylabel('Maximum Q Value from Starting Square')
plt.title('Q Values')
plt.subplots_adjust(right=0.7)
plt.savefig('results_simple/final_q' + names[e] + '.png', dpi=300)
print("qs plotted")