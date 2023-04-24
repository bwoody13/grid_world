import numpy as np
from matplotlib import pyplot as plt
from joblib import Parallel, delayed


names = {0: '0', .1: '0.1', .2: '0.2', .3: '0.3', .4: '0.4', .5: '0.5', .6: '0.6', .7: '0.7', .8: '0.8', .9: '0.9', 1: '1'}

strings = {0: '0', .1: '1', .2: '2', .3: '3', .4: '4', .5: '5', .6: '6', .7: '7', .8: '8', .9: '9', 1: '10'}


legend_labels = ['Optimal']
e = .8

folder = 'nps_beta/'

plt.figure().set_figwidth(15)

plt.axhline(y = 0.2, color = 'k', linestyle = 'dashed')
for a in [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]:
    r = np.load(folder + 'RL' + strings[a] + '_08_r.npy')
    plt.plot(r)
    legend_labels.append(names[a])
plt.legend(legend_labels, bbox_to_anchor=(1.01, 1), loc="upper left")
plt.xlabel('Number of Time Steps')
plt.ylabel('Average Reward per Time Step')
plt.title('Rewards for R-Learning with Different Betas')
plt.subplots_adjust(right=0.7)
plt.savefig('results_beta/RL_r.png', dpi=300)
print("rs plotted")
plt.clf()

plt.axhline(y = 0.36, color = 'k', linestyle = 'dashed')
for a in [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]:

    max_Qs = np.load(folder + 'RL' + strings[a] + '_08_maxQ.npy')
    plt.plot(max_Qs)

plt.legend(legend_labels, bbox_to_anchor=(1.01, 1), loc="upper left")
plt.xlabel('Number of Time Steps')
plt.ylabel('Maximum Q Value from Starting Square')
plt.title('Q Values for R-Leraning with Different Betas')
plt.subplots_adjust(right=0.7)
plt.savefig('results_beta/RL_q.png', dpi=300)
print("qs plotted")



######################


plt.figure().set_figwidth(15)

plt.axhline(y = 0.2, color = 'k', linestyle = 'dashed')
for a in [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]:
    r = np.load(folder + 'RQ' + strings[a] + '_08_r.npy')
    plt.plot(r)
    legend_labels.append(names[a])
plt.legend(legend_labels, bbox_to_anchor=(1.01, 1), loc="upper left")
plt.xlabel('Number of Time Steps')
plt.ylabel('Average Reward per Time Step')
plt.title('Rewards for RQ-Learning with Different Betas')
plt.subplots_adjust(right=0.7)
plt.savefig('results_beta/RQ_r.png', dpi=300)
print("rs plotted")
plt.clf()

plt.axhline(y = 0.36, color = 'k', linestyle = 'dashed')
for a in [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]:

    max_Qs = np.load(folder + 'RQ' + strings[a] + '_08_maxQ.npy')
    plt.plot(max_Qs)

plt.legend(legend_labels, bbox_to_anchor=(1.01, 1), loc="upper left")
plt.xlabel('Number of Time Steps')
plt.ylabel('Maximum Q Value from Starting Square')
plt.title('Q Values for RQ-Leraning with Different Betas')
plt.subplots_adjust(right=0.7)
plt.savefig('results_beta/RQ_q.png', dpi=300)
print("qs plotted")



##############

##LAMBDA


legend_labels = ['Optimal']
e = .8

folder = 'nps_lamb/'

plt.figure().set_figwidth(15)

plt.axhline(y = 0.2, color = 'k', linestyle = 'dashed')
for a in [.1, .2, .3, .4, .5, .6, .7, .8, .9]:
    r = np.load(folder + 'QL' + strings[a] + '_08_r.npy')
    plt.plot(r)
    legend_labels.append(names[a])
plt.legend(legend_labels, bbox_to_anchor=(1.01, 1), loc="upper left")
plt.xlabel('Number of Time Steps')
plt.ylabel('Average Reward per Time Step')
plt.title('Rewards for Q-Lambda with Different Lambdas')
plt.subplots_adjust(right=0.7)
plt.savefig('results_lamb/QL_r.png', dpi=300)
print("rs plotted")
plt.clf()

plt.axhline(y = 0.36, color = 'k', linestyle = 'dashed')
for a in [.1, .2, .3, .4, .5, .6, .7, .8, .9]:

    max_Qs = np.load(folder + 'QL' + strings[a] + '_08_maxQ.npy')
    plt.plot(max_Qs)

plt.legend(legend_labels, bbox_to_anchor=(1.01, 1), loc="upper left")
plt.xlabel('Number of Time Steps')
plt.ylabel('Maximum Q Value from Starting Square')
plt.title('Q Values for Q-Lambda with Different Lambdas')
plt.subplots_adjust(right=0.7)
plt.savefig('results_lamb/QL_q.png', dpi=300)
print("qs plotted")



######################


plt.figure().set_figwidth(15)

plt.axhline(y = 0.2, color = 'k', linestyle = 'dashed')
for a in [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]:
    r = np.load(folder + 'SARSAL' + strings[a] + '_08_r.npy')
    plt.plot(r)
    legend_labels.append(names[a])
plt.legend(legend_labels, bbox_to_anchor=(1.01, 1), loc="upper left")
plt.xlabel('Number of Time Steps')
plt.ylabel('Average Reward per Time Step')
plt.title('Rewards for SARSA-Lambda with Different Lambdas')
plt.subplots_adjust(right=0.7)
plt.savefig('results_lamb/SARSAL_r.png', dpi=300)
print("rs plotted")
plt.clf()

plt.axhline(y = 0.36, color = 'k', linestyle = 'dashed')
for a in [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]:

    max_Qs = np.load(folder + 'SARSAL' + strings[a] + '_08_maxQ.npy')
    plt.plot(max_Qs)

plt.legend(legend_labels, bbox_to_anchor=(1.01, 1), loc="upper left")
plt.xlabel('Number of Time Steps')
plt.ylabel('Maximum Q Value from Starting Square')
plt.title('Q Values for SARSA-Lambda with Different Lambdas')
plt.subplots_adjust(right=0.7)
plt.savefig('results_lamb/SARSAL_q.png', dpi=300)
print("qs plotted")