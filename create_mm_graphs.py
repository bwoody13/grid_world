import numpy as np
from matplotlib import pyplot as plt
from joblib import Parallel, delayed


names = {1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 8: '8', 10: '10', 12: '12'}

names_legend = {1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 8: '8', 10: '10', 12: '12'}


legend_labels = ['Optimal']
e = .8

folder = 'nps_mm/'

plt.figure().set_figwidth(15)

plt.axhline(y = 0.2, color = 'k', linestyle = 'dashed')
for a in [1, 2, 3, 4, 6, 8, 10, 12]:

    r = np.load(folder + 'mm' + names[a] + '_r.npy')
    plt.plot(r)
    legend_labels.append(names_legend[a])
plt.legend(legend_labels, bbox_to_anchor=(1.01, 1), loc="upper left")
plt.xlabel('Number of Time Steps')
plt.ylabel('Average Reward per Time Step')
plt.title('Rewards for Maxmin Q-Learning with Different Values for N')
plt.subplots_adjust(right=0.7)
plt.savefig('results_mm/final_r.png', dpi=300)
print("rs plotted")
plt.clf()

plt.axhline(y = 0.36, color = 'k', linestyle = 'dashed')
for a in [1, 2, 3, 4, 6, 8, 10, 12]:

    max_Qs = np.load(folder + 'mm' + names[a] + '_maxQ.npy')
    plt.plot(max_Qs)

plt.legend(legend_labels, bbox_to_anchor=(1.01, 1), loc="upper left")
plt.xlabel('Number of Time Steps')
plt.ylabel('Maximum Q Value from Starting Square')
plt.title('Q Values for Maxmin Q-Learning with Different Values for N')
plt.subplots_adjust(right=0.7)
plt.savefig('results_mm/final_q.png', dpi=300)
print("qs plotted")