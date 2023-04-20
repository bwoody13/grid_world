# Grid World
Trying various RL algorithms on grid_wrold

Expanding on https://github.com/MushroomRL/mushroom-rl/blob/dev/examples/grid_world_td.py, which replicates the experiments on the Grid World MDP as presented in: "Double Q-Learning", Hasselt H. V.. 2010.

To run the script, run ```python grid_world.py```. This trains each agent on grid world, saves the models in ```\nps```, and produces a few graphical results.  

Adding an agent consists of adding it to anywhere in the script that there is a list of the agents (imports, names and ```for a in [...]```), and (if needed) specifying any additional parameters the agent needs in ```algorithm_params```

## Agents Tested Are: 
(All from https://mushroomrl.readthedocs.io/en/latest/source/mushroom_rl.algorithms.value.html)
### Implemented in original script:
QLearning, DoubleQLearning, WeightedQLearning, SpeedyQLearning, SARSA
### Added thus far:
SARSALambda, ExpectedSARSA, QLambda, RLearning, MaxminQLearning, RQLearning (these weren't too much trouble to add)
### Not yet added (don't necessarily need to get through all of these, depends on how much trouble they are to implement):
FQI, DoubleFQI, BoostedFQI, LSPI, DQN and its variants (seems like these may take more effort)
FQI (and, presumably, most/all of these methods) doesn't have a Q-table, which it seems like the script is made to build and save. So the script would need some more reworking to get these running. But perhaps just comparing the ones that were already there and the ones we added is enough.

