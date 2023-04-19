# Grid World
Trying various RL algorithms on grid_wrold

Expanding on https://github.com/MushroomRL/mushroom-rl/blob/dev/examples/grid_world_td.py, which replicates the experiments on the Grid World MDP as presented in: "Double Q-Learning", Hasselt H. V.. 2010.

To run the script, run ```python grid_world.py```. This trains each agent on grid world, saves the models in ```\nps```, and produces a few graphical results.  

Adding an agent consists of adding it to anywhere in the script that there is a list of the agents (imports, names and ```for a in [...]```), and (if needed) specifying any additional parameters the agent needs in ```algorithm_params```

## Agents Tested Are: 
(All from https://mushroomrl.readthedocs.io/en/latest/source/mushroom_rl.algorithms.value.html)
### Implemented in original script:
QLearning, DoubleQLearning, WeightedQLearning, SpeedyQLearning, SARSA,
### Implemented thus far:
SARSALambda, ExpectedSARSA, QLambda, RLearning, MaxminQLearning, RQLearning
### Not yet implemented (don't necessarily need to get through all of these, depends on how much trouble they are to implement):
FQI, DoubleFQI, BoostedFQI, LSPI, DQN and its variants

