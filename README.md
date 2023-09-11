# Grid World
Trying various RL algorithms on grid_wrold

To read final report open the "Research_Project-Report.pdf" file

Expanding on https://github.com/MushroomRL/mushroom-rl/blob/dev/examples/grid_world_td.py, which replicates the experiments on the Grid World MDP as presented in: "Double Q-Learning", Hasselt H. V.. 2010.

To run the original script, run ```python grid_world.py```. This trains each agent on grid world, saves the models in ```\nps```, and produces a few graphical results. 

Adding an agent consists of adding it to anywhere in the script that there is a list of the agents (imports, names and ```for a in [...]```), and (if needed) specifying any additional parameters the agent needs in ```algorithm_params```.

## Agents Tested Are: 
(All from https://mushroomrl.readthedocs.io/en/latest/source/mushroom_rl.algorithms.value.html)
### Implemented in original script:
QLearning, DoubleQLearning, WeightedQLearning, SpeedyQLearning, SARSA
### Added by us:
SARSALambda, ExpectedSARSA, QLambda, RLearning, MaxminQLearning, RQLearning


## How to use:

```python grid_world_simple.py``` to run script comparing all models, then ```python create_graphs``` to graph the results (into ```results_simple```).

```python grid_world_mmq.py``` to run script comparing different n's for Maxmin Q learning, then ```python create_mm_graphs``` to graph the results (into ```results_mm```)

```python grid_world_beta.py``` and ```python grid_world_beta.py``` to run script comparing different hyperparameters, then ```python create_graphs_beta_lamb``` to graph the results (into ```results_beta``` and ```results_beta```)

