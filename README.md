# ZeroGo

## Current models:
* models/AC: actor critic models based on 5x5 board
* models/AlphaGo: policy and value agents on 19x19 board. Policy v0-0-0 is based on the previous NN model with 27% accuracy

## midterm_agents
* Contains all agents before midterm, including:
- random (5x5 and 19x19)
- greedy (5x5 and 19x19)
- depth_pruned (5x5)
- alpha_beta (5x5)
- mcts (5x5)
- actor_critic (5x5)
- NN (19x19)

* 2 game hosts
- __5x5_host.py
- __19x19_host.py

## rl.py
* process to simulate games and train the RL agents

## dlgo bug fixes
* AlphaGo MCTS bugs 
> https://github.com/maxpumperla/deep_learning_and_the_game_of_go/issues/55
* dlgo-rl-simulate.py line 50: white_player, black_player = agent1, agent2

## NN_end_to_end.py
* end to end script from downloading data to open web app
* steps:
1. set encoder
2. set data processor/generator
3. construct NN layers
4. compile model and train
5. save model
6. initiate deep learning agent
7. start web app