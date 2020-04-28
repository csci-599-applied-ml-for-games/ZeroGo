# ZeroGo
ZeroGo is an attempt to use machine learning techniques to develop agents for the game of Go using limited hardware resources. It uses the AlphaGo style Monte-Carlo Tree Search to combine the results of a policy agent and a value agent. By reducing the dimension of board encoder and the number of layers/units of the network, the ZeroGo agent is able to operate on macbooks and PCs. Depending on the search depth and rollout times, ZeroGo agent reaches up to 1-dan player level.

<a href="https://github.com/csci-599-applied-ml-for-games/ZeroGo/tree/master/deliverables/final">Click to see all deliverables of ZeroGo</a>

## To run a human vs bot game:
* $python3 dl_app.py
> ZeroGo is developed using tensorflow version 1.13, you might need to use tensorflow 1.x to run the program
* Go to http://127.0.0.1:5000/static/play_predict_19.html


## Current models:
* models/AC: actor critic models based on 5x5 board
* models/AlphaGo: policy and value agents on 19x19 board. Policy v0-0-0 is based on the previous NN model with 27% accuracy

## midterm_agents
#### Contains all agents before midterm, including:
- random (5x5 and 19x19)
- greedy (5x5 and 19x19)
- depth_pruned (5x5)
- alpha_beta (5x5)
- mcts (5x5)
- actor_critic (5x5)
- NN (19x19)

#### 2 game hosts
- __5x5_host.py
- __19x19_host.py

## rl.py
* process to simulate games and train the RL agents

## alphago.py
* process to debug and test the alphago MCTS agent

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