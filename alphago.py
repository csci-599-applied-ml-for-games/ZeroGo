import os
import h5py
import time
import numpy as np

from dlgo import rl
from dlgo import scoring
from dlgo import goboard_fast as goboard
from dlgo.goboard_fast import Move
from dlgo.gotypes import Player, Point
from dlgo.utils import print_board, print_move

from dlgo import agent
from dlgo.agent.predict import DeepLearningAgent, load_prediction_agent
from dlgo.agent import load_prediction_agent, load_policy_agent, AlphaGoMCTS
from dlgo.rl import load_value_agent

# Load policy agent and value agent
fast_policy = load_prediction_agent(
    h5py.File('models/AlphaGo/alphago_policyv0-0-0.h5', 'r'))
strong_policy = load_policy_agent(
    h5py.File('models/AlphaGo/alphago_policyv0-0-0.h5', 'r'))
value = load_value_agent(
    h5py.File('models/AlphaGo/alphago_valuev1-0-1.h5', 'r'))

# Create AlphaGo MCTS agent based on the policy agent and the value agent
alphago = AlphaGoMCTS(
    strong_policy, fast_policy, value, 
    depth=10, rollout_limit=50, num_simulations=100)

# Test duration for selecting a move
game_state = goboard.GameState.new_game(19)
start_time = time.time()
next_move = alphago.select_move(game_state)
exec_time = time.time() - start_time
print(exec_time)

# Testing the value agent
board_size = 19
game = goboard.GameState.new_game(board_size)
bots = {
    Player.black: agent.naive.RandomBot(),
    Player.white: fast_policy,
}
while not game.is_over():
    bot_move = bots[game.next_player].select_move(game)
    print('player: {}, val: {}'.format(game.next_player, value.predict(game)))
    game = game.apply_move(bot_move)