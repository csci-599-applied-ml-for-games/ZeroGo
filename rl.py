import h5py
import os

from dlgo.networks.zerogo import alphago_model
from dlgo.encoders.sevenplane import SevenPlaneEncoder
from dlgo.rl.simulate import experience_simulation
from dlgo.rl import ValueAgent, load_experience, load_value_agent
from dlgo.agent.pg import PolicyAgent
from dlgo.agent import DeepLearningAgent, load_prediction_agent, load_policy_agent, AlphaGoMCTS

go_board_rows, go_board_cols = 19, 19
num_classes = go_board_rows * go_board_cols
encoder = SevenPlaneEncoder((go_board_rows, go_board_cols))
input_shape = (encoder.num_planes, go_board_rows, go_board_cols)

# load policy agent and its opponent
sl_agent = load_prediction_agent(
    h5py.File('models/AlphaGo/alphago_policyv0-0-0.h5', 'r')
    )  # this model is based on the NN model (28% accuracy)
sl_opponent = load_prediction_agent(
    h5py.File('models/AlphaGo/alphago_policyv0-0-0.h5', 'r')
    )  # same with the above agent
alphago_rl_agent = PolicyAgent(sl_agent.model, encoder) 
opponent = PolicyAgent(sl_agent.model, encoder)

# load value agent
alphago_value_network = alphago_model(input_shape)
alphago_value = ValueAgent(alphago_value_network, encoder)  # this value agent is not trained yet

# Simulate games
for i in range(10):
    print('Simulation exp ', i)
    num_games = 100
    experience = experience_simulation(num_games, alphago_rl_agent, opponent)
    with h5py.File('exp{}.h5'.format(i), 'w') as exp_out: 
        experience.serialize(exp_out)

# Train policy agent
for i in range(10):
    exp_file = 'exp{}.h5'.format(i)
    if os.path.exists(exp_file):
        experience = load_experience(h5py.File(exp_file, 'r'))
        alphago_rl_agent.train(experience)

# Train value agent
for i in range(10):
    exp_file = 'exp{}.h5'.format(i)
    if os.path.exists(exp_file):
        experience = load_experience(h5py.File(exp_file, 'r'))
        alphago_value.train(
            experience, lr=1e-4, use_value_rewards=True
            )   # i used another more complex reward method instead of purely -1 and 1
                # not sure if it works better

# Serialize agents
with h5py.File('policyv1-0-0', 'w') as policy_agent_out: 
    alphago_rl_agent.serialize(policy_agent_out)
with h5py.File('valuev1-0-0', 'w') as value_agent_out: 
    alphago_value.serialize(value_agent_out)


## Evaluate the Policy Agent
import time
from collections import namedtuple
from dlgo import scoring
from dlgo.goboard_fast import GameState, Player, Point

def simulate_game(black_player, white_player):
    moves = []
    game = GameState.new_game(BOARD_SIZE)
    agents = {
        Player.black: black_player,
        Player.white: white_player,
    }
    while not game.is_over():
        next_move = agents[game.next_player].select_move(game)
        moves.append(next_move)
        #if next_move.is_pass:
        #    print('%s passes' % name(game.next_player))
        game = game.apply_move(next_move)

    game_result = scoring.compute_game_result(game)
    print(game_result)

    return GameRecord(
        moves=moves,
        winner=game_result.winner,
        margin=game_result.winning_margin,
    )

class GameRecord(namedtuple('GameRecord', 'moves winner margin')):
  pass

BOARD_SIZE = 19
agent1 = alphago_rl_agent
agent2 = opponent

wins = 0
losses = 0
color1 = Player.black
for i in range(500):
    print('Simulating game %d/%d...' % (i + 1, 500))
    if color1 == Player.black:
        black_player, white_player = agent1, agent2
    else:
        white_player, black_player = agent1, agent2
    game_record = simulate_game(black_player, white_player)
    if game_record.winner == color1:
        wins += 1
    else:
        losses += 1
    color1 = color1.other
print('Agent 1 record: %d/%d' % (wins, wins + losses))