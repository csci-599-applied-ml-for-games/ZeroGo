import os
import random
import time
import h5py
import argparse
from collections import namedtuple

from dlgo.agent.greedy import GreedyBot
from dlgo.agent.naive import RandomBot
from dlgo import mcts
from dlgo.agent.predict import DeepLearningAgent, load_prediction_agent

from dlgo import gotypes
from dlgo import scoring
from dlgo.utils import print_board, print_move, point_from_coords
from dlgo.goboard_fast import GameState, Player, Point

class GameRecord(namedtuple('GameRecord', 'moves winner margin')):
    pass

def parse_agent(ag):
    if ag == 'rand':
        return RandomBot()
    if ag == 'greedy':
        return GreedyBot()
    if ag == 'mcts':
        return mcts.MCTSAgent(500, temperature=1.4)
    if ag == 'nn':
        model_file = h5py.File("train5000_compiled.h5", "r")
        bot_from_file = load_prediction_agent(model_file)
        return bot_from_file

def simulate_game(black_player, white_player, board_size, verbose=False):
    black_times = 0
    white_times = 0
    moves = []
    game = GameState.new_game(board_size)
    agents = {
        Player.black: black_player,
        Player.white: white_player,
    }
    while not game.is_over():
        start_time = time.time()
        next_move = agents[game.next_player].select_move(game)
        dur = time.time() - start_time
        if game.next_player == Player.black:
            black_times += dur
        else:
            white_times += dur
        game = game.apply_move(next_move)
        if verbose: 
            if dur < 0.1:
                time.sleep(0.05)
            print(chr(27) + "[2J")
            print_move(game.next_player, next_move)
            print_board(game.board)
            print()
    if verbose:
        print(chr(27) + "[2J")
        print_board(game.board)
    game_result = scoring.compute_game_result(game, komi=1.5)
    print(game_result)

    return GameRecord(
        moves=moves,
        winner=game_result.winner,
        margin=game_result.winning_margin,
    ), black_times, white_times

def play_games(agent1, agent2, num_games, board_size, verbose=False):
    wins, losses = 0, 0
    agent1_time, agent2_time =0, 0
    color1 = Player.black
    for i in range(num_games):
        print('Simulating game %d/%d...' % (i + 1, num_games))
        if color1 == Player.black:
            black_player, white_player = agent1, agent2
        else:
            white_player, black_player = agent1, agent2
        game_record, black_times, white_times = simulate_game(black_player, white_player, board_size, verbose)
        if game_record.winner == color1:
            print('Agent 1 wins')
            wins += 1
        else:
            print('Agent 2 wins')
            losses += 1
        if color1 == Player.black:
            agent1_time += black_times
            agent2_time += white_times
        else:
            agent1_time += white_times
            agent2_time += black_times
        if not verbose:
            print('Agent 1 record: %d/%d' % (wins, wins + losses))
            print('Agent 1 time: ', agent1_time)
            print('Agent 2 time: ', agent2_time)
        color1 = color1.other
    return wins, losses

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent1', default='rand')
    parser.add_argument('--agent2', default='rand')
    parser.add_argument('--verbose', '-v', type=bool, default=False)
    parser.add_argument('--board_size', '-b', type=int, default=19)
    parser.add_argument('--num_games', '-n', type=int, default=100)
    args = parser.parse_args()

    agent1 = parse_agent(args.agent1)
    agent2 = parse_agent(args.agent2)
    board_size = args.board_size
    num_games = args.num_games
    verbose = args.verbose

    play_games(agent1, agent2, num_games, board_size, verbose)