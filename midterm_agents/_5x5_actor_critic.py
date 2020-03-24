import datetime
import multiprocessing
import os
import random
import shutil
import time
import tempfile
from collections import namedtuple

import h5py
import numpy as np

from dlgo import goboard_fast as goboard
from dlgo import gotypes
from dlgo.utils import print_board, print_move, point_from_coords

from dlgo import kerasutil
from dlgo import scoring
from dlgo import rl
from dlgo.rl import ac_pass
from dlgo.goboard_fast import GameState, Player, Point

BOARD_SIZE = 5

def load_agent(filename):
    with h5py.File(filename, 'r') as h5file:
        return ac_pass.load_passing_ac_agent(h5file)

# def play_games(args):
#     agent = load_agent(agent1_fname)

#     wins, losses = 0, 0
#     color1 = Player.black
#     for i in range(num_games):
#         print('Simulating game %d/%d...' % (i + 1, num_games))
#         if color1 == Player.black:
#             black_player, white_player = agent1, agent2
#         else:
#             white_player, black_player = agent1, agent2
#         game_record = simulate_game(black_player, white_player, board_size)
#         if game_record.winner == color1:
#             print('Agent 1 wins')
#             wins += 1
#         else:
#             print('Agent 2 wins')
#             losses += 1
#         print('Agent 1 record: %d/%d' % (wins, wins + losses))
#         color1 = color1.other
#     return wins, losses

def main():
    game = goboard.GameState.new_game(BOARD_SIZE)
    bot = load_agent('agent_00015000.hdf5')

    while not game.is_over():
        print_board(game.board)
        if game.next_player == gotypes.Player.black:
            human_move = input('-- ')
            
            point = point_from_coords(human_move.strip())
            move = goboard.Move.play(point)
        else:
            move = bot.select_move(game)
        print_move(game.next_player, move)
        game = game.apply_move(move)


if __name__ == '__main__':
    main()
