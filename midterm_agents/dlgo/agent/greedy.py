import numpy as np

from dlgo.agent.base import Agent
from dlgo.agent.helpers_fast import is_point_an_eye
from dlgo.goboard_fast import Move
from dlgo import gotypes
from dlgo.gotypes import Point


__all__ = ['GreedyBot']

def capture_diff(game_state):
    black_stones = 0
    white_stones = 0
    for r in range(1, game_state.board.num_rows + 1):
        for c in range(1, game_state.board.num_cols + 1):
            p = gotypes.Point(r, c)
            color = game_state.board.get(p)
            if color == gotypes.Player.black:
                black_stones += 1
            elif color == gotypes.Player.white:
                white_stones += 1
    diff = black_stones - white_stones
    if game_state.next_player == gotypes.Player.black:
        return diff
    return -1 * diff

class GreedyBot(Agent):
    def __init__(self):
        Agent.__init__(self)
        self.dim = None
        self.point_cache = []

    def _update_cache(self, dim):
        self.dim = dim
        rows, cols = dim
        self.point_cache = []
        for r in range(1, rows + 1):
            for c in range(1, cols + 1):
                self.point_cache.append(Point(row=r, col=c))

    def select_move(self, game_state):
        """Choose a random valid move that preserves our own eyes."""
        dim = (game_state.board.num_rows, game_state.board.num_cols)
        if dim != self.dim:
            self._update_cache(dim)

        idx = np.arange(len(self.point_cache))
        current_diff = capture_diff(game_state)
        greedy_cands = []
        for i in idx:
            p = self.point_cache[i]
            if game_state.is_valid_move(Move.play(p)) and not is_point_an_eye(game_state.board, p, game_state.next_player):
                new_game_state = game_state.apply_move(Move.play(p))
                new_diff = -1 * capture_diff(new_game_state)
                if new_diff > current_diff:
                    greedy_cands = [Move.play(p)]
                    current_diff = new_diff
                if new_diff == current_diff:
                    greedy_cands.append(Move.play(p))

        if greedy_cands:
             return np.random.choice(greedy_cands)
        return Move.pass_turn()
