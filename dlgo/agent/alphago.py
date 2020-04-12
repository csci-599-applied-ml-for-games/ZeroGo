# tag::alphago_imports[]
import numpy as np
from dlgo.agent.base import Agent
from dlgo.goboard_fast import Move
from dlgo import kerasutil
from dlgo import scoring
from dlgo.utils import komi_eval, print_board, print_move
import operator
# end::alphago_imports[]

DEBUG = 0

__all__ = [
    'AlphaGoNode',
    'AlphaGoMCTS'
]


# tag::init_alphago_node[]
class AlphaGoNode:
    def __init__(self, parent=None, probability=1.0, q_sum=0, q_value=0):
        self.parent = parent  # <1>
        self.children = {}  # <1>

        self.visit_count = 0
        self.q_sum = q_sum
        self.q_value = q_value
        self.prior_value = probability  # <2>
        self.u_value = probability  # <3>
# <1> Tree nodes have one parent and potentially many children.
# <2> A node is initialized with a prior probability.
# <3> The utility function will be updated during search.
# end::init_alphago_node[]

# tag::select_node[]
    def select_child(self):
        return max(self.children.items(),
                   key=lambda child: child[1].q_value + \
                   child[1].u_value)
# end::select_node[]

# tag::expand_children[]
    def expand_children(self, moves, probabilities, values):
        for move, prob, value in zip(moves, probabilities, values):
            if move not in self.children:
                self.children[move] = AlphaGoNode(parent=self, probability=prob, 
                q_sum=value, q_value=value)
# end::expand_children[]

# tag::update_values[]
    def update_values(self, leaf_value):
        if self.parent is not None:
            self.parent.update_values(leaf_value)  # <1>

        self.visit_count += 1  # <2>

        self.q_sum += leaf_value
        self.q_value = self.q_sum / self.visit_count  # <3>

        if self.parent is not None:
            c_u = 3
            self.u_value = c_u * np.sqrt(self.parent.visit_count) \
                * self.prior_value / (1 + self.visit_count)  # <4>

# <1> We update parents first to ensure we traverse the tree top to bottom.
# <2> Increment the visit count for this node.
# <3> Add the specified leaf value to the Q-value, normalized by visit count.
# <4> Update utility with current visit counts.
# end::update_values[]


# tag::alphago_mcts_init[]
class AlphaGoMCTS(Agent):
    def __init__(self, policy_agent, fast_policy_agent, value_agent,
                 lambda_value=0.5, num_simulations=1000,
                 depth=50, rollout_limit=100,
                 verbose=False):
        #depth needs to be even
        if depth % 50 == 1:
            depth += 1
        self.verbose = verbose
                
        self.policy = policy_agent
        self.rollout_policy = fast_policy_agent
        self.value = value_agent

        self.lambda_value = lambda_value
        self.num_simulations = num_simulations
        self.depth = depth
        self.rollout_limit = rollout_limit
        self.root = AlphaGoNode()
# end::alphago_mcts_init[]

# tag::alphago_mcts_rollout[]
    def select_move(self, game_state):
        if self.verbose >= 2:
            print('================Searching phase================')
        for simulation in range(self.num_simulations):  # <1>
            current_state = game_state
            node = self.root
            for depth in range(self.depth):  # <2>
                if not node.children:  # <3>
                    if current_state.is_over():
                        break
                    moves, probabilities, values = self.policy_probabilities(current_state)  # <4>
                    node.expand_children(moves, probabilities, values)  # <4>

                move, node = node.select_child()  # <5>
                if self.verbose >= 2 and simulation % 10 == 0 and depth == 0:
                    print('Simulation {}/{}'.format(simulation, self.num_simulations))
                    print('Expanding move:')
                    print_move(None, move)
                    print('num of visits: {0}, Q value: {1:.4f}, u value: {2:.4f}'.format(
                        node.visit_count,
                        float(node.q_value), 
                        float(node.u_value)
                        ))
                current_state = current_state.apply_move(move)  # <5>

            value = self.value.predict(current_state)  # <6>
            rollout = self.policy_rollout(current_state)  # <6>

            weighted_value = (1 - self.lambda_value) * value + \
                self.lambda_value * rollout  # <7>

            node.update_values(weighted_value)  # <8>
# <1> From current state play out a number of simulations
# <2> Play moves until the specified depth is reached.
# <3> If the current node doesn't have any children...
# <4> ... expand them with probabilities from the strong policy.
# <5> If there are children, we can select one and play the corresponding move.
# <6> Compute output of value network and a rollout by the fast policy.
# <7> Determine the combined value function.
# <8> Update values for this node in the backup phase
# end::alphago_mcts_rollout[]

# tag::alphago_mcts_selection[]
        move = max(self.root.children, key=lambda move:  # <1>
                   self.root.children.get(move).visit_count)  # <1>
        if self.verbose >= 1:
            vals = {}
            for child in self.root.children:
                vals[child] = [self.root.children[child].visit_count,
                self.root.children[child].q_value,
                self.root.children[child].u_value,
                self.root.children[child].q_value + self.root.children[child].u_value]
            #sort by visit count
            sorted_vals = sorted(vals.items(), key=lambda x: x[1][0], reverse=True)
            print('================Candidate moves================')
            for i in range(5):
                print_move(None, sorted_vals[i][0])
                print("num of visits: {0}, q value: {1:.4f}, u value: {2:.4f}, sum value: {3:.4f}".format(
                    sorted_vals[i][1][0], 
                    float(sorted_vals[i][1][1]), 
                    float(sorted_vals[i][1][2]), 
                    float(sorted_vals[i][1][3])
                ))        

        self.root = AlphaGoNode()
        if move in self.root.children:  # <2>
            self.root = self.root.children[move]
            self.root.parent = None
        if self.verbose >= 3:
            print('================Current value: {0:.4f}================'.format(
                float(self.value.predict(game_state))
            ))

        if DEBUG:
            print('================Return Move================')
            print_move(None, move)
        return move
# <1> Pick most visited child of the root as next move.
# <2> If the picked move is a child, set new root to this child node.
# end::alphago_mcts_selection[]

# tag::alphago_policy_probs[]
    def policy_probabilities(self, game_state):
        encoder = self.policy._encoder
        outputs = self.policy.predict(game_state)
        legal_moves = game_state.legal_moves()
        if not legal_moves:
            return [], [], []
        encoded_points = [encoder.encode_point(move.point) for move in legal_moves if move.point]
        legal_outputs = outputs[encoded_points]

        top10_points = np.argsort(legal_outputs)[::-1][:10]
        top10_moves, top10_outputs = [], []
        for p in top10_points:
            top10_moves.append(legal_moves[p])
            top10_outputs.append(legal_outputs[p])
        
        normalized_outputs = top10_outputs / np.sum(top10_outputs)

        # predict value
        vals = []
        for move in top10_moves:
            new_state = game_state.apply_move(move)
            vals.append(float(self.value.predict(new_state)) * self.lambda_value)
        return top10_moves, normalized_outputs, vals
# end::alphago_policy_probs[]

# tag::alphago_policy_rollout[]
    def policy_rollout(self, game_state):
        my_side = game_state.next_player
        for step in range(self.rollout_limit):
            if game_state.is_over():
                break
            move_probabilities = self.rollout_policy.predict(game_state)
            encoder = self.rollout_policy.encoder
            for idx in np.argsort(move_probabilities)[::-1]:
                max_point = encoder.decode_point_index(idx)
                greedy_move = Move(max_point)
                if greedy_move in game_state.legal_moves():
                    game_state = game_state.apply_move(greedy_move)
                    break

        moves_cnt = len(game_state.previous_states)
        komi = komi_eval(moves_cnt)
        game_result = scoring.compute_game_result(game_state, komi=komi)
        
        winner = game_result.winner
        if winner is not None:
            win = 1 if winner == my_side else -1
            win_margin = np.clip(game_result.winning_margin, -10, 10) * 0.1
            if win == -1:
                win_margin = -1 * win_margin
            return win_margin
        else:
            return 0

        # winner = game_state.winner()
        # if winner:
        #     if winner == my_side:
        #         return 1
        #     else:
        #         return -1
        # else:
        #     return 0

        

# end::alphago_policy_rollout[]


    def serialize(self, h5file):
        raise IOError("AlphaGoMCTS agent can\'t be serialized" +
                       "consider serializing the three underlying" +
                       "neural networks instad.")
