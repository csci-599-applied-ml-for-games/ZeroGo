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