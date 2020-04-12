import h5py

from dlgo.agent.naive import RandomBot
from dlgo.httpfrontend.server import get_web_app
from dlgo.agent.predict import DeepLearningAgent, load_prediction_agent
from dlgo.agent import load_prediction_agent, load_policy_agent, AlphaGoMCTS
from dlgo.rl import load_value_agent, load_ac_agent

## Alphago agent
fast_policy = load_prediction_agent(
    h5py.File('./models/AlphaGo/alphago_policyv0-0-0.h5', 'r'))
strong_policy = load_policy_agent(
    h5py.File('./models/AlphaGo/alphago_policyv0-0-0.h5', 'r'))
value = load_value_agent(
    h5py.File('./models/AlphaGo/alphago_valuev1-2-6.h5', 'r'))
alphago = AlphaGoMCTS(
    strong_policy, fast_policy, value, 
    depth=8, rollout_limit=1, num_simulations=30, verbose=1)

alphago_head = AlphaGoMCTS(
    strong_policy, fast_policy, value, 
    depth=10, rollout_limit=10, num_simulations=30, verbose=3)

## NN agent
# model_file = h5py.File("./models/debugged models/agentv1-0-1.h5", "r")
# bot_from_file = load_prediction_agent(model_file)

## AC agent
# model_file = h5py.File("./models/AC/agent_00021000.hdf5")
# bot_from_file = load_ac_agent(model_file)

web_app = get_web_app({'predict': alphago_head})
web_app.run(threaded=False)
