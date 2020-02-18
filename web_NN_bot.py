import h5py

from dlgo.agent.naive import RandomBot
from dlgo.httpfrontend.server import get_web_app
from dlgo.agent.predict import DeepLearningAgent, load_prediction_agent

model_file = h5py.File("models/book_model.h5", "r")
bot_from_file = load_prediction_agent(model_file)

web_app = get_web_app({'predict': bot_from_file})
web_app.run()
