import h5py
from dlgo.networks import small, large
from keras.models import Sequential
from keras.layers.core import Dense
from dlgo.encoders.sevenplane import SevenPlaneEncoder
from dlgo.encoders.oneplane import OnePlaneEncoder
from dlgo.agent.predict import DeepLearningAgent, load_prediction_agent

go_board_rows, go_board_cols = 19, 19
num_classes = go_board_rows * go_board_cols
encoder = OnePlaneEncoder((go_board_rows, go_board_cols))  # <1>
input_shape = (encoder.num_planes, go_board_rows, go_board_cols)

network_layers = small.layers(input_shape)
model = Sequential()
for layer in network_layers:
    model.add(layer)
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
model.load_weights('small_model_epoch_5.h5')
deep_learning_bot = DeepLearningAgent(model, encoder)
f = h5py.File('models/book_model.h5','w')
deep_learning_bot.serialize(f)