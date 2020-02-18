import os
import sys
import h5py

from keras.models import Sequential
from keras.layers import Dense

from dlgo.data.parallel_processor import GoDataProcessor
from dlgo.encoders.oneplane import OnePlaneEncoder
from dlgo.encoders.sevenplane import SevenPlaneEncoder

from dlgo.networks import small, large
from keras.models import Sequential
from keras.layers.core import Dense
from keras.callbacks import ModelCheckpoint

from dlgo.httpfrontend import get_web_app
from dlgo.agent.predict import DeepLearningAgent, load_prediction_agent

# hyper params
go_board_rows, go_board_cols = 19, 19
num_classes = go_board_rows * go_board_cols
num_games = 1000
epochs = 50
batch_size = 128

encoder = SevenPlaneEncoder((go_board_rows, go_board_cols))  # <1>
processor = GoDataProcessor(encoder=encoder.name())  # <2>
generator = processor.load_go_data('train', num_games, use_generator=True)  # <3>
test_generator = processor.load_go_data('test', num_games, use_generator=True)

input_shape = (encoder.num_planes, go_board_rows, go_board_cols)
network_layers = large.layers(input_shape)
model = Sequential()
for layer in network_layers:
    model.add(layer)
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])


model.fit_generator(generator=generator.generate(batch_size, num_classes),  # <1>
                    epochs=epochs,
                    steps_per_epoch=generator.get_num_samples() / batch_size,  # <2>
                    validation_data=test_generator.generate(batch_size, num_classes),  # <3>
                    validation_steps=test_generator.get_num_samples() / batch_size,  # <4>
                    callbacks=[ModelCheckpoint('./checkpoints/small_model_epoch_{epoch}.h5')])  # <5>

model.evaluate_generator(generator=test_generator.generate(batch_size, num_classes),
                         steps=test_generator.get_num_samples() / batch_size)  # <6>
# <1> We specify a training data generator for our batch size...
# <2> ... and how many training steps per epoch we execute.
# <3> An additional generator is used for validation...
# <4> ... which also needs a number of steps.
# <5> After each epoch we persist a checkpoint of the model.
# <6> For evaluation we also speficy a generator and the number of steps.

model.save("model_path.h5")
model.load_weights('small_model_epoch_x.h5')

deep_learning_bot = DeepLearningAgent(model, encoder)
f = h5py.File('deep_bot.h5','w')
deep_learning_bot.serialize(f)

# tag::e2e_load_agent[]
model_file = h5py.File("deep_bot.h5", "r")
bot_from_file = load_prediction_agent(model_file)

web_app = get_web_app({'predict': bot_from_file})
web_app.run()
# end::e2e_load_agent[]