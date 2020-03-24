from keras.models import Sequential
from keras.layers import LeakyReLU, PReLU
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D

def alphago_model(input_shape, is_policy_net=False, num_classes=19*19):

    model = Sequential()
    layers = [
      Conv2D(48, (7, 7), input_shape=input_shape, padding='same', data_format='channels_first'),
      LeakyReLU(alpha=0.1),

      Conv2D(48, (5, 5), padding='same', data_format='channels_first'),
      LeakyReLU(alpha=0.1),

      Conv2D(32, (5, 5), padding='same', data_format='channels_first'),
      LeakyReLU(alpha=0.1),

      Conv2D(32, (5, 5), padding='same', data_format='channels_first'),
      LeakyReLU(alpha=0.1),
    ]

    for layer in layers:
      model.add(layer)

    if is_policy_net:
        model.add(Flatten())
        model.add(Dropout(0.25))
        model.add(Dense(num_classes, activation='softmax'))
        return model
    else:
        model.add(
            Conv2D(192, 3, padding='same',
                   data_format='channels_first', activation='relu'))
        model.add(
            Conv2D(filters=1, kernel_size=1, padding='same',
                   data_format='channels_first', activation='relu'))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(1, activation='tanh'))
        return model