from __future__ import print_function

from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Activation, Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.regularizers import l2
from keras import backend as K


def root_mean_squared_error(y_true, y_pred):
    """
    RMSE loss function
    """
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def center_normalize(x):
    """
    Custom activation for online sample-wise center and std. normalization
    """
    return (x - K.mean(x)) / K.std(x)


def get_model():
    model = Sequential()
    model.add(Activation(activation=center_normalize, input_shape=(30, 64, 64)))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(96, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(96, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(128, 2, 2, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(128, 2, 2, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, W_regularizer=l2(1e-3)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))

    adam = Adam(lr=0.0001)
    model.compile(optimizer=adam, loss=root_mean_squared_error)
    return model
