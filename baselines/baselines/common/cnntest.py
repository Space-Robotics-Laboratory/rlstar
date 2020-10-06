from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
from baselines.a2c import utils
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch
from baselines.common.mpi_running_mean_std import RunningMeanStd
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

mapping = {}

keras = tf.keras


def register(name):
    def _thunk(func):
        mapping[name] = func
        return func

    return _thunk


def unit_testCNN():
    pixel = 32
    imgSize = pixel * pixel
    stateSize = 0
    totalSize = imgSize + stateSize
    H = 32
    W = 32
    input = np.zeros(3 * H * W)
    input = tf.reshape(input, (3, 1, H, W))
    output = mergedCNN(input)


@register("mergedCNN")
def mergedCNN(input):
    stateSize = 30
    H = 32
    W = 32
    imgSize = H * W
    # img = img.reshape((1,) + img.shape)
    inputs = tf.keras.Input(shape=(H, W,))
    model = Sequential()
    model.add(Conv2D(16, (3, 3), activation="relu", input_shape=(H, W, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(16, (3, 3), activation="relu"))
    model.add(Flatten())
    model.add(Dense(64, activation="relu"))
    model.add(Dense(10))

    model.summary()

    output = model.predict(input)

    return output


unit_testCNN()

# later


#   x1 = tf.keras.layers.Conv2D(filters=32, kernal_size=4, activation="relu")(inputs)
#  x2 = tf.keras.layers.Conv2D(filters=16, kernal_size=8)(inputs)
# x3 = tf.keras.layers.Concatenate(axis=-1)[x1,x2]
# model.add(MaxPooling2D(pool_size=(2, 2), stides=(2, 2)))
# submodels = []
# numFilters = [32, 16, 4]
#  i = 0
#  for kernal in (2, 4, 8):
#  submodel = Sequential()
#  submodel.add(Conv2D(filters=numFilters[i], kernal_size=(kernal, kernal), activation="relu"), input_shape=input)
# i+=1
#  submodels.append(submodel)
# bigCNN = Sequential()
# bigCNN.add(Merge(submodels, mode="concat"))
# bigCNN.add(Dense(128))
# bigCNN.add(Activation("relu"))
