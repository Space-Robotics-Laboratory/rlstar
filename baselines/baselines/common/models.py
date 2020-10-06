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


@register("gabin")
def gabin(**conv_kwargs):
    def network_fn(X, nenv=1):
        return build_gabin(X, **conv_kwargs)
        # return contra_mixed_cnn(X, use_loc=True, use_loc_r=True, **conv_kwargs)

    return network_fn


def build_gabin(X, **conv_kwargs):
    """
    CNN from Nature paper with additional features for last two FC layers.
    """
    # print("cnn2")
    # print(X)
    stateSize = 37

    # split up the data which is passed into different layers
    info_features_clover1 = tf.squeeze(X[:, 0:stateSize, 0:1, -1], axis=2)
    info_features_clover2 = tf.squeeze(X[:, 0:stateSize, 1:2, -1], axis=2)

    activ(fc(info_features_clover1, '1fc1', nh=256, init_scale=np.sqrt(2)))

    activ(fc(info_features_clover2, '2fc1', nh=256, init_scale=np.sqrt(2)))

    x4 = tf.concat([h3, j3, info_features], axis=1)

    return


@register("mlp3")
def mlp3(num_layers=2, num_hidden=64, activation=tf.tanh, layer_norm=False):
    """
    Stack of fully-connected layers to be used in a policy / q-function approximator

    Parameters:
    ----------

    num_layers: int                 number of fully-connected layers (default: 2)

    num_hidden: int                 size of fully-connected layers (default: 64)

    activation:                     activation function (default: tf.tanh)

    Returns:
    -------

    function that builds fully connected network with a given input tensor / placeholder
    """

    def network_fn3(X):
        h = tf.layers.flatten(X)
        h = fc(h, 'mlp_fc1', nh=64, init_scale=np.sqrt(2))
        h = activation(h)
        h = fc(h, 'mlp_fc2', nh=128, init_scale=np.sqrt(2))
        h = activation(h)
        h = fc(h, 'mlp_fc3', nh=164, init_scale=np.sqrt(2))
        h = activation(h)
        h = fc(h, 'mlp_fc4', nh=128, init_scale=np.sqrt(2))
        h = activation(h)
        h = fc(h, 'mlp_fc5', nh=64, init_scale=np.sqrt(2))
        h = activation(h)

        print("\n\nFinal output: ", h)
        return h

    return network_fn3

@register("mlp2")
def mlp2(num_layers=2, num_hidden=64, activation=tf.tanh, layer_norm=False):
    """
    Stack of fully-connected layers to be used in a policy / q-function approximator

    Parameters:
    ----------

    num_layers: int                 number of fully-connected layers (default: 2)

    num_hidden: int                 size of fully-connected layers (default: 64)

    activation:                     activation function (default: tf.tanh)

    Returns:
    -------

    function that builds fully connected network with a given input tensor / placeholder
    """

    def network_fn2(X):
        h = tf.layers.flatten(X)
        h = fc(h, 'mlp_fc1', nh=64, init_scale=np.sqrt(2))
        h = fc(h, 'mlp_fc2', nh=128, init_scale=np.sqrt(2))
        h = fc(h, 'mlp_fc3', nh=164, init_scale=np.sqrt(2))
        h = fc(h, 'mlp_fc4', nh=128, init_scale=np.sqrt(2))
        h = fc(h, 'mlp_fc5', nh=64, init_scale=np.sqrt(2))
        h = activation(h)

        print("\n\nFinal output: ", h)
        return h

    return network_fn2


@register("mlp")
def mlp(num_layers=2, num_hidden=64, activation=tf.tanh, layer_norm=False):
    """
    Stack of fully-connected layers to be used in a policy / q-function approximator

    Parameters:
    ----------

    num_layers: int                 number of fully-connected layers (default: 2)

    num_hidden: int                 size of fully-connected layers (default: 64)

    activation:                     activation function (default: tf.tanh)

    Returns:
    -------

    function that builds fully connected network with a given input tensor / placeholder
    """

    def network_fn(X):
        h = tf.layers.flatten(X)
        for i in range(num_layers):
            h = fc(h, 'mlp_fc{}'.format(i), nh=num_hidden, init_scale=np.sqrt(2))
            if layer_norm:
                h = tf.contrib.layers.layer_norm(h, center=True, scale=True)
            h = activation(h)
        print("\n\nFinal output: ", h)
        return h

    return network_fn


@register("cnn2")
def cnn2(**conv_kwargs):
    def network_fn(X, nenv=1):
        return build_cnn2(X, **conv_kwargs)
        # return contra_mixed_cnn(X, use_loc=True, use_loc_r=True, **conv_kwargs)

    return network_fn


def build_cnn2(X, **conv_kwargs):
    """
    CNN from Nature paper with additional features for last two FC layers.
    """
    # print("cnn2")
    # print(X)
    stateSize = 37
    lidar1 = 32  # size of lidar 1
    lidar2 = 64  # resolution of lidar 2

    # split up the data which is passed into different layers
    info_features = tf.squeeze(X[:, 0:stateSize, 0:1, -1], axis=2)
    lidar1 = X[:, 0:lidar1, 0:lidar1, 1:2]
    lidar2 = X[:, 0:lidar2, 0:lidar2, 2:3]

    # LIDAR 1 CNN
    activ = tf.nn.relu
    h = activ(conv(lidar1, 'c1', nf=32, rf=8, stride=2, init_scale=np.sqrt(2), **conv_kwargs))
    h2 = activ(conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2), **conv_kwargs))
    h3 = activ(conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2), **conv_kwargs))
    h3 = conv_to_fc(h3)

    # LIDAR 2 CNN
    activ2 = tf.nn.relu
    j = activ(conv(lidar2, 'cj1', nf=64, rf=4, stride=2, init_scale=np.sqrt(2), **conv_kwargs))
    j2 = activ(conv(j, 'cj2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2), **conv_kwargs))
    j3 = activ(conv(j2, 'cj3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2), **conv_kwargs))
    j3 = conv_to_fc(j3)

    x4 = tf.concat([h3, j3, info_features], axis=1)

    return activ(fc(x4, 'fc1', nh=256, init_scale=np.sqrt(2)))


@register("cnnlstm")
def cnnlstm(**conv_kwargs):
    def network_fn(X, nenv=1):
        return build_cnnlstm(X, **conv_kwargs)

    return network_fn


def build_cnnlstm(X, nlstm=128, nenv=1, **conv_kwargs):
    """
    CNN from Nature paper with additional features for last two FC layers.
    """
    # print("cnn2")
    # print(X)
    stateSize = 37
    lidar1 = 32  # size of lidar 1
    lidar2 = 64  # resolution of lidar 2

    # info_features = tf.squeeze(X[:,0:1,0:stateSize,-1],axis=1)
    info_features = tf.squeeze(X[:, 0:stateSize, 0:1, -1], axis=2)
    # print("size of info features")
    # print(info_features)

    lidar1 = X[:, 0:lidar1, 0:lidar1, 1:2]  # far lidar 32x32
    # print("size of lidar 1")
    # print(lidar1)
    lidar2 = X[:, 0:lidar2, 0:lidar2, 2:3]  # near lidar 64x64
    # print("size of lidar 2")
    # print(lidar2)

    # LIDAR 1 CNN
    activ = tf.nn.relu
    h = activ(conv(lidar1, 'c1', nf=32, rf=8, stride=2, init_scale=np.sqrt(2), **conv_kwargs))
    h2 = activ(conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2), **conv_kwargs))
    h3 = activ(conv(h2, 'c3', nf=64, rf=4, stride=1, init_scale=np.sqrt(2), **conv_kwargs))
    h3 = conv_to_fc(h3)

    # LIDAR 2 CNN
    activ2 = tf.nn.relu
    j = activ(conv(lidar2, 'jc1', nf=64, rf=4, stride=2, init_scale=np.sqrt(2), **conv_kwargs))
    j2 = activ(conv(j, 'jc2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2), **conv_kwargs))
    j3 = activ(conv(j2, 'jc3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2), **conv_kwargs))
    j3 = conv_to_fc(j3)

    # state info, first 1 layer dense then merge
    x = activ(fc(info_features, "fc1", nh=128, init_scale=np.sqrt(2)))
    merged = tf.concat([h3, j3, x], axis=1)

    m1 = activ(fc(merged, 'fc_merged1', nh=256, init_scale=np.sqrt(2)))

    nbatch = m1.shape[0]
    nsteps = nbatch // nenv
    M = tf.placeholder(tf.float32, [nbatch])
    S = tf.placeholder(tf.float32, [nenv, 2 * nlstm])  # states
    xs = batch_to_seq(info_features, nenv, nsteps)
    ms = batch_to_seq(M, nenv, nsteps)
    # assume no layer norm for now
    x2, snew = utils.lstm(xs, ms, S, scope="merged_lstm1", nh=nlstm)

    out = seq_to_batch(x2)  # maybe I can feed this to a second lstm

    initial_state = np.zeros(S.shape.as_list(), dtype=float)

    return out, {"S": S, "M": M, "state": snew, "initial_state": initial_state}


@register("cnn1")
def cnn_small(**conv_kwargs):
    def network_fn(X):
        stateSize = 33
        # imgSize = 40
        pixel = 32
        size = pixel * pixel
        # IMG_SHAPE = (pixel, pixel, 3)
        # state = tf.slice(X, [0, 0], [-1, stateSize])  # 21 state plus 4 for goal --> 25 but no goals so --> 21
        # print(state.shape)
        # image = tf.slice(X, [0, stateSize], [-1, imgSize])

        h = tf.cast(X, tf.float32)  # / 255.
        activ = tf.nn.relu
        h = activ(conv(h, 'c1', nf=8, rf=8, stride=4, init_scale=np.sqrt(2), **conv_kwargs))
        h = activ(conv(h, 'c2', nf=16, rf=4, stride=2, init_scale=np.sqrt(2), **conv_kwargs))
        h = conv_to_fc(h)
        h = activ(fc(h, 'fc1', nh=128, init_scale=np.sqrt(2)))
        return h

    return network_fn


@register("cnn_small")
def cnn_small(**conv_kwargs):
    def network_fn(X):
        h = tf.cast(X, tf.float32) / 255.

        activ = tf.nn.relu
        h = activ(conv(h, 'c1', nf=8, rf=8, stride=4, init_scale=np.sqrt(2), **conv_kwargs))
        h = activ(conv(h, 'c2', nf=16, rf=4, stride=2, init_scale=np.sqrt(2), **conv_kwargs))
        h = conv_to_fc(h)
        h = activ(fc(h, 'fc1', nh=128, init_scale=np.sqrt(2)))
        return h

    return network_fn


@register("pre")
def pre():
    def network_fn(X):
        # cnn test
        pixel = 160
        size = pixel * pixel * 3
        IMG_SHAPE = (pixel, pixel, 3)
        stateSize = 21
        # https://www.tensorflow.org/tutorials/images/transfer_learning
        # This feature extractor converts each 160x160x3 image into a 5x5x1280 block of features. See what it does to the example batch of images:
        # Create the base model from the pre-trained model MobileNet V2


        print(image.shape)
        image = tf.reshape(image, [pixel, pixel, 3])
        print(image.shape)
        print(image.shape)
        image = tf.cast(image, tf.float32) / 255.
        print(image.shape)
        print(image)

        ## create the models
        base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                       include_top=False,
                                                       weights='imagenet')
        base_model.trainable = False
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        prediction_layer = keras.layers.Dense(16)  # converts to just yes no, but I want to know where in image
        model = tf.keras.Sequential([
            base_model,
            global_average_layer,
            prediction_layer
        ])

        base_model.trainable = False
        # cameraMidInput = prediction_layer

        second_model = tf.keras.Sequential([
            base_model,
            global_average_layer,
            prediction_layer
        ])
        camInput = tf.keras.layers.Input(shape=IMG_SHAPE, name="camera_input")

        stateInput = tf.keras.layers.Input(shape=(stateSize,), name="state_input")
        stateDense1 = tf.keras.layers.Dense(128)(stateInput)  # (b)  #(stateInput)
        stateDense2 = tf.keras.layers.Dense(256)(stateDense1)
        stateDense3 = tf.keras.layers.Dense(256)(stateDense2)
        with tf.variable_scope("convnet"):
            a = tf.placeholder(tf.float32, shape=(160, 160, 3))
            b = tf.placeholder(tf.float32, shape=(21,))
            outputs = tf.placeholder(tf.float32, [None, 2])
            feed_dict = {a: image, b: state}
            cameraMidInput = second_model.predict(camInput, steps=1)
            merged = tf.keras.layers.concatenate([stateDense3, cameraMidInput])
            mergedDense1 = tf.keras.layers.Dense(512)(merged)
            mergedDense2 = tf.keras.layers.Dense(256)(mergedDense1)
            mergedDense3 = tf.keras.layers.Dense(128)(mergedDense2)
            model = tf.keras.Model(inputs=[stateInput, camInput, merged], outputs=[mergedDense3])
            model.summary()
            # out = tf.sess.run(feed_dict = {a: image, b: state})
            out = model.predict([state, image])
            return out

            # concat = tf.keras.layers.concatenate()([stateMidInput, cameraMidInput])
            # model = tf.keras.Model(inputs=[state, image], outputs=[finalOut])

    return network_fn


@register("pre2")
def pre2():
    def network_fn(X):
        # cnn test
        pixel = 160
        size = pixel * pixel * 3
        IMG_SHAPE = (pixel, pixel, 3)
        stateSize = 21
        # https://www.tensorflow.org/tutorials/images/transfer_learning
        # This feature extractor converts each 160x160x3 image into a 5x5x1280 block of features. See what it does to the example batch of images:
        # Create the base model from the pre-trained model MobileNet V2

        state = tf.slice(X, [0, 0], [-1, stateSize])  # 21 state plus 4 for goal --> 25 but no goals so --> 21
        print(state.shape)
        image = tf.slice(X, [0, stateSize], [-1, size])
        print(image.shape)
        image = tf.reshape(image, IMG_SHAPE)
        print(image.shape)
        print(image.shape)
        image = tf.cast(image, tf.float32) / 255.
        print(image.shape)
        print(image)
        a = tf.placeholder(tf.float32, shape=(None, pixel, pixel, 3))
        b = tf.placeholder(tf.float32, shape=(None, 21))

        ## create the models
        base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                       include_top=False,
                                                       weights='imagenet')
        base_model.trainable = False
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        prediction_layer = keras.layers.Dense(16)  # converts to just yes no, but I want to know where in image
        model = tf.keras.Sequential([
            base_model,
            global_average_layer,
            prediction_layer
        ])

        second_model = tf.keras.Sequential([
            base_model,
            global_average_layer,
            prediction_layer
        ])
        camInput = tf.keras.layers.Input(shape=(pixel, pixel, 3), name="camera_input")
        camInput.shape
        stateInput = tf.keras.layers.Input(shape=(stateSize,), name="state_input")
        stateDense1 = tf.keras.layers.Dense(128)(stateInput)  # (b)  #(stateInput)
        stateDense2 = tf.keras.layers.Dense(256)(stateDense1)
        stateDense3 = tf.keras.layers.Dense(256)(stateDense2)

        with tf.variable_scope("convnet"):
            outputs = tf.placeholder(tf.float32, [None, 2])
            feed_dict = {a: image, b: state}
            cameraMidInput = second_model.output  # a or camInputs
            merged = tf.keras.layers.concatenate([stateDense3, cameraMidInput])
            mergedDense1 = tf.keras.layers.Dense(512)(merged)
            mergedDense2 = tf.keras.layers.Dense(256)(mergedDense1)
            mergedDense3 = tf.keras.layers.Dense(128)(mergedDense2)
            model = tf.keras.Model(inputs=[stateInput, camInput],
                                   outputs=[mergedDense3, cameraMidInput, stateDense3])
            model.summary()
            with tf.Session() as sess:
                out = tf.sess.run(feed_dict={a: image, b: state})
            return out

    return network_fn


@register("dual")
def dual(num_layers_1=2, num_layers_2=4, num_hidden_1=64, num_hidden_2=(9 + 64), activation=tf.tanh, layer_norm=False,
         convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)]):
    def network_fn(X):
        # cnn test
        pixel = 160
        size = pixel * pixel * 3
        IMG_SHAPE = (pixel, pixel, 3)
        stateSize = 21
        # https://www.tensorflow.org/tutorials/images/transfer_learning
        # This feature extractor converts each 160x160x3 image into a 5x5x1280 block of features. See what it does to the example batch of images:
        # Create the base model from the pre-trained model MobileNet V2

        state = tf.slice(X, [0, 0], [-1, stateSize])  # 21 state plus 4 for goal --> 25 but no goals so --> 21
        print(state.shape)
        image = tf.slice(X, [0, stateSize], [-1, size])

        print(image.shape)
        # image = tf.reshape(image, [pixel, pixel, 3, -1])
        image = tf.reshape(image, [pixel, pixel, 3])
        print(image.shape)
        # image = tf.transpose(image, [3, 0, 1, 2])  # reshape for NCHW format
        # image = tf.transpose(image, [0, 1, 2])  # reshape for NCHW format
        # image = tf.transpose(image, [1, 3, 2, 0])  # reshape for NCHW format
        print(image.shape)
        image = tf.cast(image, tf.float32) / 255.
        print(image.shape)
        print(image)

        ## create the models
        base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                       include_top=False,
                                                       weights='imagenet')
        base_model.trainable = False
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        prediction_layer = keras.layers.Dense(16)  # converts to just yes no, but I want to know where in image
        model = tf.keras.Sequential([
            base_model,
            global_average_layer,
            prediction_layer
        ])

        base_model.trainable = False
        # cameraMidInput = prediction_layer

        second_model = tf.keras.Sequential([
            base_model,
            global_average_layer,
            prediction_layer
        ])
        camInput = tf.keras.layers.Input(shape=IMG_SHAPE, name="camera_input")

        stateInput = tf.keras.layers.Input(shape=(stateSize,), name="state_input")
        stateDense1 = tf.keras.layers.Dense(128)(stateInput)  # (b)  #(stateInput)
        stateDense2 = tf.keras.layers.Dense(256)(stateDense1)
        stateDense3 = tf.keras.layers.Dense(256)(stateDense2)

        ## last part

        # IMAGE CNN
        with tf.variable_scope("convnet"):
            # feature_batch = base_model(image)  # if 160x160x3
            # print("feature batch! ", feature_batch)  # should be 5x5x1280 or something
            # global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
            # feature_batch_average = global_average_layer(feature_batch)
            # print("prints if class is present ", feature_batch_average.shape)  # flattens the above into 1280 vector
            # prediction_layer = keras.layers.Dense(16)  # converts to just yes no, but I want to know where in image
            # prediction_batch = prediction_layer(feature_batch_average)
            # print(prediction_batch.shape)  # converts to yes or no  (should be batch size, 9)
            # cameraMidInput = prediction_layer
            # base_model.summary()
            # camNN = tf.keras.Sequential([base_model, global_average_layer])

            a = tf.placeholder(tf.float32, shape=(160, 160, 3))
            b = tf.placeholder(tf.float32, shape=(21,))

            outputs = tf.placeholder(tf.float32, [None, 2])
            feed_dict = {a: image, b: state}
            cameraMidInput = second_model.predict(camInput, steps=1)

            # stateMidInput = output3

            merged = tf.keras.layers.concatenate([stateDense3, cameraMidInput])
            mergedDense1 = tf.keras.layers.Dense(512)(merged)
            mergedDense2 = tf.keras.layers.Dense(256)(mergedDense1)
            mergedDense3 = tf.keras.layers.Dense(128)(mergedDense2)

            model = tf.keras.Model(inputs=[stateInput, camInput, merged], outputs=[mergedDense3])
            model.summary()
            # return model
            # out = tf.sess.run(feed_dict = {a: image, b: state})
            out = model.predict([state, image])
            return out

            # concat = tf.keras.layers.concatenate()([stateMidInput, cameraMidInput])
            # model = tf.keras.Model(inputs=[state, image], outputs=[finalOut])

            """
             camInput = tf.keras.layers.Input(shape=IMG_SHAPE, name="camera_input")
            # camInput = base_model.input
             postPreTrainedModel = base_model.predict(camInput, steps=1)  #(a)#  # base_model.outputs()(camInput)  # (2280, )
             camPool = tf.keras.layers.GlobalAveragePooling2D()(postPreTrainedModel)
             camDense1 = tf.keras.layers.Dense(16)(camPool)
             """

            """ 
             camNN = tf.keras.Sequential([
                 base_model,
                 global_average_layer,
                 prediction_layer
             ])
             """

            # STATE MLP
            """
            # STATE MLP
            for i in range(num_layers_1):
            h1 = fc(state, 'mlp_fc1{}'.format(i), nh=num_hidden_1, init_scale=np.sqrt(2))
            if layer_norm:
                h1 = tf.contrib.layers.layer_norm(h1, center=True, scale=True)
            h1 = activation(h1)
        # print("\n\nMLP output: ", h1)
        """
        """
        out = tf.cast(image, tf.float32) / 255.
        with tf.variable_scope("convnet"):
            for num_outputs, kernel_size, stride in convs:
                out = tf.contrib.layers.convolution2d(out,
                                                      num_outputs=num_outputs,
                                                      kernel_size=kernel_size,
                                                      stride=stride,
                                                      activation_fn=tf.nn.relu,
                                                      )
        # print("before conv to fc", out)
        h2 = conv_to_fc(out)
        out = h2
        # h2 = activation(h2)
        # print("CNN output: ", h2)
        
        """
        # h2 = prediction_batch
        """
        # COMBINE MLP + CNN
        h3 = tf.concat([h1, h2], 1)
        # print("conc out: ", h3)

        for i in range(num_layers_2):
            h3 = fc(h3, 'mlp_fc2{}'.format(i), nh=num_hidden_2, init_scale=np.sqrt(2))
            if layer_norm:
                h3 = tf.contrib.layers.layer_norm(h3, center=True, scale=True)
            h3 = activation(h3)

        # print("\n\nFinal output: ", h3)
        """
        # return out

    return network_fn


def tamirCustomCNN(unscaled_images, **conv_kwargs):
    """
    CNN using video combining with states for FCNN
    """
    imgDim = 28 * 28 * 3
    states = unscaled_images[imgDim:]
    unscaled_images = unscaled_images[:imgDim]
    scaled_images = tf.cast(unscaled_images, tf.float32) / 255.
    activ = tf.nn.relu
    h = activ(conv(scaled_images, 'c1', nf=32, rf=4, stride=1, init_scale=np.sqrt(2),
                   **conv_kwargs))
    h2 = activ(conv(h, 'c2', nf=64, rf=4, stride=1, init_scale=np.sqrt(2), **conv_kwargs))
    h3 = activ(conv(h2, 'c3', nf=64, rf=2, stride=1, init_scale=np.sqrt(2), **conv_kwargs))
    h3 = conv_to_fc(h3)
    CNNOut = activ(fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2)))
    x = states
    x = tf.layers.flatten(X)
    concx = np.concatenate((x, CNNOut))

    h = tf.layers.flatten(concx)
    activ2 = tf.tanh
    num_layers = 5
    num_hidden = 80
    for i in range(num_layers):
        h = fc(h, 'mlp_fc{}'.format(i), nh=num_hidden, init_scale=np.sqrt(2))
        h = activ2(h)
    return h



def build_impala_cnn(unscaled_images, depths=[16, 32, 32], **conv_kwargs):
    """
    Model used in the paper "IMPALA: Scalable Distributed Deep-RL with
    Importance Weighted Actor-Learner Architectures" https://arxiv.org/abs/1802.01561
    """

    layer_num = 0

    def get_layer_num_str():
        nonlocal layer_num
        num_str = str(layer_num)
        layer_num += 1
        return num_str

    def conv_layer(out, depth):
        return tf.layers.conv2d(out, depth, 3, padding='same', name='layer_' + get_layer_num_str())

    def residual_block(inputs):
        depth = inputs.get_shape()[-1].value

        out = tf.nn.relu(inputs)

        out = conv_layer(out, depth)
        out = tf.nn.relu(out)
        out = conv_layer(out, depth)
        return out + inputs

    def conv_sequence(inputs, depth):
        out = conv_layer(inputs, depth)
        out = tf.layers.max_pooling2d(out, pool_size=3, strides=2, padding='same')
        out = residual_block(out)
        out = residual_block(out)
        return out

    out = tf.cast(unscaled_images, tf.float32) / 255.

    for depth in depths:
        out = conv_sequence(out, depth)

    out = tf.layers.flatten(out)
    out = tf.nn.relu(out)
    out = tf.layers.dense(out, 256, activation=tf.nn.relu, name='layer_' + get_layer_num_str())

    return out





@register("tamir")
def cnn(**conv_kwargs):
    def network_fn(X):
        return tamirCustomCNN(X, **conv_kwargs)

    return network_fn


@register("impala_cnn")
def impala_cnn(**conv_kwargs):
    def network_fn(X):
        return build_impala_cnn(X)

    return network_fn


@register("lstm")
def lstm(nlstm=128, layer_norm=False):
    """
    Builds LSTM (Long-Short Term Memory) network to be used in a policy.
    Note that the resulting function returns not only the output of the LSTM
    (i.e. hidden state of lstm for each step in the sequence), but also a dictionary
    with auxiliary tensors to be set as policy attributes.

    Specifically,
        S is a placeholder to feed current state (LSTM state has to be managed outside policy)
        M is a placeholder for the mask (used to mask out observations after the end of the episode, but can be used for other purposes too)
        initial_state is a numpy array containing initial lstm state (usually zeros)
        state is the output LSTM state (to be fed into S at the next call)


    An example of usage of lstm-based policy can be found here: common/tests/test_doc_examples.py/test_lstm_example

    Parameters:
    ----------

    nlstm: int          LSTM hidden state size

    layer_norm: bool    if True, layer-normalized version of LSTM is used

    Returns:
    -------

    function that builds LSTM with a given input tensor / placeholder
    """

    def network_fn(X, nenv=1):
        nbatch = X.shape[0]
        nsteps = nbatch // nenv

        h = tf.layers.flatten(X)

        M = tf.placeholder(tf.float32, [nbatch])  # mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, 2 * nlstm])  # states

        xs = batch_to_seq(h, nenv, nsteps)
        ms = batch_to_seq(M, nenv, nsteps)

        if layer_norm:
            h5, snew = utils.lnlstm(xs, ms, S, scope='lnlstm', nh=nlstm)
        else:
            h5, snew = utils.lstm(xs, ms, S, scope='lstm', nh=nlstm)

        h = seq_to_batch(h5)
        initial_state = np.zeros(S.shape.as_list(), dtype=float)

        return h, {'S': S, 'M': M, 'state': snew, 'initial_state': initial_state}

    return network_fn







@register("conv_only")
def conv_only(convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)], **conv_kwargs):
    '''
    convolutions-only net

    Parameters:
    ----------

    conv:       list of triples (filter_number, filter_size, stride) specifying parameters for each layer.

    Returns:

    function that takes tensorflow tensor as input and returns the output of the last convolutional layer

    '''

    def network_fn(X):
        out = tf.cast(X, tf.float32) / 255.
        with tf.variable_scope("convnet"):
            for num_outputs, kernel_size, stride in convs:
                out = tf.contrib.layers.convolution2d(out,
                                                      num_outputs=num_outputs,
                                                      kernel_size=kernel_size,
                                                      stride=stride,
                                                      activation_fn=tf.nn.relu,
                                                      **conv_kwargs)

        return out

    return network_fn


def _normalize_clip_observation(x, clip_range=[-5.0, 5.0]):
    rms = RunningMeanStd(shape=x.shape[1:])
    norm_x = tf.clip_by_value((x - rms.mean) / rms.std, min(clip_range), max(clip_range))
    return norm_x, rms


def get_network_builder(name):
    """
    If you want to register your own network outside models.py, you just need:

    Usage Example:
    -------------
    from baselines.common.models import register
    @register("your_network_name")
    def your_network_define(**net_kwargs):
        ...
        return network_fn

    """
    if callable(name):
        return name
    elif name in mapping:
        return mapping[name]
    else:
        raise ValueError('Unknown network type: {}'.format(name))

