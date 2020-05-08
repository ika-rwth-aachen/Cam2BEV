# ==============================================================================
# MIT License
#
# Copyright 2020 Institute for Automotive Engineering of RWTH Aachen University.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense, Conv2D, Conv2DTranspose, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate

from third_party.spatial_transformer import SpatialTransformer


def encoder(input, udepth=3, filters1=8, kernel_size=(3,3), activation=tf.nn.relu, batch_norm=True, dropout=0.1):

    t = input
    encoder_layers = udepth * [None]

    # common parameters
    pool_size = (2,2)
    padding = "same"

    # layer creation with successive pooling
    for d in range(udepth):
        filters = (2**d) * filters1
        t = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, activation=activation)(t)
        t = BatchNormalization()(t) if batch_norm else t
        t = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, activation=activation)(t)
        t = encoder_layers[d] = BatchNormalization()(t) if batch_norm else t
        if d < (udepth - 1):
            t = MaxPooling2D(pool_size=pool_size, padding=padding)(t)
            t = Dropout(rate=dropout)(t) if dropout > 0 else t

    return encoder_layers


def joiner(list_of_encoder_layers, thetas, filters1=8, kernel_size=(3,3), activation=tf.nn.relu, batch_norm=True, double_skip_connection=False):

    n_inputs = len(list_of_encoder_layers)
    udepth = len(list_of_encoder_layers[0])
    encoder_layers = udepth * [None]

    for d in range(udepth):
        filters = (2**d) * filters1
        shape = list_of_encoder_layers[0][d].shape[1:]

        warped_maps = []
        for i in range(n_inputs): # use Spatial Transformer with constant homography transformation before concatenating
            # Problem w/ trainable theta: regularization necessary, huge loss, always went to loss=nan
            t = SpatialTransformer(shape, shape, theta_init=thetas[i], theta_const=True)(list_of_encoder_layers[i][d])
            warped_maps.append(t)
        t = Concatenate()(warped_maps) if n_inputs > 1 else warped_maps[0]
        t = Conv2D(filters=filters, kernel_size=kernel_size, padding="same", activation=activation)(t)
        t = BatchNormalization()(t) if batch_norm else t
        t = Conv2D(filters=filters, kernel_size=kernel_size, padding="same", activation=activation)(t)
        t = warped = BatchNormalization()(t) if batch_norm else t

        if not double_skip_connection:

            t = encoder_layers[d] = warped

        else:

            nonwarped_maps = []
            for i in range(n_inputs): # also concat non-warped maps
                t = list_of_encoder_layers[i][d]
                nonwarped_maps.append(t)
            t = Concatenate()(nonwarped_maps) if n_inputs > 1 else nonwarped_maps[0]
            t = Conv2D(filters=filters, kernel_size=kernel_size, padding="same", activation=activation)(t)
            t = BatchNormalization()(t) if batch_norm else t
            t = Conv2D(filters=filters, kernel_size=kernel_size, padding="same", activation=activation)(t)
            t = nonwarped = BatchNormalization()(t) if batch_norm else t

            # concat both
            t = Concatenate()([warped, nonwarped])
            t = Conv2D(filters=filters, kernel_size=kernel_size, padding="same", activation=activation)(t)
            t = BatchNormalization()(t) if batch_norm else t
            t = Conv2D(filters=filters, kernel_size=kernel_size, padding="same", activation=activation)(t)
            t = encoder_layers[d] = BatchNormalization()(t) if batch_norm else t

    return encoder_layers


def decoder(encoder_layers, udepth=3, filters1=8, kernel_size=(3,3), activation=tf.nn.relu, batch_norm=True, dropout=0.1):

    # start at lowest encoder layer
    t = encoder_layers[udepth-1]

    # common parameters
    strides = (2,2)
    padding = "same"

    # layer expansion symmetric to encoder
    for d in reversed(range(udepth-1)):
        filters = (2**d) * filters1
        t = Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(t)
        t = Concatenate()([encoder_layers[d], t])
        t = Dropout(rate=dropout)(t) if dropout > 0 else t
        t = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, activation=activation)(t)
        t = BatchNormalization()(t) if batch_norm else t
        t = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, activation=activation)(t)
        t = BatchNormalization()(t) if batch_norm else t

    return t


def get_network(input_shape, n_output_channels, n_inputs, thetas, 
                udepth = 5, 
                filters1 = 16, 
                kernel_size = (3,3), 
                activation = tf.nn.relu, 
                batch_norm = True, 
                dropout = 0.1,
                double_skip_connection = False):

    # build inputs
    inputs = [Input(input_shape) for i in range(n_inputs)]

    # encode all inputs separately
    list_of_encoder_layers = []
    for i in inputs:
        encoder_layers = encoder(i, udepth, filters1, kernel_size, activation, batch_norm, dropout)
        list_of_encoder_layers.append(encoder_layers)

    # fuse encodings of all inputs at all layers
    encoder_layers = joiner(list_of_encoder_layers, thetas, filters1, kernel_size, activation, batch_norm, double_skip_connection)

    # decode from bottom to top layer
    reconstruction = decoder(encoder_layers, udepth, filters1, kernel_size, activation, batch_norm, dropout)

    # build final prediction layer
    prediction = Conv2D(filters=n_output_channels, kernel_size=kernel_size, padding="same", activation=activation)(reconstruction)
    prediction = Activation("softmax")(prediction)

    return Model(inputs, prediction)
