# ==============================================================================
# Copyright (c) 2017 Octavio
#
# This work is licensed under the terms of the MIT license.
# For a copy, see https://opensource.org/licenses/MIT.
# ==============================================================================

# ==============================================================================
# https://github.com/oarriaga/STN.keras
# Commit: 9ab0160777d33af768da10d1f201c3c321cbac47
# Modifications:
#   - import fixes
#   - single file
#   - projective transform support
#   - add Model containing a SpatialTransformer
#
# TODO: throws error: "topological sort failed"
#   - training works, error appears only in validation mode
#   - validation works if theta is constant
#   - might be related to a concat-op
#   - https://github.com/tensorflow/tensorflow/issues/24816
# ==============================================================================


import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, Activation, MaxPooling2D, Flatten, Conv2D, Dense, Multiply
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K


def K_meshgrid(x, y):
    return tf.meshgrid(x, y)


def K_linspace(start, stop, num):
    return tf.linspace(start, stop, num)


class BilinearInterpolation(Layer):
    """Performs bilinear interpolation as a keras layer
    References
    ----------
    [1]  Spatial Transformer Networks, Max Jaderberg, et al.
    [2]  https://github.com/skaae/transformer_network
    [3]  https://github.com/EderSantana/seya
    """

    def __init__(self, output_size, **kwargs):
        self.output_size = output_size
        super(BilinearInterpolation, self).__init__(**kwargs)

    def get_config(self):
        return {
            'output_size': self.output_size,
        }

    def compute_output_shape(self, input_shapes):
        height, width = self.output_size
        num_channels = input_shapes[0][-1]
        return (None, height, width, num_channels)

    def call(self, tensors, mask=None):
        X, transformation = tensors
        output = self._transform(X, transformation, self.output_size)
        return output

    def _interpolate(self, image, sampled_grids, output_size):

        batch_size = K.shape(image)[0]
        height = K.shape(image)[1]
        width = K.shape(image)[2]
        num_channels = K.shape(image)[3]

        x = K.cast(K.flatten(sampled_grids[:, 0:1, :]), dtype='float32')
        y = K.cast(K.flatten(sampled_grids[:, 1:2, :]), dtype='float32')

        x = .5 * (x + 1.0) * K.cast(width, dtype='float32')
        y = .5 * (y + 1.0) * K.cast(height, dtype='float32')

        x0 = K.cast(x, 'int32')
        x1 = x0 + 1
        y0 = K.cast(y, 'int32')
        y1 = y0 + 1

        max_x = int(K.int_shape(image)[2] - 1)
        max_y = int(K.int_shape(image)[1] - 1)

        x0 = K.clip(x0, 0, max_x)
        x1 = K.clip(x1, 0, max_x)
        y0 = K.clip(y0, 0, max_y)
        y1 = K.clip(y1, 0, max_y)

        pixels_batch = K.arange(0, batch_size) * (height * width)
        pixels_batch = K.expand_dims(pixels_batch, axis=-1)
        flat_output_size = output_size[0] * output_size[1]
        base = K.repeat_elements(pixels_batch, flat_output_size, axis=1)
        base = K.flatten(base)

        # base_y0 = base + (y0 * width)
        base_y0 = y0 * width
        base_y0 = base + base_y0
        # base_y1 = base + (y1 * width)
        base_y1 = y1 * width
        base_y1 = base_y1 + base

        indices_a = base_y0 + x0
        indices_b = base_y1 + x0
        indices_c = base_y0 + x1
        indices_d = base_y1 + x1

        flat_image = K.reshape(image, shape=(-1, num_channels))
        flat_image = K.cast(flat_image, dtype='float32')
        pixel_values_a = K.gather(flat_image, indices_a)
        pixel_values_b = K.gather(flat_image, indices_b)
        pixel_values_c = K.gather(flat_image, indices_c)
        pixel_values_d = K.gather(flat_image, indices_d)

        x0 = K.cast(x0, 'float32')
        x1 = K.cast(x1, 'float32')
        y0 = K.cast(y0, 'float32')
        y1 = K.cast(y1, 'float32')

        area_a = K.expand_dims(((x1 - x) * (y1 - y)), 1)
        area_b = K.expand_dims(((x1 - x) * (y - y0)), 1)
        area_c = K.expand_dims(((x - x0) * (y1 - y)), 1)
        area_d = K.expand_dims(((x - x0) * (y - y0)), 1)

        values_a = area_a * pixel_values_a
        values_b = area_b * pixel_values_b
        values_c = area_c * pixel_values_c
        values_d = area_d * pixel_values_d
        return values_a + values_b + values_c + values_d

    def _make_regular_grids(self, batch_size, height, width):
        # making a single regular grid
        x_linspace = K_linspace(-1., 1., width)
        y_linspace = K_linspace(-1., 1., height)
        x_coordinates, y_coordinates = K_meshgrid(x_linspace, y_linspace)
        x_coordinates = K.flatten(x_coordinates)
        y_coordinates = K.flatten(y_coordinates)
        ones = K.ones_like(x_coordinates)
        grid = K.concatenate([x_coordinates, y_coordinates, ones], 0)

        # repeating grids for each batch
        grid = K.flatten(grid)
        grids = K.tile(grid, K.stack([batch_size]))
        return K.reshape(grids, (batch_size, 3, height * width))

    def _transform(self, X, transformation, output_size):
        
        batch_size, num_channels = K.shape(X)[0], X.shape[3]

        transformation = K.reshape(transformation, shape=(batch_size, 3, 3))

        # create regular grid (-1,1)x(-1,1)
        regular_grids = self._make_regular_grids(batch_size, *output_size)

        # transform regular grid
        sampled_grids = K.batch_dot(transformation, regular_grids)

        # homogeneous coords: divide by 3rd component w
        w = tf.math.reciprocal(sampled_grids[:, 2, :])
        w = tf.reshape(w, (batch_size, 1, w.shape[-1]))
        w = tf.tile(w, [1, 2, 1])
        sampled_grids = Multiply()([sampled_grids[:, 0:2, :], w])

        # interpolate image
        interpolated_image = self._interpolate(X, sampled_grids, output_size)
        new_shape = (batch_size, output_size[0], output_size[1], num_channels)
        interpolated_image = K.reshape(interpolated_image, new_shape)

        return interpolated_image


def STN_example_model(input_shape=(60, 60, 1), sampling_size=(30, 30), num_classes=10):

    image = Input(shape=input_shape)
    locnet = MaxPooling2D(pool_size=(2, 2))(image)
    locnet = Conv2D(20, (5, 5))(locnet)
    locnet = MaxPooling2D(pool_size=(2, 2))(locnet)
    locnet = Conv2D(20, (5, 5))(locnet)
    locnet = Flatten()(locnet)
    locnet = Dense(50)(locnet)
    locnet = Activation('relu')(locnet)
    def get_initial_weights(output_size):
        b = np.zeros((2, 3), dtype='float32')
        b[0, 0] = 1
        b[1, 1] = 1
        W = np.zeros((output_size, 6), dtype='float32')
        weights = [W, b.flatten()]
        return weights
    weights = get_initial_weights(50)
    locnet = Dense(6, weights=weights)(locnet)
    x = BilinearInterpolation(sampling_size)([image, locnet])
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(32, (3, 3))(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(256)(x)
    x = Activation('relu')(x)
    x = Dense(num_classes)(x)
    x = Activation('softmax')(x)
    return Model(inputs=image, outputs=x)


def _spatial_transformer(input, output_shape, theta_init=np.eye(3), theta_const=False, loc_downsample=3, dense_units=20, filters=16, kernel_size=(3,3), activation=tf.nn.relu, dense_reg=0.0):

    theta_init = theta_init.flatten().astype(np.float32)

    if not theta_const:

        t = input

        # initialize transform to identity
        init_weights = [np.zeros((dense_units, 9), dtype=np.float32), theta_init]

        # localization network
        for d in range(loc_downsample):
            t = Conv2D(filters=filters, kernel_size=kernel_size, padding="same", activation=activation)(t)
            t = MaxPooling2D(pool_size=(2,2), padding="same")(t)
        t = Flatten()(t)
        t = Dense(dense_units)(t)

        k_reg = tf.keras.regularizers.l2(dense_reg) if dense_reg > 0 else None
        b_reg = tf.keras.regularizers.l2(dense_reg) if dense_reg > 0 else None
        theta = Dense(9, weights=init_weights, kernel_regularizer=k_reg, bias_regularizer=b_reg)(t) # transformation parameters

    else:

        theta = tf.tile(theta_init, tf.shape(input)[0:1])

    # transform feature map
    output = BilinearInterpolation(output_shape)([input, theta])

    return output


def SpatialTransformer(input_shape, output_shape, theta_init=np.eye(3), theta_const=False, loc_downsample=3, dense_units=20, filters=16, kernel_size=(3,3), activation=tf.nn.relu, dense_reg=0.0):

    input = Input(input_shape)
    output = _spatial_transformer(input, output_shape[0:2], theta_init, theta_const, loc_downsample, dense_units, filters, kernel_size, activation, dense_reg)

    return Model(input, output)
