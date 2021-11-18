#!/usr/bin/env python

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

import importlib
import os
import sys
from datetime import datetime
import numpy as np
import cv2
import tensorflow as tf
import configargparse

import utils


# parse parameters from config file or CLI
parser = configargparse.ArgParser()
parser.add("-c", "--config", is_config_file=True, help="config file")
parser.add("-it", "--input-training",           type=str, required=True, nargs="+", help="directory/directories of input samples for training")
parser.add("-lt", "--label-training",           type=str, required=True,            help="directory of label samples for training")
parser.add("-nt", "--max-samples-training",     type=int, default=None,             help="maximum number of training samples")
parser.add("-iv", "--input-validation",         type=str, required=True, nargs="+", help="directory/directories of input samples for validation")
parser.add("-lv", "--label-validation",         type=str, required=True,            help="directory of label samples for validation")
parser.add("-nv", "--max-samples-validation",   type=int, default=None,             help="maximum number of validation samples")
parser.add("-is",  "--image-shape",           type=int, required=True, nargs=2, help="image dimensions (HxW) of inputs and labels for network")
parser.add("-ohi", "--one-hot-palette-input", type=str, required=True,          help="xml-file for one-hot-conversion of input images")
parser.add("-ohl", "--one-hot-palette-label", type=str, required=True,          help="xml-file for one-hot-conversion of label images")
parser.add("-m",    "--model",                      type=str,   required=True,              help="Python file defining the neural network")
parser.add("-uh",   "--unetxst-homographies",       type=str,   default=None,               help="Python file defining a list H of homographies to be used in uNetXST model")
parser.add("-e",    "--epochs",                     type=int,   required=True,              help="number of epochs for training")
parser.add("-bs",   "--batch-size",                 type=int,   required=True,              help="batch size for training")
parser.add("-lr",   "--learning-rate",              type=float, default=1e-4,               help="learning rate of Adam optimizer for training")
parser.add("-lw",   "--loss-weights",               type=float, default=None,   nargs="+",  help="factors for weighting classes differently in loss function")
parser.add("-esp",  "--early-stopping-patience",    type=int,   default=10,                 help="patience for early-stopping due to converged validation mIoU")
parser.add("-si",   "--save-interval",  type=int, default=5,        help="epoch interval between exports of the model")
parser.add("-o",    "--output-dir",     type=str, required=True,    help="output dir for TensorBoard and models")
parser.add("-mw",   "--model-weights",  type=str, default=None,     help="weights file of trained model for training continuation")
conf, unknown = parser.parse_known_args()


# determine absolute filepaths
conf.input_training         = [utils.abspath(path) for path in conf.input_training]
conf.label_training         = utils.abspath(conf.label_training)
conf.input_validation       = [utils.abspath(path) for path in conf.input_validation]
conf.label_validation       = utils.abspath(conf.label_validation)
conf.one_hot_palette_input  = utils.abspath(conf.one_hot_palette_input)
conf.one_hot_palette_label  = utils.abspath(conf.one_hot_palette_label)
conf.model                  = utils.abspath(conf.model)
conf.unetxst_homographies   = utils.abspath(conf.unetxst_homographies) if conf.unetxst_homographies is not None else conf.unetxst_homographies
conf.model_weights          = utils.abspath(conf.model_weights) if conf.model_weights is not None else conf.model_weights
conf.output_dir             = utils.abspath(conf.output_dir)


# load network architecture module
architecture = utils.load_module(conf.model)


# get max_samples_training random training samples
n_inputs = len(conf.input_training)
files_train_input = [utils.get_files_in_folder(folder) for folder in conf.input_training]
files_train_label = utils.get_files_in_folder(conf.label_training)
_, idcs = utils.sample_list(files_train_label, n_samples=conf.max_samples_training)
files_train_input = [np.take(f, idcs) for f in files_train_input]
files_train_label = np.take(files_train_label, idcs)
image_shape_original_input = utils.load_image(files_train_input[0][0]).shape[0:2]
image_shape_original_label = utils.load_image(files_train_label[0]).shape[0:2]
print(f"Found {len(files_train_label)} training samples")

# get max_samples_validation random validation samples
files_valid_input = [utils.get_files_in_folder(folder) for folder in conf.input_validation]
files_valid_label = utils.get_files_in_folder(conf.label_validation)
_, idcs = utils.sample_list(files_valid_label, n_samples=conf.max_samples_validation)
files_valid_input = [np.take(f, idcs) for f in files_valid_input]
files_valid_label = np.take(files_valid_label, idcs)
print(f"Found {len(files_valid_label)} validation samples")


# parse one-hot-conversion.xml
conf.one_hot_palette_input = utils.parse_convert_xml(conf.one_hot_palette_input)
conf.one_hot_palette_label = utils.parse_convert_xml(conf.one_hot_palette_label)
n_classes_input = len(conf.one_hot_palette_input)
n_classes_label = len(conf.one_hot_palette_label)


# build dataset pipeline parsing functions
def parse_sample(input_files, label_file):
    # parse and process input images
    inputs = []
    for inp in input_files:
        inp = utils.load_image_op(inp)
        inp = utils.resize_image_op(inp, image_shape_original_input, conf.image_shape, interpolation=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        inp = utils.one_hot_encode_image_op(inp, conf.one_hot_palette_input)
        inputs.append(inp)
    inputs = inputs[0] if n_inputs == 1 else tuple(inputs)
    # parse and process label image
    label = utils.load_image_op(label_file)
    label = utils.resize_image_op(label, image_shape_original_label, conf.image_shape, interpolation=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    label = utils.one_hot_encode_image_op(label, conf.one_hot_palette_label)
    return inputs, label

# build training data pipeline
dataTrain = tf.data.Dataset.from_tensor_slices((tuple(files_train_input), files_train_label))
dataTrain = dataTrain.shuffle(buffer_size=conf.max_samples_training, reshuffle_each_iteration=True)
dataTrain = dataTrain.map(parse_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE)
dataTrain = dataTrain.batch(conf.batch_size, drop_remainder=True)
dataTrain = dataTrain.repeat(conf.epochs)
dataTrain = dataTrain.prefetch(1)
print("Built data pipeline for training")

# build validation data pipeline
dataValid = tf.data.Dataset.from_tensor_slices((tuple(files_valid_input), files_valid_label))
dataValid = dataValid.map(parse_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE)
dataValid = dataValid.batch(1)
dataValid = dataValid.repeat(conf.epochs)
dataValid = dataValid.prefetch(1)
print("Built data pipeline for validation")


# build model
if conf.unetxst_homographies is not None:
  uNetXSTHomographies = utils.load_module(conf.unetxst_homographies)
  model = architecture.get_network((conf.image_shape[0], conf.image_shape[1], n_classes_input), n_classes_label, n_inputs=n_inputs, thetas=uNetXSTHomographies.H)
else:
  model = architecture.get_network((conf.image_shape[0], conf.image_shape[1], n_classes_input), n_classes_label)
if conf.model_weights is not None:
  model.load_weights(conf.model_weights)
optimizer = tf.keras.optimizers.Adam(learning_rate=conf.learning_rate)
if conf.loss_weights is not None:
    loss = utils.weighted_categorical_crossentropy(conf.loss_weights)
else:
    loss = tf.keras.losses.CategoricalCrossentropy()
metrics = [tf.keras.metrics.CategoricalAccuracy(), utils.MeanIoUWithOneHotLabels(num_classes=n_classes_label)]
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
print(f"Compiled model {os.path.basename(conf.model)}")


# create output directories
model_output_dir = os.path.join(conf.output_dir, datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
tensorboard_dir = os.path.join(model_output_dir, "TensorBoard")
checkpoint_dir  = os.path.join(model_output_dir, "Checkpoints")
if not os.path.exists(tensorboard_dir):
    os.makedirs(tensorboard_dir)
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)


# create callbacks to be called after each epoch
n_batches_train = len(files_train_label) // conf.batch_size
n_batches_valid = len(files_valid_label)
tensorboard_cb      = tf.keras.callbacks.TensorBoard(tensorboard_dir, update_freq="epoch", profile_batch=0)
checkpoint_cb       = tf.keras.callbacks.ModelCheckpoint(os.path.join(checkpoint_dir, "e{epoch:03d}_weights.hdf5"), save_freq=n_batches_train*conf.save_interval, save_weights_only=True)
best_checkpoint_cb  = tf.keras.callbacks.ModelCheckpoint(os.path.join(checkpoint_dir, "best_weights.hdf5"), save_best_only=True, monitor="val_mean_io_u_with_one_hot_labels", mode="max", save_weights_only=True)
early_stopping_cb   = tf.keras.callbacks.EarlyStopping(monitor="val_mean_io_u_with_one_hot_labels", mode="max", patience=conf.early_stopping_patience, verbose=1)
callbacks = [tensorboard_cb, checkpoint_cb, best_checkpoint_cb, early_stopping_cb]


# start training
print("Starting training...")
model.fit(dataTrain,
          epochs=conf.epochs, steps_per_epoch=n_batches_train,
          validation_data=dataValid, validation_freq=1, validation_steps=n_batches_valid,
          callbacks=callbacks)
