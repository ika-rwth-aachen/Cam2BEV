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
import tqdm
import numpy as np
import cv2
import tensorflow as tf
import configargparse

import utils


# parse parameters from config file or CLI
parser = configargparse.ArgParser()
parser.add("-c",    "--config", is_config_file=True, help="config file")
parser.add("-ip",   "--input-testing",          type=str, required=True, nargs="+", help="directory/directories of input samples for testing")
parser.add("-np",   "--max-samples-testing",    type=int, default=None,             help="maximum number of testing samples")
parser.add("-is",   "--image-shape",            type=int, required=True, nargs=2,   help="image dimensions (HxW) of inputs and labels for network")
parser.add("-ohi",  "--one-hot-palette-input",  type=str, required=True,            help="xml-file for one-hot-conversion of input images")
parser.add("-ohl",  "--one-hot-palette-label",  type=str, required=True,            help="xml-file for one-hot-conversion of label images")
parser.add("-m",    "--model",                  type=str, required=True,            help="Python file defining the neural network")
parser.add("-uh",   "--unetxst-homographies",   type=str, default=None,             help="Python file defining a list H of homographies to be used in uNetXST model")
parser.add("-mw",   "--model-weights",          type=str, required=True,            help="weights file of trained model")
parser.add("-pd",   "--prediction-dir",         type=str, required=True,            help="output directory for storing predictions of testing data")
conf, unknown = parser.parse_known_args()


# determine absolute filepaths
conf.input_testing          = [utils.abspath(path) for path in conf.input_testing]
conf.one_hot_palette_input  = utils.abspath(conf.one_hot_palette_input)
conf.one_hot_palette_label  = utils.abspath(conf.one_hot_palette_label)
conf.model                  = utils.abspath(conf.model)
conf.unetxst_homographies   = utils.abspath(conf.unetxst_homographies) if conf.unetxst_homographies is not None else conf.unetxst_homographies
conf.model_weights          = utils.abspath(conf.model_weights)
conf.prediction_dir         = utils.abspath(conf.prediction_dir)


# load network architecture module
architecture = utils.load_module(conf.model)


# get max_samples_testing samples
files_input = [utils.get_files_in_folder(folder) for folder in conf.input_testing]
_, idcs = utils.sample_list(files_input[0], n_samples=conf.max_samples_testing)
files_input = [np.take(f, idcs) for f in files_input]
n_inputs = len(conf.input_testing)
n_samples = len(files_input[0])
image_shape_original = utils.load_image(files_input[0][0]).shape[0:2]
print(f"Found {n_samples} samples")


# parse one-hot-conversion.xml
conf.one_hot_palette_input = utils.parse_convert_xml(conf.one_hot_palette_input)
conf.one_hot_palette_label = utils.parse_convert_xml(conf.one_hot_palette_label)
n_classes_input = len(conf.one_hot_palette_input)
n_classes_label = len(conf.one_hot_palette_label)


# build model
if conf.unetxst_homographies is not None:
  uNetXSTHomographies = utils.load_module(conf.unetxst_homographies)
  model = architecture.get_network((conf.image_shape[0], conf.image_shape[1], n_classes_input), n_classes_label, n_inputs=n_inputs, thetas=uNetXSTHomographies.H)
else:
  model = architecture.get_network((conf.image_shape[0], conf.image_shape[1], n_classes_input), n_classes_label)
model.load_weights(conf.model_weights)
print(f"Reloaded model from {conf.model_weights}")


# build data parsing function
def parse_sample(input_files):
    # parse and process input images
    inputs = []
    for inp in input_files:
        inp = utils.load_image_op(inp)
        inp = utils.resize_image_op(inp, image_shape_original, conf.image_shape, interpolation=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        inp = utils.one_hot_encode_image_op(inp, conf.one_hot_palette_input)
        inputs.append(inp)
    inputs = inputs[0] if n_inputs == 1 else tuple(inputs)
    return inputs


# create output directory
if not os.path.exists(conf.prediction_dir):
    os.makedirs(conf.prediction_dir)


# run predictions
print(f"Running predictions and writing to {conf.prediction_dir} ...")
for k in tqdm.tqdm(range(n_samples)):

    input_files = [files_input[i][k] for i in range(n_inputs)]

    # load sample
    inputs = parse_sample(input_files)

    # add batch dim
    if n_inputs > 1:
        inputs = [np.expand_dims(i, axis=0) for i in inputs]
    else:
        inputs = np.expand_dims(inputs, axis=0)

    # run prediction
    prediction = model.predict(inputs).squeeze()

    # convert to output image
    prediction = utils.one_hot_decode_image(prediction, conf.one_hot_palette_label)

    # write to disk
    output_file = os.path.join(conf.prediction_dir, os.path.basename(files_input[0][k]))
    cv2.imwrite(output_file, cv2.cvtColor(prediction, cv2.COLOR_RGB2BGR))
