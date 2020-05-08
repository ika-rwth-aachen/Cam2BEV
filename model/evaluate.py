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
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import tensorflow as tf
import configargparse

import utils


# parse parameters from config file or CLI
parser = configargparse.ArgParser()
parser.add("-c",    "--config", is_config_file=True, help="config file")
parser.add("-iv",   "--input-validation",       type=str, required=True, nargs="+", help="directory/directories of input samples for validation")
parser.add("-lv",   "--label-validation",       type=str, required=True,            help="directory of label samples for validation")
parser.add("-nv",   "--max-samples-validation", type=int, default=None,             help="maximum number of validation samples")
parser.add("-is",   "--image-shape",            type=int, required=True, nargs=2,   help="image dimensions (HxW) of inputs and labels for network")
parser.add("-ohi",  "--one-hot-palette-input",  type=str, required=True,            help="xml-file for one-hot-conversion of input images")
parser.add("-ohl",  "--one-hot-palette-label",  type=str, required=True,            help="xml-file for one-hot-conversion of label images")
parser.add("-cn",   "--class-names",            type=str, required=True, nargs="+", help="class names to annotate confusion matrix axes")
parser.add("-m",    "--model",                  type=str, required=True,            help="Python file defining the neural network")
parser.add("-uh",   "--unetxst-homographies",   type=str, default=None,             help="Python file defining a list H of homographies to be used in uNetXST model")
parser.add("-mw",   "--model-weights",          type=str, required=True,            help="weights file of trained model")
conf, unknown = parser.parse_known_args()


# determine absolute filepaths
conf.input_validation       = [utils.abspath(path) for path in conf.input_validation]
conf.label_validation       = utils.abspath(conf.label_validation)
conf.one_hot_palette_input  = utils.abspath(conf.one_hot_palette_input)
conf.one_hot_palette_label  = utils.abspath(conf.one_hot_palette_label)
conf.model                  = utils.abspath(conf.model)
conf.unetxst_homographies   = utils.abspath(conf.unetxst_homographies) if conf.unetxst_homographies is not None else conf.unetxst_homographies
conf.model_weights          = utils.abspath(conf.model_weights)


# load network architecture module
architecture = utils.load_module(conf.model)


# get max_samples_validation random validation samples
files_input = [utils.get_files_in_folder(folder) for folder in conf.input_validation]
files_label = utils.get_files_in_folder(conf.label_validation)
_, idcs = utils.sample_list(files_label, n_samples=conf.max_samples_validation)
files_input = [np.take(f, idcs) for f in files_input]
files_label = np.take(files_label, idcs)
n_inputs = len(conf.input_validation)
n_samples = len(files_label)
image_shape_original_input = utils.load_image(files_input[0][0]).shape[0:2]
image_shape_original_label = utils.load_image(files_label[0]).shape[0:2]
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


# evaluate confusion matrix
print("Evaluating confusion matrix ...")
confusion_matrix = np.zeros((n_classes_label, n_classes_label), dtype=np.int64)
for k in tqdm.tqdm(range(n_samples)):

    input_files = [files_input[i][k] for i in range(n_inputs)]
    label_file = files_label[k]

    # load sample
    inputs, label = parse_sample(input_files, label_file)

    # add batch dim
    if n_inputs > 1:
        inputs = [np.expand_dims(i, axis=0) for i in inputs]
    else:
        inputs = np.expand_dims(inputs, axis=0)

    # run prediction
    prediction = model.predict(inputs).squeeze()

    # compute confusion matrix
    label = np.argmax(label, axis=-1)
    prediction = np.argmax(prediction, axis=-1)
    sample_confusion_matrix = tf.math.confusion_matrix(label.flatten(), prediction.flatten(), num_classes=n_classes_label).numpy()

    # sum confusion matrix over dataset
    confusion_matrix += sample_confusion_matrix


# normalize confusion matrix rows (What percentage of class X has been predicted to be class Y?)
confusion_matrix_norm = confusion_matrix / np.sum(confusion_matrix, axis=1)[:, np.newaxis]


# compute per-class IoU
row_sum = np.sum(confusion_matrix, axis=0)
col_sum = np.sum(confusion_matrix, axis=1)
diag    = np.diag(confusion_matrix)
intersection = diag
union = row_sum + col_sum - diag
ious = intersection / union
iou = {}
for idx, v in enumerate(ious):
    iou[conf.class_names[idx]] = v


# print metrics
print("\nPer-class IoU:")
for k, v in iou.items():
    print(f"  {k}: {100*v:3.2f}%")
print("\nConfusion Matrix:")
print(confusion_matrix)
print("\nNormalized Confusion Matrix:")
print(confusion_matrix_norm)


# plot confusion matrix
confusion_matrix_df = pd.DataFrame(confusion_matrix_norm*100, conf.class_names, conf.class_names)
plt.figure(figsize=(8,8))
hm = sb.heatmap(confusion_matrix_df,
                annot=True,
                fmt=".2f",
                square=True,
                vmin=0,
                vmax=100,
                cbar_kws={"label": "%", "shrink": 0.8},
                cmap=plt.cm.Blues)
hm.set_xticklabels(hm.get_xticklabels(), rotation=30)
plt.ylabel("True Label")
plt.xlabel("Predicted Label")


# save confusion matrix and class ious to file and export plot
eval_folder = os.path.join(os.path.dirname(conf.model_weights), os.pardir, "Evaluation")
if not os.path.exists(eval_folder):
  os.makedirs(eval_folder)
filename = os.path.join(eval_folder, "confusion_matrix.txt")
np.savetxt(filename, confusion_matrix, fmt="%d")
filename = os.path.join(eval_folder, "class_iou.txt")
np.savetxt(filename, ious, fmt="%f")
filename = os.path.join(eval_folder, "confusion_matrix.pdf")
plt.savefig(filename, bbox_inches="tight")
