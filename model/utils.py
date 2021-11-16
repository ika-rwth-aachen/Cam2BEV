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

import os
import sys
import numpy as np
import random
import tensorflow as tf
import cv2
import xml.etree.ElementTree as xmlET
from tqdm import tqdm
import importlib


def abspath(path):

    return os.path.abspath(os.path.expanduser(path))


def get_files_in_folder(folder):

    return sorted([os.path.join(folder, f) for f in os.listdir(folder)])



def sample_list(*ls, n_samples, replace=False):

    n_samples = min(len(ls[0]), n_samples)
    idcs = np.random.choice(np.arange(0, len(ls[0])), n_samples, replace=replace)
    samples = zip([np.take(l, idcs) for l in ls])
    return samples, idcs


def load_module(module_file):

    name = os.path.splitext(os.path.basename(module_file))[0]
    dir = os.path.dirname(module_file)
    sys.path.append(dir)
    spec = importlib.util.spec_from_file_location(name, module_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def load_image(filename):

    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def load_image_op(filename):

    img = tf.io.read_file(filename)
    img = tf.image.decode_png(img, channels=3)
    return img


def resize_image(img, shape, interpolation=cv2.INTER_CUBIC):

    # resize relevant image axis to length of corresponding target axis while preserving aspect ratio
    axis = 0 if float(shape[0]) / float(img.shape[0]) > float(shape[1]) / float(img.shape[1]) else 1
    factor = float(shape[axis]) / float(img.shape[axis])
    img = cv2.resize(img, (0,0), fx=factor, fy=factor, interpolation=interpolation)

    # crop other image axis to match target shape
    center = img.shape[int(not axis)] / 2.0
    step = shape[int(not axis)] / 2.0
    left = int(center-step)
    right = int(center+step)
    if axis == 0:
        img = img[:, left:right]
    else:
        img = img[left:right, :]

    return img


def resize_image_op(img, fromShape, toShape, cropToPreserveAspectRatio=True, interpolation=tf.image.ResizeMethod.BICUBIC):

    if not cropToPreserveAspectRatio:
        img = tf.image.resize(img, toShape, method=interpolation)

    else:

        # first crop to match target aspect ratio
        fx = toShape[1] / fromShape[1]
        fy = toShape[0] / fromShape[0]
        relevantAxis = 0 if fx < fy else 1
        if relevantAxis == 0:
            crop = fromShape[0] * toShape[1] / toShape[0]
            img = tf.image.crop_to_bounding_box(img, 0, int((fromShape[1] - crop) / 2), fromShape[0], int(crop))
        else:
            crop = fromShape[1] * toShape[0] / toShape[1]
            img = tf.image.crop_to_bounding_box(img, int((fromShape[0] - crop) / 2), 0, int(crop), fromShape[1])

        # then resize to target shape
        img = tf.image.resize(img, toShape, method=interpolation)

    return img


def one_hot_encode_image(image, palette):

    one_hot_map = []

    # find instances of class colors and append layer to one-hot-map
    for class_colors in palette:
        class_map = np.zeros(image.shape[0:2], dtype=bool)
        for color in class_colors:
            class_map = class_map | (image == color).all(axis=-1)
        one_hot_map.append(class_map)

    # finalize one-hot-map
    one_hot_map = np.stack(one_hot_map, axis=-1)
    one_hot_map = one_hot_map.astype(np.float32)

    return one_hot_map


def one_hot_encode_image_op(image, palette):

    one_hot_map = []

    for class_colors in palette:

        class_map = tf.zeros(image.shape[0:2], dtype=tf.int32)

        for color in class_colors:
            # find instances of color and append layer to one-hot-map
            class_map = tf.bitwise.bitwise_or(class_map, tf.cast(tf.reduce_all(tf.equal(image, color), axis=-1), tf.int32))
        one_hot_map.append(class_map)

    # finalize one-hot-map
    one_hot_map = tf.stack(one_hot_map, axis=-1)
    one_hot_map = tf.cast(one_hot_map, tf.float32)

    return one_hot_map


def one_hot_decode_image(one_hot_image, palette):

    # create empty image with correct dimensions
    height, width = one_hot_image.shape[0:2]
    depth = palette[0][0].size
    image = np.zeros([height, width, depth])

    # reduce all layers of one-hot-encoding to one layer with indices of the classes
    map_of_classes = one_hot_image.argmax(2)

    for idx, class_colors in enumerate(palette):
        # fill image with corresponding class colors
        image[np.where(map_of_classes == idx)] = class_colors[0]

    image = image.astype(np.uint8)

    return image


def parse_convert_xml(conversion_file_path):

    defRoot = xmlET.parse(conversion_file_path).getroot()

    one_hot_palette = []
    class_list = []
    for idx, defElement in enumerate(defRoot.findall("SLabel")):
        from_color = np.fromstring(defElement.get("fromColour"), dtype=int, sep=" ")
        to_class = np.fromstring(defElement.get("toValue"), dtype=int, sep=" ")
        if to_class in class_list:
             one_hot_palette[class_list.index(to_class)].append(from_color)
        else:
            one_hot_palette.append([from_color])
            class_list.append(to_class)

    return one_hot_palette


def get_class_distribution(folder, shape, palette):

    # get filepaths
    files = [os.path.join(folder, f) for f in os.listdir(folder) if not f.startswith(".")]

    n_classes = len(palette)

    def get_img(file, shape, interpolation=cv2.INTER_NEAREST, one_hot_reduce=False):
        img = load_image(file)
        img = resize_image(img, shape, interpolation)
        img = one_hot_encode_image(img, palette)
        return img

    px = shape[0] * shape[1]

    distribution = {}
    for k in range(n_classes):
        distribution[str(k)] = 0

    i = 0
    bar = tqdm(files)
    for f in bar:

        img = get_img(f, shape)

        classes = np.argmax(img, axis=-1)

        unique, counts = np.unique(classes, return_counts=True)

        occs = dict(zip(unique, counts))

        for k in range(n_classes):
            occ = occs[k] if k in occs.keys() else 0
            distribution[str(k)] = (distribution[str(k)] * i + occ / px) / (i+1)

        bar.set_postfix(distribution)

        i += 1

    return distribution


def weighted_categorical_crossentropy(weights):

    def wcce(y_true, y_pred):
        Kweights = tf.constant(weights)
        if not tf.is_tensor(y_pred): y_pred = tf.constant(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        return tf.keras.backend.categorical_crossentropy(y_true, y_pred) * tf.keras.backend.sum(y_true * Kweights, axis=-1)

    return wcce


class MeanIoUWithOneHotLabels(tf.keras.metrics.MeanIoU):

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.argmax(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)
