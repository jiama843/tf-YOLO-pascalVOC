from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from PIL import Image
from zipfile import ZipFile

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

tf.logging.set_verbosity(tf.logging.INFO)


def yolo_model(features, labels, mode):
    """Model function for YOLO."""

    """7x7x30 output"""

    # Input Layer, reshape to 448, 448, 3
    input_layer = tf.reshape(features, [-1, 448, 448, 1])
    # tf.Print(input_layer)

    print(input_layer)

    # Layer 1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=64,
        kernel_size=[7, 7],
        strides=(2, 2),
        padding="same",
        activation=tf.nn.relu)

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Layer 2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=192,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding="same",
        activation=tf.nn.relu)

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Layer 3
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=128,
        kernel_size=[1, 1],
        strides=(1, 1),
        padding="same",
        activation=tf.nn.relu)

    conv4 = tf.layers.conv2d(
        inputs=conv3,
        filters=256,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding="same",
        activation=tf.nn.relu)

    conv5 = tf.layers.conv2d(
        inputs=conv4,
        filters=256,
        kernel_size=[1, 1],
        strides=(1, 1),
        padding="same",
        activation=tf.nn.relu)

    conv6 = tf.layers.conv2d(
        inputs=conv5,
        filters=512,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding="same",
        activation=tf.nn.relu)

    pool3 = tf.layers.max_pooling2d(inputs=conv6, pool_size=[2, 2], strides=2)

    # Layer 4

    # conv7 and conv8 are run 4 times
    conv8 = pool3
    for i in range(1, 4):
        conv7 = tf.layers.conv2d(
            inputs=conv8,
            filters=256,
            kernel_size=[1, 1],
            strides=(1, 1),
            padding="same",
            activation=tf.nn.relu)

        conv8 = tf.layers.conv2d(
            inputs=conv7,
            filters=512,
            kernel_size=[3, 3],
            strides=(1, 1),
            padding="same",
            activation=tf.nn.relu)

    conv9 = tf.layers.conv2d(
        inputs=conv8,
        filters=512,
        kernel_size=[1, 1],
        strides=(1, 1),
        padding="same",
        activation=tf.nn.relu)

    conv10 = tf.layers.conv2d(
        inputs=conv9,
        filters=1024,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding="same",
        activation=tf.nn.relu)

    pool4 = tf.layers.max_pooling2d(inputs=conv10, pool_size=[2, 2], strides=2)

    # Layer 5

    # conv11 and conv12 are run 2 times
    conv12 = pool4
    for i in range(1, 2):
        conv11 = tf.layers.conv2d(
            inputs=conv12,
            filters=512,
            kernel_size=[1, 1],
            strides=(1, 1),
            padding="same",
            activation=tf.nn.relu)

        conv12 = tf.layers.conv2d(
            inputs=conv11,
            filters=1024,
            kernel_size=[3, 3],
            strides=(1, 1),
            padding="same",
            activation=tf.nn.relu)

    conv13 = tf.layers.conv2d(
        inputs=conv12,
        filters=1024,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding="same",
        activation=tf.nn.relu)

    conv14 = tf.layers.conv2d(
        inputs=conv13,
        filters=1024,
        kernel_size=[3, 3],
        strides=(2, 2),
        padding="same",
        activation=tf.nn.relu)

    # Layer 6

    conv15 = tf.layers.conv2d(
        inputs=conv14,
        filters=1024,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding="same",
        activation=tf.nn.relu)

    conv16 = tf.layers.conv2d(
        inputs=conv15,
        filters=1024,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding="same",
        activation=tf.nn.relu)

    # Layer 7 (Fully Connected)

    # print(conv14)

    inputflat1 = tf.reshape(conv16, [-1, 7 * 7 * 1024])
    dense1 = tf.layers.dense(inputs=inputflat1, units=4096, activation=tf.nn.relu)
    # dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Layer 8 (Fully Connected)

    dense2 = tf.layers.dense(inputs=dense1, units=1470, activation=tf.nn.relu)
    result = tf.reshape(dense2, [-1, 7, 7, 30])

    return result


def main(unused_argv):
    file_path = "../input/pascalvoc/VOC2012/*"

    filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(file_path + "*.jpg"))

    """with ZipFile(zip_path) as myzip:
        files_in_zip = myzip.namelist()

    ZipFile(zip_path) as myzip:
        with myzip.open(files_in_zip) as myfile:
            img = Image.open(myfile)"""

    image_reader = tf.WholeFileReader()

    _, image_file = image_reader.read(filename_queue)

    print(image_file)

    # image_file =
    image = tf.image.decode_jpeg(image_file)
    resized_image = tf.image.resize_images(image, [448, 448])

    print(resized_image)

    # Scale to image sizes later
    paddings = [[223, 223], [223, 223]]

    # Kinda messy, hopefully it can all be handled in one statement
    features = resized_image  # np.array(resized_image, dtype=np.float32)
    # features = tf.pad(features, paddings, "CONSTANT") #add padding to images to compensate for the size

    labels = np.array(["meme"])
    mode = 1
    output = yolo_model(features, labels, mode)

    print(output)

    sess = tf.InteractiveSession()
    # sess.run(tf.global_variables_initializer())

    tfPrintOutput = tf.Print(output, [output], message="output is:")
    outputTensor = tfPrintOutput.eval()
    sess.run(tf.global_variables_initializer())

    print(sess.run(outputTensor))

# in main, we run the session
if __name__ == "__main__":
    tf.app.run()