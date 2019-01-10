#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from keras.preprocessing.image import img_to_array, array_to_img


def normalize_data(data):
    numerator = data - np.expand_dims(np.mean(data, 1), 1)
    denominator = np.expand_dims(np.std(data, 1), 1)

    return numerator / (denominator + 1e-7)


def load_mnist(normalize=True):
    # import mnist dataset
    mnist = input_data.read_data_sets('MNIST/', one_hot=True)

    train_x = mnist.train.images
    train_y = mnist.train.labels

    valid_x = mnist.validation.images
    valid_y = mnist.validation.labels

    test_x = mnist.test.images
    test_y = mnist.test.labels

    # Normalize images
    if normalize:
        train_x = normalize_data(train_x)
        valid_x = normalize_data(valid_x)
        test_x = normalize_data(test_x)

    print(train_x.shape, valid_x.shape, test_x.shape)
    # Convert the image to 3 channels
    train_x = np.dstack([train_x] * 3)
    valid_x = np.dstack([valid_x] * 3)
    test_x = np.dstack([test_x] * 3)

    # Reshape images as per the tensor format required by tensorflow
    train_x = train_x.reshape(-1, 28, 28, 3)
    valid_x = valid_x.reshape(-1, 28, 28, 3)
    test_x = test_x.reshape(-1, 28, 28, 3)

    # resize the images 48*48 as required by VGG16
    train_x = np.asarray([img_to_array(array_to_img(im, scale=False).resize((48, 48))) for im in train_x])
    valid_x = np.asarray([img_to_array(array_to_img(im, scale=False).resize((48, 48))) for im in valid_x])
    test_x = np.asarray([img_to_array(array_to_img(im, scale=False).resize((48, 48))) for im in test_x])

    print(train_x.shape, valid_x.shape, test_x.shape)
    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)
