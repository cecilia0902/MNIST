#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input


def save_feature(input_data, name, batch_size=16, img_width=48, img_height=48, img_depth=3):
    """
    This function is used to save the features extracting from VGG16
    input_data: train, valid or test data
    name: 'train', 'valid', or 'test'
    """


    # preprocessing the input data
    x_input = preprocess_input(input_data[0])

    # Create the base model
    base_model = VGG16(weights='imagenet',
                       include_top=False,
                       input_shape=(img_height, img_width, img_depth)
                       )
    # print(base_model.summary())

    # Extracting features
    input_fea = base_model.predict(np.array(x_input), batch_size=batch_size, verbose=1)
    # print(input_fea.shape)

    # Flatten extracted features
    input_fea = np.reshape(input_fea,
                           (input_fea.shape[0], (1 * 1 * 512))
                           )

    # Save features for future use
    np.save('vgg16/{}_features'.format(name), input_fea)
    print('{} features saved'.format(name))

