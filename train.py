#!/usr/bin/env python
# -*- coding:utf-8 -*-


import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras import callbacks
import matplotlib.pyplot as plt
from feature_extract import save_feature
from utils import load_mnist


def check_feature(file_path, input_data, name):
    # check if the features file exists
    if not os.path.exists(file_path):
        save_feature(input_data, name)


def train_model(train_fea_file, validation_fea_file, epochs, callback):
    """
    train the model using features extracted by VGG16
    train_fea_file: features of training data
    validation_fea_file: features of validation data
    """

    check_feature(train_fea_file, train, 'train')
    check_feature(validation_fea_file, valid, 'valid')

    # load features of training data and validation data
    train_fea = np.load(train_fea_file)
    valid_fea = np.load(validation_fea_file)

    # Define the densely connected classifier followed by finally dense layer for the number of classes
    model = Sequential()
    model.add(Dense(512, activation='relu', input_dim=(1*1*512)))
    model.add(Dense(10, activation='softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # Train the model
    history = model.fit(
        train_fea,
        train[1],
        epochs=epochs,
        verbose=1,
        validation_data=(valid_fea, valid[1]),
        callbacks=callback)
    return history, model

def save_model(trained_model):
    trained_model.save('trained_model.h5')

def plot_loss_acc(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.title('Training and validation accuracy')
    plt.plot(epochs, acc, 'red', label='Training acc')
    plt.plot(epochs, val_acc, 'blue', label='Validation acc')
    plt.legend()

    plt.figure()
    plt.title('Training and validation loss')
    plt.plot(epochs, loss, 'red', label='Training loss')
    plt.plot(epochs, val_loss, 'blue', label='Validation loss')

    plt.legend()
    plt.show()

# Incorporating reduced learning and early stopping for callback
reduce_learning = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=2,
    verbose=1,
    mode='auto',
    epsilon=0.0001,
    cooldown=2,
    min_lr=0
)

early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=7,
    verbose=1,
    mode='auto'
)
callbacks = [reduce_learning, early_stopping]

def main():

    train_fea_path = 'train_features.npy'
    val_fea_path = 'valid_features.npy'
    HISTORY, MODEL = train_model(train_fea_path, val_fea_path, 100, callbacks)
    save_model(MODEL)
    plot_loss_acc(HISTORY)


if __name__ == '__main__':
    train, valid, test = load_mnist(normalize=True)
    main()
