#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import itertools
import numpy as np
from keras.models import load_model
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from utils import load_mnist
from feature_extract import save_feature

def plot_confusion_matrix(conf_m, classes,
                          normalize=False,
                          title='Confusion Matrix',
                          cmap=plt.cm.Blues):

    """
    This function is used to visualize the confusion matrix
    conf_m: confusion matrix
    classes: label
    """
    plt.imshow(conf_m, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        conf_m = conf_m.astype('float') / conf_m.sum(axis=1)[:, np.newaxis]

    thresh = conf_m.max() / 2.
    for i, j in itertools.product(range(conf_m.shape[0]), range(conf_m.shape[1])):
        plt.text(j, i, conf_m[i, j],
                 horizontalalignment="center",
                 color="white" if conf_m[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.show()


def main():

    _, _, test = load_mnist(normalize=True)  # load testing data
    model_path = 'trained_model.h5'  # trained model path
    test_fea_path = 'test_features.npy'  # check if testing data feature file exists
    if not os.path.exists(test_fea_path):
        save_feature(test, 'test')

    x_test = np.load(test_fea_path)  # load testing data features

    MODEL = load_model(model_path)  # load trained model
    score = MODEL.evaluate(x_test, test[1], verbose=1)  # evaluate the model
    print('The accuracy of test dataset is: ', score[1]*100)

    y_pred = MODEL.predict(x_test)  # Predict the values from the testing dataset
    y_pred = np.argmax(y_pred, axis=1)  # Convert predictions classes to one hot vectors
    y_true = np.argmax(test[1], axis=1)  # Convert test observations to one hot vectors
    confusion_m = confusion_matrix(y_true, y_pred)  # compute the confusion matrix
    print(confusion_m)
    plot_confusion_matrix(confusion_m, classes=range(10))  # plot the confusion matrix


if __name__ == '__main__':
    main()

