import tensorflow as tf
from tensorflow import keras

import os
import tempfile

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import pickle

# metrics
METRICS = [
    keras.metrics.TruePositives(name='tp'),
    keras.metrics.FalsePositives(name='fp'),
    keras.metrics.TrueNegatives(name='tn'),
    keras.metrics.FalseNegatives(name='fn'), 
    keras.metrics.BinaryAccuracy(name='accuracy'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),
    # keras.metrics.AUC(name='auc'),
]

def main():

    dims = 64
    
    classes = ["normal","sars-cov-2"]
    x_train, y_train, x_test, y_test = load_dataset(dims)

    model = make_model()
    model.summary()

    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=1)
    plot_metrics(history)
    pred = model.predict(x_test)
    plot_cm(y_test, pred)

def make_model(metrics = METRICS, output_bias=None):
    model = keras.Sequential([
        keras.layers.Conv2D(8,kernel_size=4,padding='same',activation='relu',input_shape=(64,64,1)),
        keras.layers.MaxPool2D(pool_size=(8, 8), strides=(8,8), padding='same'),
        keras.layers.Conv2D(16,kernel_size=2,padding='same',activation='relu',input_shape=(64,64,1)),
        keras.layers.MaxPool2D(pool_size=(4, 4), strides=(4,4), padding='same'),
        keras.layers.Flatten(),
        keras.layers.Dense(1, activation='sigmoid'),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(lr=1e-3),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=metrics)

    return model

def plot_cm(labels, predictions, p=0.5):
  cm = confusion_matrix(labels, predictions > p)
  plt.clf()
  plt.figure(figsize=(5,5))
  sns.heatmap(cm, annot=True, fmt="d")
  plt.title('Confusion matrix @{:.2f}'.format(p))
  plt.ylabel('Actual label')
  plt.xlabel('Predicted label')

  plt.savefig("./covid/plots/cm_d.png")

  print('Non-SARS-CoV-2 Detected (True Negatives): ', cm[0][0])
  print('Non-SARS-CoV-2 Incorrectly Detected (False Positives): ', cm[0][1])
  print('SARS-CoV-2 Transactions Missed (False Negatives): ', cm[1][0])
  print('SARS-CoV-2 (True Positives): ', cm[1][1])
  print('Total Radiographs: ', np.sum(cm[1]))

def plot_metrics(history):
  metrics =  ['loss', 'precision', 'recall']
  plt.clf()
  for n, metric in enumerate(metrics):
    name = metric.replace("_"," ").capitalize()
    plt.subplot(2,2,n+1)
    plt.plot(history.epoch,  history.history[metric], color='blue', label='Train')
    plt.plot(history.epoch, history.history['val_'+metric],
             color='blue', linestyle="--", label='Val')
    plt.xlabel('Epoch')
    plt.ylabel(name)
    if metric == 'loss':
      plt.ylim([0, plt.ylim()[1]])
    else:
      plt.ylim([0,1])

    plt.legend()
    plt.savefig("./covid/plots/train_metrics_d.png")


def load_dataset(dims):

    # add d to use distorted aug dataset
    # add b to use resampled aug dataset
    x_train = pickle.load(open("./covid/" + str(dims) + "/x_train_d.p","rb"))
    y_train = pickle.load(open("./covid/" + str(dims) + "/y_train_d.p","rb"))
    x_test = pickle.load(open("./covid/" + str(dims) + "/x_test.p","rb"))
    y_test = pickle.load(open("./covid/" + str(dims) + "/y_test.p","rb"))

    x_train = np.asarray(x_train) / 255
    y_train = np.asarray(y_train)
    x_test = np.asarray(x_test) / 255
    y_test = np.asarray(y_test)

    # y_train = one_hot(y_train,2)
    # y_test = one_hot(y_test,2)
    # one hot encoding

    x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
    x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)

    print(x_train.shape)
    print(y_train.shape)

    return x_train, y_train, x_test, y_test

def one_hot(arr,nc):
    res = np.zeros((arr.shape[0],nc))
    # assume nc is correctly provided

    for i,val in enumerate(arr):
        res[i][val] = 1

    return res

if __name__ == "__main__":
    main()