import os
import argparse
import json
import logging
import numpy as np
import tensorflow as tf
from datetime import datetime
from enum import Enum, unique
from tensorflow.keras import datasets, layers, models, optimizers
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.decomposition import FastICA, PCA
from defaults import *
from data_interact import read_dataset, longest_substring, plot_sample
from data_interact import calculate_accuracy, add_zeros
from data_interact import associate, plot_results, save_all_images
from SignalGenerator import generate_random_signal, save_signal, read_signal
from SignalGenerator import generate_step_signal

logging.basicConfig(
    format='%(levelname)s [%(module)s]: %(message)s'
)
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


@unique
class State(Enum):
    HEALTHY = 1
    EPILEPTIC = 2


class Patient:
    def __init__(self, name, state, tag, samples):
        self.name = name
        self.state = state
        self.tag = tag
        self.samples = samples


if __name__ == "__main__":
    patients = list()
    with open(LABELING_PATH) as f:
        content = f.read().split('\n')[:-1]
    samples_labels = {i.split(',')[0]:i.split(',')[1] for i in content}
    for ds in [A_DATASET, B_DATASET, C_DATASET, D_DATASET, E_DATASET]:
        bn = os.path.basename(ds)
        tag = longest_substring(os.listdir(os.path.expandvars(ds)))
        d = read_dataset(ds)
        samples = dict()
        for k, v in d.items():
            samples[k] = {'x': v, 'y': samples_labels[k]}
        patient = Patient(
            name=bn,
            state=State.HEALTHY if bn in HEALTHY_PATIENTS else State.EPILEPTIC,
            tag=tag,
            samples=samples
        )
        patients.append(patient)


    X = list()
    y = list()
    for i in range(1, 101):
        for p in patients:
            X.append(p.samples[f'sample_{p.tag}{add_zeros(i, 3)}']['x'])
            y.append([int(p.samples[f'sample_{p.tag}{add_zeros(i, 3)}']['y'])])
            y[-1].append(1 if y[-1][0] == 0 else 0)


    #fig, (ax1, ax2) = plt.subplots(2)
    #fig.suptitle('Reduced sample frequency')
    #ax1.plot(X[5])
    #ax1.set_title(f"Data extracted with 147 Hz")
    #for j, row in enumerate(X):
    #    X[j] = list(e for i, e in enumerate(row) if i%2==0)
    #ax2.plot(X[5])
    #ax2.set_title(f"Data extracted with 34.8 Hz")
    #plt.show()

    X_train = np.array(X[:400])
    y_train = np.array(y[:400]).reshape(400, 2)

    X_test = np.array(X[400:])
    y_test = np.array(y[400:]).reshape(100, 2)


    x_tr_l, x_tr_w = X_train.shape
    x_ts_l, x_ts_w = X_test.shape
    input_shape = (x_tr_w, 1)

    model = models.Sequential()
    model.add(layers.Conv1D(filters=32, kernel_size=3, activation='relu',
              input_shape=input_shape))
    model.add(layers.MaxPooling1D(pool_size=20))
    model.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=20))
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dense(2, activation='sigmoid'))

    log_dir = "logs/fit/test_2_conv_layer" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1)

    model.summary()
    opt = optimizers.SGD(learning_rate=0.1, nesterov=True, momentum=0.9)
    #opt = optimizers.Adam()
    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.CategoricalCrossentropy(
            from_logits=False,
            label_smoothing=0,
            reduction="auto",
            name="categorical_crossentropy"),
        metrics=['accuracy'])

    x_tr_l, x_tr_w = X_train.shape
    X_train = X_train.reshape(x_tr_l, x_tr_w, NOF_CHANNELS)
    x_ts_l, x_ts_w = X_test.shape
    X_test = X_test.reshape(x_ts_l, x_ts_w, NOF_CHANNELS)

    history = model.fit(
        X_train, y_train, epochs=100, validation_data=(X_test, y_test),
        batch_size=64, callbacks=[tensorboard_callback])
