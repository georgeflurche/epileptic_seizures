import os
import argparse
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.decomposition import FastICA, PCA
from data_interact import read_dataset, plot_sample, calculate_accuracy
from data_interact import associate, plot_results
from SignalGenerator import generate_random_signal, save_signal, read_signal
from SignalGenerator import generate_step_signal

logging.basicConfig(
    format='%(levelname)s [%(module)s]: %(message)s'
)
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


A_DATASET = "$WORK/A"
B_DATASET = "$WORK/B"
C_DATASET = "$WORK/C"
D_DATASET = "$WORK/D"
E_DATASET = "$WORK/E"


if __name__ == "__main__":
    data_set = read_dataset(A_DATASET)
    eeg_signal = np.array(data_set['sample_Z001']).reshape((4097, 1))
    artifact_signal = generate_step_signal(
        4097, 400, np.max(eeg_signal), np.min(eeg_signal)).reshape((4097, 1))
    time = np.linspace(0, 8, 4097)
    artifact_signal = 100*signal.sawtooth(2*np.pi*time, width=0.5).reshape((4097, 1))
    artifact_signal = np.array(generate_random_signal(-80, 80, 4097)).reshape((4097, 1))
    x1 = 0.8*eeg_signal + 0.2*artifact_signal
    x2 = 0.3*eeg_signal + 0.7*artifact_signal

    ica = FastICA(n_components=2, max_iter=20000)
    S = np.concatenate((eeg_signal, artifact_signal), axis=1)
    X = np.concatenate((x1, x2), axis=1)
    S_ = ica.fit_transform(X)
    A_ = ica.mixing_

    s1_ica, s2_ica = associate(S_, eeg_signal, artifact_signal)
    eeg_accuracy_ica = calculate_accuracy(eeg_signal[:, 0], s1_ica)
    _logger.info(
        f"The matching score for recovering the EEG signal using ICA algorithm "
        f"is {round(eeg_accuracy_ica, 2)}%")

    pca = PCA(n_components=2)
    H = pca.fit_transform(X)
    s1_pca, s2_pca = associate(H, eeg_signal, artifact_signal)
    eeg_accuracy_pca = calculate_accuracy(eeg_signal[:, 0], s1_pca)
    _logger.info(
        f"The matching score for recovering the EEG signal using PCA algorithm "
        f"is {round(eeg_accuracy_pca, 2)}%")
    
    models = [S, X,
        np.concatenate((s1_ica.reshape((4097, 1)), s2_ica.reshape((4097, 1))),
                       axis=1),
        np.concatenate((s1_pca.reshape((4097, 1)), s2_pca.reshape((4097, 1))),
                       axis=1)]

    #plot_sample(s1_ica, eeg_signal)
    plot_results(models)

    plt.show()
    

