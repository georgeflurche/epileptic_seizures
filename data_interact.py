import os
import logging
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(
    format='%(levelname)s [%(module)s]: %(message)s'
)
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


def read_dataset(dataset_path):
    '''
    This function gets as input the directory with all the txt samples
    corresponding to a patient and returns a dictionary with the structure:
    {
        "sample_<sample_name1>: <samples_array1>
        "sample_<sample_name2>: <samples_array2>        
        ...
    }
    '''
    abs_path = os.path.expandvars(dataset_path)
    if not os.path.isdir(abs_path):
        return None
    all_files = os.listdir(abs_path)
    data_set = dict()
    for sample in all_files:
        with open(os.path.join(abs_path, sample)) as f:
            s_arr = list(int(i) for i in f.read().strip().split("\n"))
        data_set[f"sample_{sample.split('.')[0]}"] = s_arr
    return data_set


def plot_sample(sample_array, *args):
    '''
    The data from the brain is sampled with a frequency of 173.6 Hz, hence
    the total of 4097 samples implie a duration of ~23.6 seconds per array
    The function gets the array as input and plots it with a x-step of 0.00576
    '''
    DEFAULT_X_STEP = 23.6/len(sample_array)
    x_array = list(DEFAULT_X_STEP * i for i in range(1, len(sample_array) + 1))
    plt.plot(x_array, sample_array)
    for signal in args:
        plt.plot(x_array, signal)
    # plt.axis(0, 24, min(sample_array), max(sample_array))
    plt.ylabel('uV')
    plt.xlabel('Time (seconds)')


def calculate_accuracy(r, s):
    N = len(r)
    if all(s[i] == r[i] for i in range(N)):
        return 100
    corr = signal.correlate(r, s)
    max_corr = np.max(corr)
    mse = sum(np.square(r-s))/N
    mean_arr = sum(r)/len(r)*np.ones(N)
    sigma = sum(np.square(r-mean_arr))/N
    max_corr_possible = (sigma+mse)*N

    return min(max_corr/max_corr_possible*100, 99.89)

def associate(S_, eeg_signal, artifact_signal):
    acc1 = calculate_accuracy(
        eeg_signal[:, 0], S_[:, 0]*eeg_signal[0][0]/S_[0][0])
    acc2 = calculate_accuracy(
        artifact_signal[:, 0], S_[:, 0]*artifact_signal[0][0]/S_[0][0])
    if acc1 > acc2:
        s1_ = S_[:, 0]*eeg_signal[0][0]/S_[0][0]
        s2_ = S_[:, 1]*artifact_signal[0][0]/S_[0][1]
    else:
        s1_ = S_[:, 1]*eeg_signal[0][0]/S_[0][1]
        s2_ = S_[:, 0]*artifact_signal[0][0]/S_[0][0]
    return s1_, s2_


def plot_results(models):
    plot = plt.figure()
    names = [
        'Standalone Signals (Reference)',
        'Mixed Signals',
        'ICA recovered signals',
        'PCA recovered signals']
    colors = ['red', 'blue']

    for i, (model, name) in enumerate(zip(models, names), 1):
        plt.subplot(len(models), 1, i)
        plt.title(name)
        for sig, color in zip(model.T, colors):
            plt.plot(sig, color=color)

def save_all_images(data_set, out_dir=f'{os.getcwd()}/dataset_images'):
    if not os.path.exists(out_dir):
        os.system(f'mkdir {out_dir}')
    
    for k, v in data_set.items():
        plot = plt.figure()        
        plt.plot(v)
        plt.savefig(os.path.join(out_dir, f'{k}.png'))
        del plot


def longest_substring(list_of_substrings):
    result = list_of_substrings[0]
    for ss in list_of_substrings[1:]:
        new_result = ''
        for i, ch in enumerate(result):
            if ch == ss[i]:
                new_result += ch
        result = new_result
    return result


def add_zeros(integer, length):
    result = str(integer)
    while len(result) < length:
        result = '0' + result
    return result



