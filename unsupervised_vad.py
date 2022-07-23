from audio_tools import *
import numpy as np
import os
import matplotlib as plt
import os
import soundfile
import numpy as np
from python_speech_features import mfcc
import csv
import argparse
import time
from scipy import signal
import pickle
import h5py
from sklearn import preprocessing
import librosa
from scipy.special import erf
import config as cfg
#from myfilters import mel_filter_bank
from mel_weights import SpectrogramToMelMatrix

##Function definitions:
def vad_help():
    print("Usage:")
    print("python unsupervised_vad.py")

#### Display tools
def plot_this(s, title=''):
    import pylab
    s = s.squeeze()
    if s.ndim == 1:
        pylab.plot(s)
    else:
        pylab.imshow(s, aspect='auto')
        pylab.title(title)
    pylab.show()

def plot_these(s1, s2):
    import pylab
    try:
        # If values are numpy arrays
        pylab.plot(s1 / max(abs(s1)), color='red')
        pylab.plot(s2 / max(abs(s2)), color='blue')
    except:
        # Values are lists
        pylab.plot(s1, color='red')
        pylab.plot(s2, color='blue')
    pylab.legend()
    pylab.show()

def plot_these1(s1, s2):
    import matplotlib.pyplot as plt

    # plt.ion()
    plt.figure(figsize=(16, 9))
    try:
        # If values are numpy arrays
        plt.plot(s1 / max(abs(s1)), color='red')
        plt.plot(s2 / max(abs(s2)), color='blue')
    except:
        # Values are lists
        plt.plot(s1, color='red')
        plt.plot(s2, color='blue')
    plt.legend()
    plt.show()
    # plt.pause(2)
    # plt.close()

#### Energy tools
def zero_mean(xframes):
    """
        remove mean of framed signal
        return zero-mean frames.
        """
    m = np.mean(xframes, axis=1)
    xframes = xframes - np.tile(m, (xframes.shape[1], 1)).T
    return xframes

def compute_nrg(xframes):
    # calculate per frame energy
    n_frames = xframes.shape[1]
    return np.diagonal(np.dot(xframes, xframes.T)) / float(n_frames)

def compute_log_nrg(xframes):
    # calculate per frame energy in log
    n_frames = xframes.shape[1]
    raw_nrgs = np.log(compute_nrg(xframes + 1e-5)) / float(n_frames)
    return (raw_nrgs - np.mean(raw_nrgs)) / (np.sqrt(np.var(raw_nrgs)))

def power_spectrum(xframes):
    """
        x: input signal, each row is one frame
        """
    X = np.fft.fft(xframes, axis=1)
    X = np.abs(X[:, :X.shape[1] / 2]) ** 2
    return np.sqrt(X)

def nrg_vad(xframes, percent_thr, nrg_thr=0., context=5):
    """
        Picks frames with high energy as determined by a
        user defined threshold.

        This function also uses a 'context' parameter to
        resolve the fluctuative nature of thresholding.
        context is an integer value determining the number
        of neighboring frames that should be used to decide
        if a frame is voiced.

        The log-energy values are subject to mean and var
        normalization to simplify the picking the right threshold.
        In this framework, the default threshold is 0.0
        """
    xframes = zero_mean(xframes)
    n_frames = xframes.shape[0]

    # Compute per frame energies:
    xnrgs = compute_log_nrg(xframes)
    xvad = np.zeros((n_frames, 1))
    for i in range(n_frames):
        start = max(i - context, 0)
        end = min(i + context, n_frames - 1)
        n_above_thr = np.sum(xnrgs[start:end] > nrg_thr)
        n_total = end - start + 1
        xvad[i] = 1. * ((float(n_above_thr) / n_total) > percent_thr)
    return xvad

def read_audio_file1(path, fmt, flag=0):
    files = []
    names = []
    for root, dir, filenames in os.walk(path):
        # print('root = ',root)
        # print('filename.len = ',len(filenames))
        for filename in filenames:
            if filename.endswith(fmt):
                # print('filename = ',filename)
                file_path = root + '/' + filename
                files.append(file_path)

                filename = filename.split('.')[0]

                if (flag == 1):
                    name = file_path.split('.')[0]
                    name = name.split('/')
                    filename = name[-3] + '_' + name[-2] + '_' + name[-1]
                names.append(filename)

    return files, names

def max_filter(vads):
    hist = 0
    win_len = 20
    half_len = int(win_len / 2)
    vad_len = len(vads)
    new_vads = []
    wri_vads = []
    for i, vad in enumerate(vads):
        if i < win_len:
            new_vads.append(float(vad))
            wri_vads.append(int(vad))
            continue

        # if (i < vad_len - half_len) and vads[i] == 0:
        if (i < vad_len - half_len):
            for j in range(i - half_len, i + half_len):
                hist = hist + vads[j]
            if hist > half_len:
                new_vads.append(1.)
                wri_vads.append(1)
            else:
                new_vads.append(0.)
                wri_vads.append(0)
        else:
            new_vads.append(float(vad))
            wri_vads.append(int(vad))

        hist = 0
    new_vads = np.array(new_vads)
    new_vads = new_vads.reshape(len(new_vads), 1)

    return new_vads, wri_vads

def get_start_end_pts(new_vad):
    starts = []
    ends = []

    flags = 1
    for i, vad in enumerate(new_vad):

        if int(new_vad[0]) == 1 and flags == 1:
            starts.append(0)
            flags = 0

        if int(new_vad[i]) == 0 and int(new_vad[i + 1]) == 1:
            starts.append(i)
        elif int(new_vad[i]) == 1 and int(new_vad[i + 1]) == 0:
            ends.append(i)

        if i == len(new_vad) - 2:
            break

    if int(new_vad[-2]) == 1 and int(new_vad[-1]) == 1:
        ends.append(len(new_vad) - 1)

    return starts, ends

def write_startEnd_info(filename, starts, ends):
    assert (len(starts) == len(ends))
    f = open(filename, 'w')

    for start, end in zip(starts, ends):
        if start > end:
            print('==========start end info err========')
        f.write(str(start) + ' ' + str(end) + '\n')

    f.close()

def write_frame_labels(fp, vads, name):
    fp.write(name + ':')
    for v in vads:
        fp.write(str(v) + ' ')
    fp.write('\n')

    return fp


def read_audio(path, target_fs=None):
    (audio, fs) = soundfile.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs
    return audio, fs


def write_audio(path, audio, sample_rate):
    soundfile.write(file=path, data=audio, samplerate=sample_rate)

fs = 16000
win_len = 256
hop_len = 128
_, s = read_wav('test.wav')
audio, _ = read_audio("test.wav")
sframes = enframe(s, win_len, hop_len)  # rows: frame index, cols: each frame
percent_high_nrg = 0.05

vad = nrg_vad(sframes, percent_high_nrg)
new_vad, wri_vad = max_filter(vad)
nn=len(vad)

index= np.arange(0,1,1)
for i in range(nn):
    if wri_vad[i] == 0:
        index = np.append(index, np.arange(i*128,i*128+128,1))
audio_out = np.delete(audio, index)
write_audio("out.wav", audio_out, fs)

# a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
# index = [2, 2, 3, 6]
#
# new_a = np.delete(a, index)

plot_these1(deframe(new_vad, win_len, hop_len), s)




'''plot_these1(deframe(new_vad, win_len, hop_len), s)
x = np.zeros(len(s))
for i in range(len(s)):
    x[i] = i
for i in range(len(vad)):
    for j in range(128):
        s[i * 128 + j] = s[i * 128 + j] * new_vad[i]
plt.plot(x, s)
plt.show()'''


