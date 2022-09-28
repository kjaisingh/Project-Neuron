import cv2
import glob
import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import os
import pprint
import mmap

from scipy import signal
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from scipy.sparse.linalg import svds
from sklearn.linear_model import Ridge
from skimage.morphology import dilation
from skimage.morphology import disk

import caiman as cm
import caiman.paths
from caiman.motion_correction import MotionCorrect
from caiman.paths import caiman_datadir
from caiman.source_extraction.volpy import utils
from caiman.source_extraction.volpy.volparams import volparams
from caiman.source_extraction.volpy.volpy import VOLPY
from caiman.summary_images import local_correlations_movie_offline
from caiman.summary_images import mean_image
from caiman.utils.utils import download_demo, download_model

'''
data: 1-d array
    one dimensional signal

window_length: int
    length of window size for temporal filter

fr: int
    number of samples per second in the video
    
hp_freq: float
    high-pass cutoff frequency to filter the signal after computing the trace
    
clip: int
    maximum number of spikes for producing templates

threshold_method: str
    adaptive_threshold method threshold based on estimated peak distribution
    simple method threshold based on estimated noise level 
    
min_spikes: int
    minimal number of spikes to be detected
    
pnorm: float
    a variable deciding the amount of spikes chosen for adaptive threshold method

threshold: float
    threshold for spike detection in simple threshold method 
    The real threshold is the value multiply estimated noise level
'''

'''
def denoise_spikes(data, window_length, fr = 1000,  hp_freq = 100,  clip = 100, min_spikes = 3, pnorm = 0.5):
    data = signal_filter(data, hp_freq, fr, order = 5)
    data = data - np.median(data)
    pks = data[signal.find_peaks(data, height = None)[0]]

    # first round of spike detection    
    thresh, _, _, low_spikes = adaptive_thresh(pks, clip, 0.25, min_spikes)
    locs = signal.find_peaks(data, height=thresh)[0]

    # spike template
    window = np.int64(np.arange(-window_length, window_length + 1, 1))
    locs = locs[np.logical_and(locs > (-window[0]), locs < (len(data) - window[-1]))]
    PTD = data[(locs[:, np.newaxis] + window)]
    PTA = np.median(PTD, 0)
    PTA = PTA - np.min(PTA)
    templates = PTA

    # whitened matched filtering based on spike times detected in the first round of spike detection
    datafilt = whitened_matched_filter(data, locs, window)    
    datafilt = datafilt - np.median(datafilt)

    # second round of spike detection on the whitened matched filtered trace
    pks2 = datafilt[signal.find_peaks(datafilt, height=None)[0]]
    thresh2, falsePosRate, detectionRate, low_spikes = adaptive_thresh(pks2, clip=0, pnorm=pnorm, min_spikes=min_spikes)  # clip=0 means no clipping
    spikes = signal.find_peaks(datafilt, height=thresh2)[0]
    
    # compute reconstructed signals and adjust shrinkage
    t_rec = np.zeros(datafilt.shape)
    t_rec[spikes] = 1
    t_rec = np.convolve(t_rec, PTA, 'same')   
    factor = np.mean(data[spikes]) / np.mean(datafilt[spikes])
    datafilt = datafilt * factor
    thresh2_normalized = thresh2 * factor

    # plot data
    x = [x for x in range(len(data))]
    plt.figure(figsize = (20, 10))
    plt.plot(x, data)
    plt.title('Raw Data')
    plt.tight_layout()
    plt.savefig("filtered/fig1.png")

    # plot threshold
    plt.figure(figsize = (20, 20))
    plt.subplot(211)
    plt.hist(pks, 500)
    plt.axvline(x = thresh, c = 'r')
    plt.title('Raw Data')
    plt.subplot(212)
    plt.hist(pks2, 500)
    plt.axvline(x = thresh2, c = 'r')
    plt.title('After Matched Filter')
    plt.tight_layout()
    plt.savefig("filtered/fig2.png")

    # plot peak-triggered average
    plt.figure(figsize = (20, 15))
    plt.plot(np.transpose(PTD), c = [0.5, 0.5, 0.5])
    plt.plot(PTA, c = 'black', linewidth = 2)
    plt.title('Peak-Triggered Average')
    plt.savefig("filtered/fig3.png")

    # plot detected spikes
    plt.figure(figsize = (40, 40))
    plt.subplot(211)
    plt.plot(data)
    plt.plot(locs, np.max(data) * 1.1 * np.ones(locs.shape), 
        color = 'r', marker = 'o', fillstyle = 'none', linestyle = 'none')
    plt.plot(spikes, np.max(data) * 1 * np.ones(spikes.shape), 
        color = 'g', marker = 'o', fillstyle = 'none', linestyle = 'none')
    plt.subplot(212)
    plt.plot(datafilt)
    plt.plot(locs, np.max(datafilt) * 1.1 * np.ones(locs.shape), 
        color = 'b', marker = 'o', fillstyle = 'none', linestyle = 'none')
    plt.plot(spikes, np.max(datafilt) * 1 * np.ones(spikes.shape), 
        color = 'y', marker = 'o', fillstyle = 'none', linestyle = 'none')
    plt.savefig("filtered/fig4.png", dpi = 500)

    return datafilt, spikes, t_rec, templates, low_spikes, thresh2_normalized
'''

def adaptive_thresh(pks, clip, pnorm = 0.5, min_spikes = 10):
    # find median of the kernel density estimation of peak heights
    spread = np.array([pks.min(), pks.max()])
    spread = spread + np.diff(spread) * np.array([-0.05, 0.05])
    low_spikes = False
    pts = np.linspace(spread[0], spread[1], 2001)
    kde = stats.gaussian_kde(pks)
    f = kde(pts)    
    xi = pts
    center = np.where(xi > np.median(pks))[0][0]

    fmodel = np.concatenate([f[0:center + 1], np.flipud(f[0:center])])
    if len(fmodel) < len(f):
        fmodel = np.append(fmodel, np.ones(len(f) - len(fmodel)) * min(fmodel))
    else:
        fmodel = fmodel[0:len(f)]

    # adjust the model so it doesn't exceed the data:
    csf = np.cumsum(f) / np.sum(f)
    csmodel = np.cumsum(fmodel) / np.max([np.sum(f), np.sum(fmodel)])
    lastpt = np.where(np.logical_and(csf[0:-1] > csmodel[0:-1] + np.spacing(1), csf[1:] < csmodel[1:]))[0]
    if not lastpt.size:
        lastpt = center
    else:
        lastpt = lastpt[0]
    fmodel[0:lastpt + 1] = f[0:lastpt + 1]
    fmodel[lastpt:] = np.minimum(fmodel[lastpt:], f[lastpt:])

    # find threshold
    csf = np.cumsum(f)
    csmodel = np.cumsum(fmodel)
    csf2 = csf[-1] - csf
    csmodel2 = csmodel[-1] - csmodel
    obj = csf2 ** pnorm - csmodel2 ** pnorm
    maxind = np.argmax(obj)
    thresh = xi[maxind]

    if np.sum(pks > thresh) < min_spikes:
        low_spikes = True
        thresh = np.percentile(pks, 100 * (1 - min_spikes / len(pks)))
    elif ((np.sum(pks > thresh) > clip) & (clip > 0)):
        thresh = np.percentile(pks, 100 * (1 - clip / len(pks)))

    ix = np.argmin(np.abs(xi - thresh))
    falsePosRate = csmodel2[ix] / csf2[ix]
    detectionRate = (csf2[ix] - csmodel2[ix]) / np.max(csf2 - csmodel2)
    return thresh, falsePosRate, detectionRate, low_spikes

def whitened_matched_filter(data, locs, window):
    N = np.ceil(np.log2(len(data)))
    censor = np.zeros(len(data))
    censor[locs] = 1
    censor = np.int16(np.convolve(censor.flatten(), np.ones([1, len(window)]).flatten(), 'same'))
    censor = (censor < 0.5)
    noise = data[censor]

    _, pxx = signal.welch(noise, fs = 2 * np.pi, window = signal.get_window('hamming', 1000), nfft = 2 ** N, detrend = False, nperseg = 1000)
    Nf2 = np.concatenate([pxx, np.flipud(pxx[1:-1])])
    scaling_vector = 1 / np.sqrt(Nf2)

    cc = np.pad(data.copy(),(0, int(2 ** N - len(data))), 'constant')    
    dd = (cv2.dft(cc,flags=cv2.DFT_SCALE+cv2.DFT_COMPLEX_OUTPUT)[:,0,:] * scaling_vector[:,np.newaxis])[:,np.newaxis,:]
    dataScaled = cv2.idft(dd)[:, 0, 0]
    PTDscaled = dataScaled[(locs[:, np.newaxis] + window)]
    PTAscaled = np.mean(PTDscaled, 0)
    datafilt = np.convolve(dataScaled, np.flipud(PTAscaled), 'same')
    datafilt = datafilt[:len(data)]
    return datafilt

def signal_filter(sg, freq, fr, order = 3, mode = 'high'):
    normFreq = freq / (fr / 2)
    b, a = signal.butter(order, normFreq, mode)
    sg = np.single(signal.filtfilt(b, a, sg, padtype='odd', padlen = 3 * (max(len(b), len(a)) - 1)))
    return sg

def width_filter(data, spikes, window_width):
    detects = []
    for i in range(0, len(spikes)):
        neighbours = [(spikes[i], data[spikes[i]])]
        low = i - 1
        high = i + 1

        while (low >= 0 and spikes[i] - spikes[low] <= window_width):
            neighbours.append((spikes[low], data[spikes[low]]))
            low -= 1


        while (high < len(spikes) and spikes[high] - spikes[i] <= window_width):
            neighbours.append((spikes[high], data[spikes[high]]))
            high += 1

        max_spike = max(neighbours, key = lambda x : x[1])
        if (data[spikes[i]] == max_spike[1]):
            detects.append(spikes[i])
    return np.array(detects)

def denoise_spikes_raw(data, window_length, window_width, fr = 1000,  hp_freq = 30,  clip = 100, min_spikes = 1, pnorm = 0.5):
    data_orig = data
    data = signal_filter(data, hp_freq, fr, order = 5)
    data = data - np.median(data)
    pks = data[signal.find_peaks(data, height = None)[0]]

    # first round of spike detection    
    thresh, _, _, low_spikes = adaptive_thresh(pks, clip, 0.25, min_spikes)
    locs = signal.find_peaks(data, height=thresh)[0]

    # spike template
    window = np.int64(np.arange(-window_length, window_length + 1, 1))
    locs = locs[np.logical_and(locs > (-window[0]), locs < (len(data) - window[-1]))]
    PTD = data_orig[(locs[:, np.newaxis] + window)]
    PTA = np.mean(PTD, 0)
    templates = PTA

    # whitened matched filtering based on spike times detected in the first round of spike detection
    datafilt = whitened_matched_filter(data, locs, window)    
    datafilt = datafilt - np.median(datafilt)

    # second round of spike detection on the whitened matched filtered trace
    pks2 = datafilt[signal.find_peaks(datafilt, height = None)[0]]
    thresh2, falsePosRate, detectionRate, low_spikes = adaptive_thresh(pks2, clip = clip, pnorm = pnorm, min_spikes = min_spikes)
    spikes = signal.find_peaks(datafilt, height = thresh2)[0]

    # third round of spike detection on the minimum window width
    detects = width_filter(data_orig, spikes, window_width)

    # compute reconstructed signals and adjust shrinkage
    t_rec = np.zeros(datafilt.shape)
    t_rec[spikes] = 1
    t_rec = np.convolve(t_rec, PTA, 'same')   
    factor = np.mean(data[detects]) / np.mean(datafilt[spikes])
    datafilt = datafilt * factor
    thresh2_normalized = thresh2 * factor

    # plot data
    x = [x for x in range(len(data_orig))]
    plt.figure(figsize = (20, 10))
    plt.plot(x, data_orig)
    plt.title('Raw Data')
    plt.tight_layout()
    plt.savefig("raw/fig1.png")

    # plot threshold
    plt.figure(figsize = (20, 20))
    plt.subplot(211)
    plt.hist(pks, 500)
    plt.axvline(x = thresh, c = 'r')
    plt.title('Raw Data')
    plt.subplot(212)
    plt.hist(pks2, 500)
    plt.axvline(x = thresh2, c = 'r')
    plt.title('After Matched Filter')
    plt.tight_layout()
    plt.savefig("raw/fig2.png")

    # plot peak-triggered average
    plt.figure(figsize = (20, 15))
    plt.plot(np.transpose(PTD), c = [0.5, 0.5, 0.5])
    plt.plot(PTA, c = 'black', linewidth = 2)
    plt.title('Peak-Triggered Average')
    plt.savefig("raw/fig3.png")

    # plot detected spikes
    plt.figure(figsize = (20, 20))
    plt.subplot(211)
    plt.plot(data_orig)
    plt.plot(locs, np.max(data_orig) * 1.1 * np.ones(locs.shape), color = 'r', marker = 'o', fillstyle = 'none', linestyle = 'none')
    locs_y = [data_orig[x] for x in detects]
    locs_y = np.asarray(locs_y)
    plt.plot(detects, locs_y, color = 'g', marker = 'o', fillstyle = 'none', linestyle = 'none')
    plt.subplot(212)
    plt.plot(datafilt)
    plt.plot(locs, np.max(datafilt) * 1.1 * np.ones(locs.shape), 
        color = 'b', marker = 'o', fillstyle = 'none', linestyle = 'none')
    plt.plot(detects, np.max(datafilt) * 1 * np.ones(detects.shape), 
        color = 'y', marker = 'o', fillstyle = 'none', linestyle = 'none')
    plt.show()

    return datafilt, spikes, t_rec, templates, low_spikes, thresh2_normalized

data = sio.loadmat('data.mat')['IntensOrig'][:, 0]
data = data[15000:20000]
window_length = 25
window_width = 10
spikes = denoise_spikes_raw(data, window_length, window_width)
# spikes = denoise_spikes(data, window_length)
