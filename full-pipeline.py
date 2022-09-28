import cv2
import glob
import h5py
import logging
import matplotlib.pyplot as plt
import mmap
import numpy as np
import os
import pprint
import scipy
import struct

try:
    cv2.setNumThreads(0)
except:
    pass

try:
    if __IPYTHON__:
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    pass

import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.paths import caiman_datadir
from caiman.source_extraction.volpy import utils
from caiman.source_extraction.volpy.volparams import volparams
from caiman.source_extraction.volpy.volpy import VOLPY
from caiman.summary_images import local_correlations_movie_offline
from caiman.summary_images import mean_image
from caiman.utils.utils import download_demo, download_model

# options: logging.DEBUG, logging.INFO, logging.WARNING, or logging.ERROR
logging.basicConfig(format = "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s]" \
                    "[%(process)d] %(message)s",
                    level = logging.ERROR)

def main():
    pass
    '''
    -------------
    PART 1: SETUP
    -------------
    '''
    fnames = 'movie/movie.tif'  # file path to movie file
    file_dir = os.path.split(fnames)[0]
    ROIs = scipy.io.loadmat('movie/masks-output.mat')['ROI']  # file path to ROIs file
    ROIs = np.transpose(ROIs, [2, 0, 1])

    # dataset-dependent parameters
    fr = 1000  # sample rate of the movie

    # motion correction parameters
    pw_rigid = False  # flag for pw-rigid motion correction
    gSig_filt = (3, 3) # size of filter, in general gSig (see below),
                                                    # change this one if algorithm does not work
    max_shifts = (5, 5) # maximum allowed rigid shift
    strides = (48, 48) # start a new patch for pw-rigid motion correction every x pixels
    overlaps = (24, 24) # overlap between pathes (size of patch strides + overlaps)
    max_deviation_rigid = 3 # maximum deviation allowed for patch with respect to rigid shifts
    border_nan = 'copy'

    opts_dict = {
        'fnames': fnames,
        'fr': fr,
        'pw_rigid': pw_rigid,
        'max_shifts': max_shifts,
        'gSig_filt': gSig_filt,
        'strides': strides,
        'overlaps': overlaps,
        'max_deviation_rigid': max_deviation_rigid,
        'border_nan': border_nan
    }
    opts = volparams(params_dict = opts_dict)

    # play the movie
    display_images = False
    if display_images:
        m_orig = cm.load(fnames)
        ds_ratio = 0.2
        moviehandle = m_orig.resize(1, 1, ds_ratio)
        moviehandle.play(q_max = 99.5, fr = 40, magnification = 4)

    # start a cluster for parallel processing
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend = 'local',
        n_processes = None,
        single_thread = False
    )

    '''
    -------------------------
    PART 2: MOTION CORRECTION
    -------------------------
    '''
    do_motion_correction = True
    mc = MotionCorrect(fnames, dview = dview, **opts.get_group('motion'))
    if do_motion_correction:
        mc.motion_correct(save_movie = True)
    else:
        mc_list = [file for file in os.listdir(file_dir) if
                   (os.path.splitext(os.path.split(fnames)[-1])[0] in file and '.mmap' in file)]
        mc.mmap_file = [os.path.join(file_dir, mc_list[0])]
        print('reuse previously saved motion corrected file: {mc.mmap_file}')

    # compare with movie
    if display_images:
        m_orig = cm.load(fnames)
        m_rig = cm.load(mc.mmap_file)
        ds_ratio = 0.2
        moviehandle = cm.concatenate(
            [m_orig.resize(1, 1, ds_ratio), m_rig.resize(1, 1, ds_ratio)], 
            axis=2
        )
        moviehandle.play(fr = 40, q_max = 99.5, magnification = 4)  # press q to exit


    '''
    ----------------------
    PART 3: MEMORY MAPPING
    ----------------------
    '''
    do_memory_mapping = True
    if do_memory_mapping:
        border_to_0 = 0 if mc.border_nan == 'copy' else mc.border_to_0
        fname_new = cm.save_memmap_join(
            mc.mmap_file,
            base_name = 'memmap_' + os.path.splitext(os.path.split(fnames)[-1])[0],
            add_to_mov = border_to_0,
            dview = dview
        )
    else:
        mmap_list = [file for file in os.listdir(file_dir) if
                     ('memmap_' + os.path.splitext(os.path.split(fnames)[-1])[0]) in file]
        fname_new = os.path.join(file_dir, mmap_list[0])
        print(f'reuse previously saved memory mapping file:{fname_new}')

    '''
    --------------------
    PART 4: SEGMENTATION
    --------------------
    '''
    # create summary images
    img = mean_image(mc.mmap_file[0], window = 1000, dview = dview)
    img = (img - np.mean(img)) / np.std(img)

    gaussian_blur = False # use gaussian blur when there is too much noise in the video
    Cn = local_correlations_movie_offline(
        mc.mmap_file[0], 
        fr = fr,
        window = fr*4,
        stride = fr*4,
        winSize_baseline = fr,
        remove_baseline = True,
        gaussian_blur = gaussian_blur,
        dview = dview
    ).max(axis=0)
    img_corr = (Cn - np.mean(Cn)) / np.std(Cn)
    summary_images = np.stack([img, img, img_corr], axis = 0).astype(np.float32)

    # save summary images which are used in the VolPy GUI
    cm.movie(summary_images).save(fnames[:-5] + '_summary_images.tif')
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(summary_images[0])
    axs[1].imshow(summary_images[2])
    axs[0].set_title('mean image')
    axs[1].set_title('corr image')

    # identify ROIs via mask RCNN model
    ''' 
    weights_path = download_model('mask_rcnn')
    ROIs = utils.mrcnn_inference(
        img = summary_images.transpose([1, 2, 0]), 
        size_range =[5, 22], # decides size range of masks to be selected
        weights_path=weights_path,
        display_result = True
    )
    cm.movie(ROIs).save(fnames[:-5] + '_mrcnn_ROIs.hdf5')
    '''

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(summary_images[0])
    axs[1].imshow(ROIs.sum(0))
    axs[0].set_title('mean image')
    axs[1].set_title('masks')
    plt.savefig("demo-fig.png")

    # restart cluster to clean up memory
    cm.stop_server(dview = dview)
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend = 'local',
        n_processes = None,
        single_thread = False, 
        maxtasksperchild = 1
    )

    '''
    ------------------------------------
    PART 5: Denoising + Spike Extraction
    ------------------------------------
    '''
    # parameters for trace denoising and spike extraction
    ROIs = ROIs # region of interests
    index = list(range(len(ROIs))) # index of neurons
    weights = None # if None, use ROIs for initialization - to reuse weights check reuse weights block
    template_size = 0.02 # half size of the window length for spike templates, default is 20 ms
    context_size = 35 # number of pixels surrounding the ROI to censor from the background PCA
    visualize_ROI = False # whether to visualize the region of interest inside the context region
    flip_signal = True # flip signal or not - True for Voltron indicator, False for others
    hp_freq_pb = 1 / 3  # parameter for high-pass filter to remove photobleaching
    clip = 100 # maximum number of spikes to form spike template
    threshold_method = 'adaptive_threshold' # adaptive_threshold or simple
    min_spikes = 10 # minimal spikes to be found
    pnorm = 0.5 # a variable deciding the amount of spikes chosen for adaptive threshold method
    threshold = 3 # threshold for finding spikes only used in simple threshold method - increase to find less spikes
    do_plot = False # plot detail of spikes, template for the last iteration
    ridge_bg = 0.01 # ridge regression regularizer strength for background removement, larger value specifies stronger regularization
    sub_freq = 20  # frequency for subthreshold extraction
    weight_update = 'ridge' # ridge or NMF for weight update
    n_iter = 2 # number of iterations alternating between estimating spike times and spatial filters

    opts_dict = {
        'fnames': fname_new,
        'ROIs': ROIs,
        'index': index,
        'weights': weights,
        'template_size': template_size,
        'context_size': context_size,
        'visualize_ROI': visualize_ROI,
        'flip_signal': flip_signal,
        'hp_freq_pb': hp_freq_pb,
        'clip': clip,
        'threshold_method': threshold_method,
        'min_spikes':min_spikes,
        'pnorm': pnorm,
        'threshold': threshold,
        'do_plot':do_plot,
        'ridge_bg':ridge_bg,
        'sub_freq': sub_freq,
        'weight_update': weight_update,
        'n_iter': n_iter
    }
    opts.change_params(params_dict = opts_dict)

    vpy = VOLPY(n_processes = n_processes, dview = dview, params = opts)
    vpy.fit(n_processes = n_processes, dview = dview)

    '''
    ---------------------
    PART 6: Visualisation
    ---------------------
    '''
    if display_images:
        print(np.where(vpy.estimates['locality'])[0])    # neurons that pass the locality test
        idx = np.where(vpy.estimates['locality'] > 0)[0]
        utils.view_components(vpy.estimates, img_corr, idx)

    # reconstructed movie
    if display_images:
        mv_all = utils.reconstructed_movie(
            vpy.estimates.copy(), 
            fnames = mc.mmap_file,
            idx = idx, 
            scope = (0,1000), 
            flip_signal = flip_signal
        )
        mv_all.play(fr = 40, magnification = 3)

    # save results in .npy format
    save_result = True
    if save_result:
        vpy.estimates['ROIs'] = ROIs
        vpy.estimates['params'] = opts
        save_name = f'volpy_{os.path.split(fnames)[1][:-5]}_{threshold_method}'
        np.save(os.path.join(file_dir, save_name), vpy.estimates)

    # reuse weights
    # set weights = reuse_weights in opts_dict dictionary
    estimates = np.load(os.path.join(file_dir, save_name+'.npy'), allow_pickle = True).item()
    reuse_weights = []
    for idx in range(ROIs.shape[0]):
        coord = estimates['context_coord'][idx]
        w = estimates['weights'][idx][coord[0][0]:coord[1][0]+1, coord[0][1]:coord[1][1]+1]
        plt.figure()
        plt.imshow(w)
        plt.colorbar()
        plt.show()
        reuse_weights.append(w)

    '''
    ---------------------
    PART 7: Cleanup
    ---------------------
    '''
    # stop and clean up log files
    cm.stop_server(dview = dview)
    log_files = glob.glob('*_LOG_*')
    for log_file in log_files:
        os.remove(log_file)

if __name__ == "__main__":
    main()
