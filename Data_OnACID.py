#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Basic demo for the CaImAn Online algorithm (OnACID) using CNMF initialization.
It demonstrates the construction of the params and online_cnmf objects and
the fit function that is used to run the algorithm.
For a more complete demo check the script demo_OnACID_mesoscope.py

@author: jfriedrich & epnev
"""

import logging
import numpy as np
import scipy as sp
import os
import hdf5storage
import time

import matplotlib.pyplot as plt

import caiman as cm
from caiman.source_extraction import cnmf as cnmf
from caiman.paths import caiman_datadir


try:
    if __IPYTHON__:
        print("Detected iPython")
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    pass

#%%
# Set up the logger; change this if you like.
# You can log to a file using the filename parameter, or make the output more or less
# verbose by setting level to logging.DEBUG, logging.INFO, logging.WARNING, or logging.ERROR

logging.basicConfig(format=
                    "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s]"\
                    "[%(process)d] %(message)s",
                    level=logging.ERROR)
    # filename="/tmp/caiman.log"
# %%
def run_CaImAn_mouse(pathMouse,onAcid=False):
  
  for f in os.listdir(pathMouse):
    if f.startswith("Session"):
      pathSession = pathMouse + f + '/'
      print("\t Session: "+pathSession)
      run_CaImAn_session(pathSession,onAcid=onAcid)
      #return cnm, Cn, opts
  
def run_CaImAn_session(pathSession,onAcid=False):
    
    pass  # For compatibility between running under Spyder and the CLI

    ### %% set paths
    #pathSession = "/media/wollex/Analyze_AS3/Data/879/Session01/"
    #pathSession = "/home/wollex/Data/Documents/Uni/2016-XXXX_PhD/Japan/Work/Data/M879/Session01"
    sv_dir = "/home/wollex/Data/Documents/Uni/2016-XXXX_PhD/Japan/Work/Data/tmp/"
    
    fname = None
    for f in os.listdir(pathSession):
      if f.startswith("thy"):
        fname = pathSession + f
        if f.endswith('.h5'):
          break
    
    if not fname or not os.path.exists(fname):
      print("No file here to process :(")
      return
    
    svname = [pathSession + "results_OnACID.mat"]
    
    t_start = time.time()
    border_thr = 5         # minimal distance of centroid to border
    
    c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None, single_thread=False)
    fname_memmap = cm.save_memmap([fname], base_name='memmap_', save_dir=sv_dir, n_chunks=20, order='C', dview=dview)  # exclude borders
    #fname_memmap = sv_dir + "memmap__d1_512_d2_512_d3_1_order_C_frames_8989_.mmap"
    cm.stop_server(dview=dview)      ## restart server to clean up memory
    
    
    # set up CNMF parameters
    params_dict ={
            
            #general data
            'fnames': [fname_memmap],
            'fr': 15,
            'decay_time': 0.47,
            'gSig': [6, 6],                     # expected half size of neurons
            
            #model/analysis
            'rf': 64//2,                           # size of patch
            'K': 200,                           # max number of components in each patch
            'nb': 2,                            # number of background components per patch
            'p': 0,                             # order of AR indicator dynamics
            'ds_factor': 1,
            
            # init
            'ssub': 2,                          # spatial subsampling during initialization
            'tsub': 2,                          # temporal subsampling during initialization
            
            #motion
            #'dxy': dxy,                     
            'stride': 8,
            'motion_correct': False,
            'pw_rigid': False,
            
            #'max_shifts': [int(a/b) for a, b in zip(max_shift_um, dxy)],          # maximum allowed rigid shift in pixels
            #'stride': tuple([int(a/b) for a, b in zip(patch_motion_um, dxy)]),    # start a new patch for pw-rigid motion correction every x pixels
            'overlaps': (8, 8),               # overlap between patches (size of patch in pixels: strides+overlaps)
            'max_deviation_rigid': 3,           # maximum deviation allowed for patch with respect to rigid shifts
            
            #online
            'init_batch': 300,                  # number of frames for initialization
            'init_method': 'bare',              # initialization method
            'update_freq': 1000,                 # update every shape at least once every update_freq steps
            'n_refit': 1,
            'epochs': 1,                        # number of times to go over data, to refine shapes and temporal traces
            #'simultaneously': True,             # demix and deconvolve simultaneously
            'use_dense': True,
            
            #make things more memory efficient
            'memory_efficient': False,
            'block_size_temp': 5000,          
            'num_blocks_per_run_temp': 20,
            'block_size_spat': 5000,
            'num_blocks_per_run_spat': 20,
            
            #quality
            'min_SNR': 2.5,                     # minimum SNR for accepting candidate components
            'rval_thr': 0.85,                   # space correlation threshold for accepting a component
            'rval_lowest': 0,
            'sniper_mode': True,                # flag for using CNN
            'use_cnn': True,
            'thresh_CNN_noisy': 0.5,           # CNN threshold for candidate components
            'min_cnn_thr': 0.8,                # threshold for CNN based classifier
            'cnn_lowest': 0.3,                  # neurons with cnn probability lower than this value are rejected
            
            #display
            'show_movie': False,
            'save_online_movie': False,
            'movie_name_online': "test_mp4v.avi"
    }
    
    opts = cnmf.params.CNMFParams(params_dict=params_dict)
    
    if ~fname.endswith('.h5'):
      opts.change_params({'motion_correct':True})
      opts.change_params({'pw_rigid':True})
    
    Cn = cm.load(fname_memmap, subindices=slice(0,None,5)).local_correlations(swap_dim=False)
    
    Yr, dims, T = cm.load_memmap(fname_memmap)
    Y = np.reshape(Yr.T, [T] + list(dims), order='F')
    
    
### ------------------ 1st run ------------------ ###
    # %% fit with online object
    cnm = cnmf.online_cnmf.OnACID(params=opts)
    cnm.fit_online()
    
    ### %% evaluate components (CNN, SNR, correlation, border-proximity)
    cnm.estimates.evaluate_components(Y,opts)
    cnm.estimates.plot_contours(img=Cn, idx=cnm.estimates.idx_components)   ## plot contours, need that one to get the coordinates
    idx_border = [] 
    for n in cnm.estimates.idx_components:
      if (cnm.estimates.coordinates[n]['CoM'] < border_thr).any() or (cnm.estimates.coordinates[n]['CoM'] > (cnm.estimates.dims[0]-border_thr)).any():
        idx_border.append(n)
    cnm.estimates.idx_components = np.setdiff1d(cnm.estimates.idx_components,idx_border)
    cnm.estimates.idx_components_bad = np.union1d(cnm.estimates.idx_components_bad,idx_border)
    idx_border = None
    cnm.estimates.select_components(use_object=True)                        #%% update object with selected components
    

### ------------------ 2nd run ------------------ ###
    ## run over data again, with found ROIs, deconvolving
    opts_dict ={
          'p': 1,                             # order of AR indicator dynamics
          'init_method': 'seeded',              # initialization method
          'init_batch': 100,                  # number of frames for initialization
          'simultaneously': True,             # demix and deconvolve simultaneously
          'update_freq': 4000,                 # update every shape at least once every update_freq steps
          'thresh_CNN_noisy': 0.8,           # CNN threshold for candidate components
    }
    opts.change_params(params_dict=opts_dict)
    cnm = cnm.fit_online()
    
    ### %% evaluate components (CNN, SNR, correlation, border-proximity)
    cnm.estimates.evaluate_components(Y,opts)
    cnm.estimates.plot_contours(img=Cn, idx=cnm.estimates.idx_components)   ## plot contours, need that one to get the coordinates
    idx_border = [] 
    for n in cnm.estimates.idx_components:
      if (cnm.estimates.coordinates[n]['CoM'] < border_thr).any() or (cnm.estimates.coordinates[n]['CoM'] > (cnm.estimates.dims[0]-border_thr)).any():
        idx_border.append(n)
    cnm.estimates.idx_components = np.setdiff1d(cnm.estimates.idx_components,idx_border)
    cnm.estimates.idx_components_bad = np.union1d(cnm.estimates.idx_components_bad,idx_border)
    idx_border = None
    cnm.estimates.select_components(use_object=True)                        #%% update object with selected components
    
    #cnm.estimates.detrend_df_f(quantileMin=8, frames_window=250)            #%% Extract DF/F values
    print('Number of components:' + str(cnm.estimates.A.shape[-1]))
    
    #%% store results in matlab array for further processing
    idx_keep = cnm.estimates.idx_components
    results = dict(A=cnm.estimates.A.todense(),
                   C=cnm.estimates.C,
                   S=cnm.estimates.S,
                   Cn=Cn,
                   YrA=cnm.estimates.YrA,
                   b=cnm.estimates.b,
                   f=cnm.estimates.f)
    hdf5storage.write(results, '.', svname[0], matlab_compatible=True)
    
    #cnm.estimates.view_components(img=Cn, idx=cnm.estimates.idx_components)
    
    print("Total time taken: " +  str(time.time()-t_start))
    
    os.remove(fname_memmap)
    #return cnm, Cn, opts
    
# %%
# This is to mask the differences between running this demo in Spyder
# versus from the CLI
#if __name__ == "__main__":
    #[cnm, Cn, opts] = main()
    #main()

#run_CaImAn_mouse("/home/wollex/Data/Documents/Uni/2016-XXXX_PhD/Japan/Work/Data/M879/",True)

#if ~onAcid:
  ## first we create a motion correction object with the specified parameters
  #c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None, single_thread=False)
  #mc = MotionCorrect(fnames_memmap, dview=dview, **opts.get_group('motion'))
  
  ## %% Run (piecewise-rigid motion) correction using NoRMCorre
  #mc.motion_correct(save_movie=True)
  
  #fname_memmap = cm.save_memmap(mc.mmap_file, base_name='memmap_', order='C', data_type=data_type, border_to_0=border_to_0)  # exclude borders
  #cm.stop_server(dview=dview)