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
def run_CaImAn_mouse(pathMouse):
  
  for f in os.listdir(pathMouse):
    if f.startswith("Session"):
      pathSession = pathMouse + f + '/'
      print("\t Session: "+pathSession)
      run_CaImAn_session(pathSession)
  
  
def run_CaImAn_session(pathSession):
    
    #plt.close('all')
    pass  # For compatibility between running under Spyder and the CLI

# %% load data

    #pathSession = "/media/wollex/Analyze_AS3/Data/879/Session01/"
    #pathSession = "/home/wollex/Data/Documents/Uni/2016-XXXX_PhD/Japan/Work/Data/M879/Session01/"
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
# %% set up some parameters
    
    border_thr = 5    # minimal distance of centroid to border
    
    patch_size = 32  # size of patch
    stride = 4  # amount of overlap between patches
    
    # set up CNMF initialization parameters
    params_dict ={
            
            #general data
            'fnames': fname,
            'fr': 15,
            'decay_time': 0.47,
            'gSig': [6, 6],  # expected half size of neurons
            
            #model/analysis
            'rf': patch_size//2,
            'stride': stride,
            'K': 200,                           # max number of components in each patch
            'nb': 2,                            # number of background components
            'p': 1,                             # order of AR indicator dynamics
            #'border_pix': 5,                   # Number of pixels to exclude around each border
            
            #motion
            'motion_correct': True,
            'pw_rigid': True,                   # flag for performing pw-rigid motion correction
            
            #online
            'init_batch': 300,                  # number of frames for initialization
            'init_method': 'bare',              # initialization method

            'update_freq': 200,                 # update every shape at least once every update_freq steps
            'n_refit': 1,
            'epochs': 2,                        # number of times to go over data, to refine shapes and temporal traces
            
            #quality
            'min_SNR': 2.5,                     # minimum SNR for accepting candidate components
            'rval_thr': 0.85,                   # space correlation threshold for accepting a component
            'sniper_mode': True,                # flag for using CNN
            'use_cnn': True,
            'thresh_CNN_noisy': 0.65,           # CNN threshold for candidate components
            'min_cnn_thr': 0.99,                # threshold for CNN based classifier
            'cnn_lowest': 0.1,                  # neurons with cnn probability lower than this value are rejected
            
            #display
            'show_movie': True,
            'save_online_movie': False,
            'movie_name_online': "test_mp4v.avi"
    }
    
    opts = cnmf.params.CNMFParams(params_dict=params_dict)
    
    if fname.endswith('.h5'):
      opts.change_params({'motion_correct':False})
      opts.change_params({'pw_rigid':False})
    else:
      opts.change_params({'motion_correct':True})
      opts.change_params({'pw_rigid':True})
    
# %% fit with online object
    cnm = cnmf.online_cnmf.OnACID(params=opts)
    cnm.fit_online()
    
    cnm.estimates.evaluate_components_CNN(opts)
    
    logging.info('Number of components:' + str(cnm.estimates.A.shape[-1]))
    
    
    #%% test this a little, so I can tell for sure, what is a good threshold
    
    #%% then, delete "bad" components from A, C and YrA before storing
    
# %% plot contours
    Cn = cm.load(fname, subindices=slice(0,None,5)).local_correlations(swap_dim=False)
    
# %% plot results
    cnm.estimates.plot_contours(img=Cn, idx=cnm.estimates.idx_components)   ## need that one to get the coordinates
    #cnm.estimates.view_components(img=Cn, idx=cnm.estimates.idx_components)
    
    idx_border = [] 
    for n in cnm.estimates.idx_components:
      #print(n)
      if (cnm.estimates.coordinates[n]['CoM'] < border_thr).any() or (cnm.estimates.coordinates[n]['CoM'] > (cnm.estimates.dims[0] -border_thr)).any():
        idx_border.append(n)
    
    cnm.estimates.idx_components = np.setdiff1d(cnm.estimates.idx_components,idx_border)
    cnm.estimates.idx_components_bad = np.union1d(cnm.estimates.idx_components_bad,idx_border)
    
    idx_keep = cnm.estimates.idx_components
    
    results = dict(A=cnm.estimates.A[:,idx_keep].todense(),
                   C=cnm.estimates.C[idx_keep,:],
                   Cn=Cn,
                   YrA=cnm.estimates.YrA[idx_keep,:],
                   b=cnm.estimates.b,
                   f=cnm.estimates.f)
    
    hdf5storage.write(results, '.', svname[0], matlab_compatible=True)
    
    
    #return cnm, Cn, opts
# %%
# This is to mask the differences between running this demo in Spyder
# versus from the CLI
#if __name__ == "__main__":
    #[cnm, Cn, opts] = main()
    #main()


      
    
###if os.path.exist(...)
  ###if extension == 'tif'
    ###c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None, single_thread=False)
    ###mc = MotionCorrect(fname, dview=dview, **opts.get_group('motion'))
    ###mc.motion_correct(save_movie=True)
    ###cm.stop_server(dview=dview)
    
    ###fname_new = cm.save_memmap(mc.mmap_file, base_name='memmap_', order='C',
                            ###border_to_0=border_to_0)  # exclude borders
  ###else
    ###fname_new = cm.save_memmap(fname, base_name='memmap_', order='C',
                            ###border_to_0=border_to_0)  # exclude borders

##fname_new = [pathSession + "memmap__d1_512_d2_512_d3_1_order_C_frames_8989_.mmap"]
#### now load the file
##Yr, dims, T = cm.load_memmap(fname_new[0])
###images = np.reshape(Yr.T, [T] + list(dims), order='F')  # load frames in python format (T x X x Y)

##opts.change_params({'fnames':fname_new})


#print('raw image: ' fname_new)

