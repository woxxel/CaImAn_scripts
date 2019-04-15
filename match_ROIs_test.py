

import numpy as np
#import os
import logging
import h5py
import time
from scipy import sparse
import caiman as cm
from caiman.base.rois import register_ROIs
import matplotlib.pyplot as plt

logging.basicConfig(format=
                    "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s]"\
                    "[%(process)d] %(message)s",
                    level=logging.debug)

def match_ROIs_test(path,A_key,Cn_key):
  
  if not isinstance(path,list):
    raise("Please provide the path as a list")
  nS = len(path)
  
  t_start = time.time()
  
  if not (len(A_key) == nS):
    A_key = list(A_key) * nS
  if not (len(Cn_key) == nS):
    Cn_key = list(Cn_key) * nS
  
  A = [[]]*nS
  Cn = [[]]*nS
  for s in range(nS):
    
    f = h5py.File(path[s][0],'r')
    if len(f[A_key[s]])==3:
      A[s] = np.array(sparse.csc_matrix((f[A_key[s]]['data'], f[A_key[s]]['ir'], f[A_key[s]]['jc'])).todense()).copy()
    else:
      A[s] = f[A_key[s]].value.transpose().copy()
    
    if len(path[s])>1:
      f.close()
      f = h5py.File(path[s][1],'r')
    Cn[s] = f[Cn_key[s]].value
    f.close()
  
  ### make sure, arrays have the same number of pixels
  max_dim = max(A[0].shape[0],A[1].shape[0])
  for s in range(nS):
    old_dim = A[s].shape[0]
    N = A[s].shape[1]
    A[s].resize(max_dim,N)
    
    if old_dim!=max_dim:
      A[s] = A[s].reshape(512,512,N).transpose(1,0,2).reshape(max_dim,N)
    else:
      Cn[s] = Cn[s].transpose(1,0)
    
    #plt.figure()
    #plt.imshow(Cn[s])
    #plt.show()
  
  
  print("Start matching")
  #[matched_ROIs1, matched_ROIs2, non_matched1, non_matched2, performance, A2] = cm.base.rois.register_ROIs(A[0], A[1], Cn[0].shape, plot_results=True)
  [matched_ROIs1, matched_ROIs2, non_matched1, non_matched2, performance, A2] = cm.base.rois.register_ROIs(A[0], A[1], Cn[0].shape, template1=Cn[0], template2=Cn[1], plot_results=True)
  print("Time taken: %s" % str(time.time()-t_start))
  
  return matched_ROIs1, matched_ROIs2, non_matched1, non_matched2, performance, A2