

import numpy as np
#import os

import h5py
import time
from scipy import sparse
import caiman as cm
from caiman.base.rois import register_ROIs

def match_ROIs_test(path,A_key,Cn_key):
  
  if not isinstance(path,list):
    raise("Please provide the path as a list")
  nS = len(path)
  print(nS)
  
  t_start = time.time()
  
  if not (len(A_key) == nS):
    A_key = list(A_key) * nS
  print(A_key)
  
  if not (len(Cn_key) == nS):
    Cn_key = list(Cn_key) * nS
  print(Cn_key)
  
  #res_name = "results_OnACID.mat"
  
  #pathMouse = "/media/wollex/Analyze_AS3/Data/879/"
  #pathSession = pathMouse + "Session01/"
  #pathSession = "/home/wollex/Data/Documents/Uni/2016-XXXX_PhD/Japan/Work/Data/M879/Session01/"
  #fname = pathSession + res_name
  #print(fname)
  A = [[]]*nS
  Cn = [[]]*nS
  for s in range(nS):
    
    f = h5py.File(path[s][0],'r')
    if len(f[A_key[s]])==3:
      A[s] = np.array(sparse.csc_matrix((f[A_key[s]]['data'], f[A_key[s]]['ir'], f[A_key[s]]['jc'])).todense())
    else:
      A[s] = f[A_key[s]].value.transpose()
    print(A[s].shape)
    
    if len(path[s])>1:
      f.close()
      f = h5py.File(path[s][1],'r')
    Cn[s] = f[Cn_key[s]].value
    f.close()
  
  ### make sure, arrays have the same number of pixels
  max_dim = max(A[0].shape[0],A[1].shape[0])
  for s in range(nS):
    A[s].resize(max_dim,A[s].shape[1])
    print(A[s].shape)
  #pathSession = pathMouse + "Session02/"
  ##pathSession = "/home/wollex/Data/Documents/Uni/2016-XXXX_PhD/Japan/Work/Data/M879/Session02/"
  #fname = pathSession + res_name
  #print(fname)
  
  #f = h5py.File(fname,'r')
  ##A2 = np.array(sparse.csc_matrix((f[A_key]['data'], f[A_key]['ir'], f[A_key]['jc'])).todense())
  #A2 = f['A'].value.transpose()
  #Cn2 = f['Cn'].value
  #f.close()
  
  #print(A1.shape)
  #print(A2.shape)
  cm.base.rois.register_ROIs(A[0], A[1], Cn[0].shape, template1=Cn[0], template2=Cn[1], plot_results=True)
  print("Time taken: %s" % str(time.time()-t_start))