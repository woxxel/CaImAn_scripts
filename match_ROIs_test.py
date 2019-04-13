

#import numpy as np
#import os

import h5py
from scipy import sparse
import caiman as cm

def match_ROIs_test():
  
  A_key = 'A2'
  res_name = "resultsCNMF_MF1_LK1.mat"
  
  
  pathSession = "/home/wollex/Data/Documents/Uni/2016-XXXX_PhD/Japan/Work/Data/M879/Session09/"
  fname = pathSession + res_name
  print(fname)
  
  f = h5py.File(fname,'r')
  A1 = np.array(sparse.csc_matrix((f[A_key]['data'], f[A_key]['ir'], f[A_key]['jc'])).todense())
  Cn1 = f['Cn'].value
  
  pathSession = "/home/wollex/Data/Documents/Uni/2016-XXXX_PhD/Japan/Work/Data/M879/Session10/"
  fname = pathSession + res_name
  print(fname)
  
  f = h5py.File(fname,'r')
  A2 = np.array(sparse.csc_matrix((f[A_key]['data'], f[A_key]['ir'], f[A_key]['jc'])).todense())
  Cn2 = f['Cn'].value
  
  cm.base.rois.register_ROIs(A1, A2, 2, template1=Cn1, template2=Cn2, plot_results=True)
  