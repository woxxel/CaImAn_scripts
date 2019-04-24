import numpy as np
import logging
import time
import imp

from scipy import sparse
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.io import savemat

import h5py
import hdf5storage

import caiman as cm
from caiman.base.rois import register_ROIs

#imp.load_source("register_ROIs","../CaImAn/caiman/base/rois.py")

logging.basicConfig(format=
                    "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s]"\
                    "[%(process)d] %(message)s",
                    level=logging.ERROR)


def compare_old_vs_new(pathMouse,sessions=None,pl=False):

#def compare_old_vs_new(A_old,A_new,sessions=None,plt=False):
  
  pathResults_old = 'resultsCNMF_MF1_LK1.mat'
  pathResults_new = 'results_OnACID.mat'
  pathSave = 'matching_old_new.mat'
  
  nS = sessions[1]-sessions[0]+1
  
  t_start = time.time()
  for s in range(sessions[0],sessions[1]+1):
    
    t_start_s = time.time()
    print('---------- Now matching session %d ----------'%s)
    dims = (512,512)
    
    path_old = '%sSession%02d/%s' % (pathMouse,s,pathResults_old)
    path_new = '%sSession%02d/%s' % (pathMouse,s,pathResults_new)
    svname = '%sSession%02d/%s' % (pathMouse,s,pathSave)
    
    #print(path_old)
    #print(path_new)
    
    f = h5py.File(path_old,'r')
    A_old = sparse.csc_matrix((f['A2']['data'], f['A2']['ir'], f['A2']['jc']))
    f.close()
    
    f = loadmat(path_new)
    A_new = f['A'].reshape(-1,dims[0],dims[1]).transpose(2,1,0).reshape(dims[0]*dims[1],-1)
    f.close()
    
    N = A_old.shape[1]
    A_old.resize(dims[0]*dims[1],N)
    print(A_old.shape)
    print(A_new.shape)
    
    #return A_old, A_new
    [matched_ROIs1, matched_ROIs2, non_matched1, non_matched2, performance, _] = cm.base.rois.register_ROIs(A_old, A_new, dims, thresh_cost=.5, plot_results=pl)
    
    #print(performance)
    
    results = dict(matched_ROIs1=matched_ROIs1,
                   matched_ROIs2=matched_ROIs2,
                   non_matched1=non_matched1,
                   non_matched2=non_matched2,
                   performance=performance)
    hdf5storage.write(results, '.', svname, matlab_compatible=True)
    
    print('---------- finished matching session %d.\t time taken: %s ----------'%(s,str(time.time()-t_start_s)))
  
  print('---------- all done. Overall time taken: %s ----------'%str(time.time()-t_start))
  return matched_ROIs1, matched_ROIs2, non_matched1, non_matched2, performance
  
  
  

def match_ROIs_test(pathMouse,sessions=None,pl=False):
  
  
  if isinstance(pathMouse,str):
    assert isinstance(sessions,tuple), 'Please provide the numbers of sessions as a tuple of start and end session to be matched'
    pathResults = 'results_OnACID.mat'
    path = [('%sSession%02d/%s' % (pathMouse,i,pathResults)) for i in range(sessions[0],sessions[1]+1)]
    pathSave = pathMouse + 'results_matching_old.mat'
  
  nS = len(path)
  
  t_start = time.time()
  
  A = [[]]*nS
  Cn = [[]]*nS
  for s in range(nS):
    print(path[s])
    f = loadmat(path[s])
    A[s] = f['A']
    Cn[s] = f['Cn']
    
    print('# ROIs: '+str(A[s].shape[1]))
    
  thr_cost = 0.3
  print("Start matching")
  print(nS)
  if nS == 2:
    [matched_ROIs1, matched_ROIs2, non_matched1, non_matched2, performance, _, scores, shifts_matched] = cm.base.rois.register_ROIs(A[0], A[1], Cn[0].shape, template1=Cn[0], template2=Cn[1], thresh_cost=thr_cost, plot_results=pl)
    print(performance)
    print("Time taken: %s" % str(time.time()-t_start))
    return matched_ROIs1, matched_ROIs2, non_matched1, non_matched2, performance, scores, shifts_matched
  else:
    [A_union, assignments, matchings, shifts] = cm.base.rois.register_multisession(A, Cn[0].shape, templates=Cn, thresh_cost=thr_cost, plot_results=pl)
    print("Time taken: %s" % str(time.time()-t_start))
    
    results = dict(A_union=A_union,
                   assignments=assignments,
                   matchings=matchings,
                   shifts=shifts)
    savemat(pathSave, results)
    
    return A_union, assignments, matchings, shifts
  
  
  
  
  
  