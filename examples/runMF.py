#!/usr/bin/env python
# cython: profile = True
from smash.models import MF
import scipy.io as io
#import cProfile

##################
# make a choice
##################

#dataset = 'movielens'
dataset = 'chembl'
#normalizeGradients = False 
normalizeGradients = True
#sideInfo = False
sideInfo = True
rowBatching = True

##################
# end of your choice
##################


# we currently have no side info for movielens
if dataset == 'movielens':
  sideInfo = False

mf = MF()

if (dataset == 'movielens'):
  trainMat = io.mmread("../data/movielens_train.mtx")
  valMat = io.mmread("../data/movielens_val.mtx")
  testMat = io.mmread("../data/movielens_test.mtx")
  
if (dataset == 'chembl'):  
  trainMat = io.mmread("../data/chembl-IC50-346targets_train.mtx")
  valMat = io.mmread("../data/chembl-IC50-346targets_val.mtx")
  testMat = io.mmread("../data/chembl-IC50-346targets_test.mtx")

mf.loadData(trainMat, valMat, testMat)
  
if (dataset == 'chembl' and sideInfo == True):  
  rowSideMat = io.mmread("../data/chembl-IC50-compound-feat.mm") # ecfp (+ ...)
  mf.loadRowSideInfo(rowSideMat)

# parameters common for the datasets
if normalizeGradients == True:
  mf.useNormalizedGradients()
if rowBatching == True:
  mf.useRowBatching()
mf.setVerbosity(1)
mf.setAlphaDecay(1.5)
mf.setLRDecay(1.0)
mf.setInitializationScalingFactor(0.3)
mf.setMinimalNumOfEpochs(3) # quasi constant
mf.setMomentum(0.8) # quasi constant
mf.setMaxEpoch(2000)
mf.setNumOfBatches(9) # quasi constant 
mf.setLatentDim(40)
mf.setEpsStop(1.e-5)

if (dataset == 'movielens'):
  mf.setInitialAlpha(0.32) # initial regularization parameter
  # here is normalization no so important as for chembl (no side info)
  if mf.normalizeGradients:
    mf.setInitialLR(320000.)
  else:
    mf.setInitialLR(320.)
  
if (dataset == 'chembl'):
  mf.setInitialAlpha(0.32) # initial regularization parameter
  # no normalization leads to a significantly worse RMSE
  if mf.normalizeGradients:
    mf.setInitialLR(3200.)
  else:                    
    mf.setInitialLR(10.)

mf.learn()
#cProfile.run('mf.learn()')