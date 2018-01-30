import numpy as np
import time as time

class SGD(object):
  def __init__(self):
    SGD.setAlgorithmPars(self)
        
  def setAlgorithmPars(self):
    
    np.random.RandomState = 0
    self.setVerbosity(1)
    self.setInitialAlpha(0.32) # Regularization parameter
    self.setInitialLR(320.)
    self.setLatentDim(10)
    self.setInitializationScalingFactor(1.)
    self.setMomentum(0.8)
    self.setMaxEpoch(2000)
    self.setNumOfBatches(9)
    self.setEpsStop(1.e-5)
    self.setMinimalNumOfEpochs(3) # quasi constant
    
  def initializeMemory(self):
    self.err_train = np.zeros(self.maxepoch)
    self.err_valid = np.zeros(self.maxepoch)
    self.err_test = np.zeros(self.maxepoch)
  
  def setVerbosity(self, ver = 1):
    self.verbosity = ver    
              
  def setInitialAlpha(self, alpha = 0.2):
    self.alpha = alpha

  def setInitialLR(self, ilr = 50.):
    self.LR = ilr

  def setLatentDim(self, ld = 10):
    self.npar = ld
    
  def setInitializationScalingFactor(self, isf = 1.):
    self.isf = isf                     

  def setMomentum(self, m = 0.8):
    self.momentum = m
    
  def setMaxEpoch(self, mpoch = 1000):
    self.maxepoch = mpoch
    self.initializeMemory()
    
  def setNumOfBatches(self, nb = 9):
    self.numbatches = nb
    
  def setEpsStop(self, es = 1e-6):
    self.epsStop = es
    
  def setMinimalNumOfEpochs(self, n = 3):
    self.minNumOfEpochs = n
    
  def prepareBatch(self, batch):
    raise NotImplementedError('subclasses must override prepareBatch()!')
    
  def computeRegTerms(self):
    raise NotImplementedError('subclasses must override computeRegTerms()!')
    
  def computePredictions(self):
    raise NotImplementedError('subclasses must override computePredictions()!')
  
  def computeLikelihoodGrads(self):
    raise NotImplementedError('subclasses must override computeLikelihoodGrads()!')
    
  def computeRegGrads(self):
    raise NotImplementedError('subclasses must override computeRegGrads()!')
      
  def aggregateGrads(self):
    raise NotImplementedError('subclasses must override aggregateGrads()!')
      
  def computeStochGrads(self):
    raise NotImplementedError('subclasses must override computeStochGrads()!')
      
  def updateMomentums(self):
    raise NotImplementedError('subclasses must override updateMomentums()!')
          
  def oneGradStep(self):
    raise NotImplementedError('subclasses must override oneGradStep()!')

  def permuteData(self):
    raise NotImplementedError('subclasses must override permuteData()!')  
          
  def trainOneBatch(self, batch):
    #print('batch %d \r' % (batch))
    self.prepareBatch(batch)
    self.computeRegTerms()
    self.computePredictions()
    self.computeStochGrads() 
    self.updateMomentums()
    self.oneGradStep()
      
  def trainError(self):
    raise NotImplementedError('subclasses must override trainError()!')

  def validError(self):
    raise NotImplementedError('subclasses must override validError()!')

  def testError(self):
    raise NotImplementedError('subclasses must override testError()!')

  def PoldToP(self):
    raise NotImplementedError('subclasses must override PoldToP()!')

  def PtoPold(self):
    raise NotImplementedError('subclasses must override PtoPold()!')  
  
  def updateParameters(self, epoch):
    if(epoch > self.minNumOfEpochs and (self.err_valid[epoch - 1] - self.err_valid[epoch])/self.err_valid[epoch - 1] < self.epsStop):
      self.PoldToP() 
      self.breakSignal = True
  
  def learn(self):
    self.setInitialValues()
    
    self.duration = time.time()
    self.breakSignal = False 
    for epoch in range(self.maxepoch):
      self.epoch = epoch
      # randomly permute the data
      self.permuteData()
      # train the individual batches
      self.PtoPold()
      for batch in range(self.numbatches):
          self.trainOneBatch(batch)
          if self.verbosity > 1:
            print('Batch %d: Training RMSE %6.4f; Validation RMSE %6.4f; Test RMSE %6.4f' % (batch, self.trainError(), self.validError(), self.testError()))

      self.err_train[epoch] = self.trainError()
      self.err_valid[epoch] = self.validError()
      self.err_test[epoch] = self.testError()
      if self.verbosity > 0:
        print('Epoch %d: Training RMSE %6.4f; Validation RMSE %6.4f; Test RMSE %6.4f' % (epoch, self.err_train[epoch], self.err_valid[epoch], self.err_test[epoch]))
      
      self.updateParameters(epoch)
      
      if(self.breakSignal):
        break    
    self.duration = time.time() - self.duration