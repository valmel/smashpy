from smash.so import SGD

# derived class of the simple momentum SGD
class SASGD(SGD):
  def __init__(self):
    super(SASGD, self).__init__()
    SASGD.setAlphaDecay(self)
    SASGD.setLRDecay(self)
  
  def setAlphaDecay(self, alphaD = 1.5):
    self.alphaD = alphaD  
    
  def setLRDecay(self, LRD = 1.1):
    self.LRD = LRD  
          
  def updateParameters(self, epoch):
    if(epoch == 0):
      self.updatedEpoch = -2
      self.prevRMSEafterUpdate = float("inf")
      self.RMSEafterUpdate = self.prevRMSEafterUpdate - 1.
      
        #if(epoch == self.updatedEpoch + 1 and (self.err_valid[epoch - 1] - self.err_valid[epoch])/self.err_valid[epoch - 1] < self.epsStop):
    if(epoch == self.updatedEpoch + 1):
      self.prevRMSEafterUpdate = self.RMSEafterUpdate 
      self.RMSEafterUpdate = self.err_valid[epoch]
      
    if(self.RMSEafterUpdate > self.prevRMSEafterUpdate):
      self.PglobToP() 
      self.breakSignal = True

    if(epoch > self.minNumOfEpochs and self.err_valid[epoch] > self.err_valid[epoch - 1]):
      self.alpha = max(self.alpha/self.alphaD, 0.01)
      self.LR = self.LR/(self.alphaD*self.LRD)
      self.PoldToP()
      self.PtoPglob()
      self.EraseMomentum()
      self.updatedEpoch = epoch

      if self.verbosity > 0:
        print('alpha = %6.4f  LR = %6.4f  \n' % (self.alpha, self.LR))