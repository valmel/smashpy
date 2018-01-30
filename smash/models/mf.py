import numpy as np
import scipy.linalg as la
import scipy.io as io
from scipy.sparse import csr_matrix
#import cProfile
from cythonFunctions import aggregate2, gradFidelity
from smash.so import SASGD

######################################################################
# The main model of the matrix factorization with/wo side information 
######################################################################

# SGD unfortunately rather closely couples model, data and its logic.
# A simple and efficient solution (arguably not so elegant as 
# a fully independent (SGD) solver) is to derive models from an abstract 
# solver class. The model has to implement the abstract member functions 
# of the solver class. This exercise is solver dependent. We however currently
# consider only SGD with momentum with adaptive regularization and 
# learning rate. The model handles all the data and latent model 
# parameters at one place. This gives the user complete freedom 
# to handle issues as he/she wishes.

class MF(SASGD):
  def __init__(self):
    super(MF, self).__init__()
    np.random.RandomState = 0
    self.hasRowSideInfo = False
    self.hasColSideInfo = False
    self.normalizeGradients = False
    self.rowBatching = False 
    
  def useRowBatching(self):
    self.rowBatching = True
    try:
      self.trainMat = self.trainMat.tocsr()
    except AttributeError:
      print('useRowBatching: please, first load the data')
      
  def useNormalizedGradients(self):
    self.normalizeGradients = True     
      
  def loadData(self, trainMat, valMat, testMat):
    self.trainMat = trainMat
    self.valMat = valMat
    self.testMat = testMat
   
    self.mean_rating = np.mean(self.trainMat.data)
 
    self.nTrainPairs = len(self.trainMat.data) # training data
    self.nValPairs = len(self.valMat.data) # validation data
    self.nTestPairs = len(self.testMat.data) # test data
            
    self.nrows = max(self.trainMat.row) + 1 # Number of rows (e.g. compounds)
    self.ncols = max(self.trainMat.col) + 1 # Number of targets (e.g. targets)
        
    self.normData = la.norm(self.trainMat.data)
    
  def loadRowSideInfo(self, rowSideMat):
    self.rowSideMat = rowSideMat
    self.nRowFeats = max(self.rowSideMat.col) + 1 # number of row features
    self.nRowSidePairs = len(self.rowSideMat.data)
    
    self.rowSideMat = self.rowSideMat.tocsr()
    self.rowSideMatT = self.rowSideMat.transpose().tocsr()
    self.hasRowSideInfo = True
   
  def setInitialValues(self):
    if self.rowBatching == True:
      self.N = self.nrows//self.numbatches
    else:
      self.N = self.nTrainPairs//self.numbatches
      
    self.U = self.isf/np.sqrt(self.npar) * np.random.randn(self.nrows, self.npar) # compounds
    self.V = self.isf/np.sqrt(self.npar) * np.random.randn(self.ncols, self.npar) # targets
    if self.hasRowSideInfo: 
      self.Us = np.zeros((self.nRowFeats, self.npar)) # ecfp's (beta)
      
    self.U_m = np.zeros((self.nrows, self.npar))
    self.V_m = np.zeros((self.ncols, self.npar))
    if self.hasRowSideInfo:
      self.Us_m = np.zeros((self.nRowFeats, self.npar))
      
    self.dU = np.zeros((self.nrows, self.npar))
    self.dV = np.zeros((self.ncols, self.npar))
    
    if self.hasRowSideInfo:
      self.dUs = np.zeros((self.nRowFeats, self.npar))
    
  def getU(self):
    return self.U

  def getV(self):
    return self.V

  def getUs(self):
    return self.Us
  
  def prepareBatch(self, batch):
    if self.rowBatching == True:
      self.bB = batch*self.N
      self.bE = min((batch + 1)*self.N, self.nrows) 
      self.tMbatch = csr_matrix((self.nrows, self.ncols))
      self.tMbatch = self.trainMatPer[self.bB:self.bE, :].tocoo() 
      self.bRowsPer = self.tMbatch.row + self.bB
      self.bRows = self.rp[self.bRowsPer] # bRows is indexing U
      self.bCols = self.tMbatch.col
      self.bVal = self.tMbatch.data
      
      if self.hasRowSideInfo:
        self.rsMbatch = self.rowSideMatPer[self.bRowsPer, :]
        self.rsMbatchT = self.rsMbatch.transpose()
    else:
      self.bB = batch*self.N
      self.bE = min((batch + 1)*self.N, self.nTrainPairs)
      self.bRows   = self.trainMat.row[self.bB : self.bE]
      self.bCols   = self.trainMat.col[self.bB : self.bE]
      self.bVal = self.trainMat.data[self.bB : self.bE]
      
      if self.hasRowSideInfo:
        self.rsMbatch = self.rowSideMat[self.bRows, :]
        self.rsMbatchT = self.rsMbatch.transpose()
      
    self.nBdata = len(self.bVal)
    self.gradL_U = np.zeros((self.nBdata, self.npar))
    self.gradL_V = np.zeros((self.nBdata, self.npar))
    
    if self.hasRowSideInfo:
      self.gradL_Us = np.zeros((self.nBdata, self.npar))  
    
  def computeRegTerms(self):
    self.regRows = np.sum(self.U[self.bRows,:]**2, 1)
    self.regCols = np.sum(self.V[self.bCols,:]**2, 1)
    if self.hasRowSideInfo:
      self.regFeats = np.sum(self.Us**2, 1)
    
  def computePredictions(self):
    self.rating = self.bVal - self.mean_rating
    ############## Compute Predictions ##############
    self.pred = np.sum(self.V[self.bCols,:]*self.U[self.bRows,:], 1) 
    if self.hasRowSideInfo:
      self.pred += np.sum(self.V[self.bCols,:]*(self.rsMbatch*self.Us), 1)
    self.fid = np.sum((self.pred - self.rating)**2)
    self.f =  self.fid + np.sum(0.5*self.alpha*(self.regRows + self.regCols))
    if self.hasRowSideInfo:
      self.f += np.sum(0.5*self.alpha*self.regFeats)
  
  def computeLikelihoodGrads(self):
    IO = np.tile(2.*(self.pred - self.rating), (self.npar, 1)).transpose()
    self.gradL_U = IO * self.V[self.bCols,:]
    self.gradL_V = IO * self.U[self.bRows,:]
    
    # cythonized version of the code above
    #gradFidelity(self.bCols, self.pred, self.rating, self.V, self.gradL_U)
    #gradFidelity(self.bRows, self.pred, self.rating, self.U, self.gradL_V)
    self.gradL_Us = self.rsMbatchT*self.gradL_U      
    
  def computeRegGrads(self):
    self.gradR_U = self.U[self.bRows,:]
    self.gradR_V = self.V[self.bCols,:]
    if self.hasRowSideInfo:
      self.gradR_Us = self.Us
      
  def aggregateGrads(self):
    for ii in range(len(self.bVal)):
      self.dU[self.bRows[ii],:] = self.dU[self.bRows[ii],:] + self.gradF_U[ii,:] 
      self.dV[self.bCols[ii],:] = self.dV[self.bCols[ii],:] + self.gradF_V[ii,:]
    
    # cythonized version of the code above
    #aggregate2(self.bRows, self.gradF_U, self.dU)
    #aggregate2(self.bCols, self.gradF_V, self.dV)
          
  def computeStochGrads(self):
    self.computeLikelihoodGrads()
    self.computeRegGrads()

    self.gradF_U = self.gradL_U + self.alpha *  self.gradR_U
    self.gradF_V = self.gradL_V + self.alpha *  self.gradR_V
    
    self.dU.fill(0.)
    self.dV.fill(0.)
    
    self.aggregateGrads()
    
    if self.hasRowSideInfo:
      self.dUs = self.gradL_Us + self.alpha *  self.gradR_Us
      
  def updateMomentums(self):
    if self.normalizeGradients:
      self.dU = self.dU/np.linalg.norm(self.dU)
      self.dV = self.dV/np.linalg.norm(self.dV)
    
    self.U_m = self.momentum * self.U_m + self.LR * self.dU/self.nBdata
    self.V_m = self.momentum * self.V_m + self.LR * self.dV/self.nBdata
    
    if self.hasRowSideInfo:
      if self.normalizeGradients:
        self.dUs = self.dUs/np.linalg.norm(self.dUs)
      self.Us_m = self.momentum * self.Us_m + self.LR * self.dUs/self.nBdata
          
  def oneGradStep(self):
    self.U =  self.U - self.U_m  
    self.V =  self.V - self.V_m
    if self.hasRowSideInfo:
      self.Us = self.Us - self.Us_m
    #print(self.Us.max())
    
  def trainError(self):
    # uses the last batch only
    batch = self.numbatches - 1
    self.prepareBatch(batch)
    self.computeRegTerms()
    self.computePredictions()
    return np.sqrt(self.fid/self.nBdata)  
    
  def predictVal(self):
    bRows = self.valMat.row
    bCols = self.valMat.col
    self.val_pred = np.sum(self.V[bCols,:]*self.U[bRows,:], 1) + self.mean_rating
    if self.hasRowSideInfo:
      self.val_pred += np.sum(self.V[bCols,:]*(self.rowSideMat[bRows, :]*self.Us), 1)
            
  def validError(self):
    self.predictVal()
    return np.sqrt(sum((self.val_pred - self.valMat.data)**2)/self.nValPairs)

  def validErrorPar(self, pred):
    return np.sqrt(sum((pred - self.valMat.data)**2)/self.nValPairs)

  def predictTest(self):
    bRows = self.testMat.row
    bCols = self.testMat.col
    self.test_pred = np.sum(self.V[bCols,:]*self.U[bRows,:], 1) + self.mean_rating
    if self.hasRowSideInfo:  
      self.test_pred += np.sum(self.V[bCols,:]*(self.rowSideMat[bRows, :]*self.Us), 1)
      
  def testError(self):
    self.predictTest()
    return np.sqrt(sum((self.test_pred - self.testMat.data)**2)/self.nTestPairs)

  def testErrorPar(self, pred):
    return np.sqrt(sum((pred - self.testMat.data)**2)/self.nTestPairs)

  def PoldToP(self):
    self.U = self.Uold
    self.V = self.Vold
    if self.hasRowSideInfo:
      self.Us = self.UsOld
    
  def PtoPold(self):
    self.Uold = self.U
    self.Vold = self.V
    if self.hasRowSideInfo:
      self.UsOld = self.Us
    
  def PglobToP(self):
    self.U = self.Uglob
    self.V = self.Vglob
    if self.hasRowSideInfo:
      self.Us = self.UsGlob
       
  def PtoPglob(self):
    self.Uglob = self.U
    self.Vglob = self.V
    if self.hasRowSideInfo:
      self.UsGlob = self.Us
       
  def EraseMomentum(self):
    self.U_m.fill(0.)
    self.V_m.fill(0.)
    if self.hasRowSideInfo:
      self.Us_m.fill(0.)
    
  def permuteData(self):
    if self.rowBatching == True:
      self.rp = np.random.permutation(self.nrows)
      self.trainMatPer = self.trainMat[self.rp, :]
      if self.hasRowSideInfo:
        self.rowSideMatPer = self.rowSideMat[self.rp, :]
    else:    
      rr = np.random.permutation(self.nTrainPairs)  
      self.trainMat.row = self.trainMat.row[rr]
      self.trainMat.col = self.trainMat.col[rr]
      self.trainMat.data = self.trainMat.data[rr]
  
  def saveFactors(self, fdir, index):
    nameU = fdir + 'U_' + index
    nameV = fdir + 'V_' + index
    if self.hasRowSideInfo:
      nameUs = fdir + 'Us_' + index    
    
    io.mmwrite(nameU, self.U)
    io.mmwrite(nameV, self.V)
    if self.hasRowSideInfo:
      io.mmwrite(nameUs, self.Us)