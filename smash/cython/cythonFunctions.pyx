from cython.parallel import parallel, prange
import numpy as np
cimport numpy as np


#cython: wraparound = False, boundscheck = False

#DTYPE = np.double
#ctypedef np.double_t DTYPE_t

#ITYPE = np.int16 
#ctypedef unsigned short ITYPE_t

#@cython.boundscheck(False) # turn off bounds-checking for entire function
#@cython.wraparound(False)  # turn off negative index wrapping for entire function
def aggregate(np.ndarray[int, ndim = 1] index, np.ndarray[np.float64_t, ndim = 1] ata, np.ndarray[np.float64_t, ndim = 1] aa):
  #assert index.dtype == ITYPE and arrayToAggr.dtype == DTYPE  
  cdef int d
  cdef int N = ata.shape[0]
          
  for d in range(N):
    aa[index[d]] = aa[index[d]] + ata[d]
    
def aggregate2(np.ndarray[int, ndim = 1] index, np.ndarray[np.float64_t, ndim = 2] ata, np.ndarray[np.float64_t, ndim = 2] aa):
  #assert index.dtype == ITYPE and arrayToAggr.dtype == DTYPE  
  cdef int d, l
  cdef int N = ata.shape[0]
  cdef int L = ata.shape[1] 
  
  for d in xrange(N):
    for l in xrange(L):
      aa[index[d], l] = aa[index[d], l] + ata[d, l]
      
def gradFidelity(np.ndarray[int, ndim = 1] index, np.ndarray[np.float64_t, ndim = 1] pred, np.ndarray[np.float64_t, ndim = 1] rating, np.ndarray[np.float64_t, ndim = 2] factor, np.ndarray[np.float64_t, ndim = 2] grad):      
  cdef int d, l
  cdef int N = grad.shape[0]
  cdef int L = grad.shape[1]
  cdef np.float64_t mult
  
  for d in xrange(N):
    mult = 2.*(pred[d] - rating[d])
    for l in xrange(L):
      grad[d, l] = mult * factor[index[d], l]
      
def aggregateMP(np.ndarray[int, ndim = 1] index, np.ndarray[np.float64_t, ndim = 1] ata, np.ndarray[np.float64_t, ndim = 1] aa):
  #assert index.dtype == ITYPE and arrayToAggr.dtype == DTYPE  
  cdef int d
  cdef int N = ata.shape[0]

  with nogil, parallel():        
    for d in prange(N, schedule = 'guided'):
      aa[index[d]] = aa[index[d]] + ata[d]      
      
def aggregate2MP(np.ndarray[int, ndim = 1] index, np.ndarray[np.float64_t, ndim = 2] ata, np.ndarray[np.float64_t, ndim = 2] aa):
  #assert index.dtype == ITYPE and arrayToAggr.dtype == DTYPE  
  cdef int d, l
  cdef int N = ata.shape[0]
  cdef int L = ata.shape[1] 
  
  with nogil, parallel():
    for d in prange(N, schedule = 'guided'):
      for l in xrange(L):
        aa[index[d], l] = aa[index[d], l] + ata[d, l]
        

def gradFidelityMP(np.ndarray[int, ndim = 1] index, np.ndarray[np.float64_t, ndim = 1] pred, np.ndarray[np.float64_t, ndim = 1] rating, np.ndarray[np.float64_t, ndim = 2] factor, np.ndarray[np.float64_t, ndim = 2] grad):      
  cdef int d, l
  cdef int N = grad.shape[0]
  cdef int L = grad.shape[1]
  cdef np.float64_t mult
  
  with nogil, parallel():
    for d in prange(N, schedule = 'guided'):
      mult = 2.*(pred[d] - rating[d])
      for l in xrange(L):
        grad[d, l] = mult * factor[index[d], l]
      

            
        