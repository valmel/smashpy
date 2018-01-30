Examples
========

1. ```python runMF.py``` allows you to run a single instance of incomplete matrix 
factorization for **Movielens** or **ChEMBL** dataset (or whathever you supply). 
For the description of the datasets, see  `../data/README.rst`_  
 
2. ``mpirun -np n python runSMASH.py`` allows you to run MC sampling in 
hyperparameter space as defined in the file. Here ``n`` is number of independent 
mpi processes.

.. _`../data/README.rst`: ../data/README.rst