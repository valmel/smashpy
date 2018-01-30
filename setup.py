#!/usr/bin/env python
import os, sys
from setuptools import setup
from setuptools.command.install import install
from Cython.Build import cythonize

def postInstallCMD():
    from subprocess import call
    dir = os.path.dirname(os.path.realpath(__file__))
    print(dir)
    call([sys.executable, dir + '/data/movielensData.py'],
         cwd = dir + '/data')
    call([sys.executable, dir + '/data/chemblData.py'],
         cwd = dir + '/data')

class postInstall(install):
    """Post-installation for installation mode."""
    def run(self):
      install.run(self)
      self.execute(postInstallCMD, (), msg = "Running post install task")

setup(
  name = 'smash',
  version = '0.1.0',
  description = 'SMASH: Sampling with Monotone Annealing based Stochastic Homotopy',
  long_description = 'Implements a simulated annealing based SGD solver.'
                    +'The homotopy in the regularization parameters facilitates'
                    +'convexification of the problem. The found solutions are close'
                    +'to the global one(s). They represent samples from the posterior'
                    +'around its modes. The solver is used to train incomplete matrix'
		    +'factorization model with or without side information.'
		    +'This is a pure Python implementation (with alternatively a bit of Cython)'
		    +'made to be readable.',
                   
  url = 'https://github.com/valmel/smashpy',
  author = 'Valdemar Melicher',
  author_email = 'Valdemar.Melicher@UAntwerpen.be',
  license = 'MIT',
  classifiers = [
    'DevelopODment Status :: 3 - Alpha',
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering :: Mathematics',
    'Topic :: Scientific/Engineering :: Incomplete Matrix Factorization',
    'Topic :: Scientific/Engineering :: Collaborative Filtering', 
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3',
  ],
  keywords = 'MF SGD SA simulated annealing convexification sampling',               
  ext_modules = cythonize("smash/cython/*.pyx"),
  packages = ['smash', 'smash.so', 'smash.models', 'smash.sampling'],
  install_requires = ['numpy', 'scipy', 'pandas', 'mpi4py'],
  package_data = {
    'examples': ['examples/runMF.py', 'examples/runSMASH.py'],
  },
  cmdclass={
        'install': postInstall,
    },
)