SMASH: Sampling with Monotone Annealing based Stochastic Homotopy
=================================================================

The contribution is few-folded:

First, a "simulated annealing" based SGD solver is implemented. Since 
our ultimate criterion is simplicity, we start from SGD with momentum.
We introduce continuation for the regularization parameter. This homotopy 
facilitates a convexification of the iterative process. The found local 
minina of the regularized cost functional (loss) are consequently close 
to the global one(s). Moreover, the convexified problem is more isotropic 
which leads to a rather quick convergence.

The initial regularization parameter is set above the optimal level 
of regularization (can be much higher). The optimal regularization 
is found automatically during the run of the method. In principle, 
no expensive hyperparameter search (or cross-validation) is necessary.

Further, we argue that the resulting SASGD produces samples around modes of 
the posterior. We employ this property and introduce Monte Carlo (MC) 
(and are plannning Quasi Monte Carlo (QMC)) sampling in hyperparameter space. 
The resulting embarasingly parallel aproximative sampler (except the prediction phase)
leads to accuracy comparable to fully Bayesian samplers.  

Installation
------------
 
 Run in the main folder:

```python setup.py install --user```

Examples
--------

Examples are located in ``examples`` directory. See the corresponding 
``examples/README.rst`` file there