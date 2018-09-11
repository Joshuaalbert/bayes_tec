import numpy as np
from .bayes_hp_opt import BayesHPOpt

def test():
    def ackley_2d(x,y):
        return -20*np.exp(-0.2*np.sqrt(0.5*(x**2 + y**2))) - np.exp(0.5*(np.cos(2*np.pi*x) + np.cos(2*np.pi*y))) + np.exp(1) + 20.
    
    def ackley_1d(x):
        return -20*np.exp(-0.2*np.sqrt(0.5*(x**2))) - np.exp(0.5*(np.cos(2*np.pi*x))) + np.exp(1) + 20.


    bo = BayesHPOpt(ackley_1d,init=None)
    bo.add_continuous_param('x',-10,10)
    bo.run('test_save.hdf5', init_design_size=6, n_iter=10, plot=True, likelihood_uncert=0.01)

