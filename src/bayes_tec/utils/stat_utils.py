import numpy as np
from scipy.optimize import fmin,minimize

def log_normal_solve(mean,std):
    mu = np.log(mean) - 0.5*np.log((std/mean)**2 + 1)
    sigma = np.sqrt(np.log((std/mean)**2 + 1))
    return mu, sigma
    
def gamma_prior(mode,std):
    """
    In general you should prefer the log_normal prior.
    """
    a = std/mode#sqrt(k)/(k-1)
    shape = (2* a**2 + np.sqrt((4 * a**2 + 1)/a**4) * a**2 + 1)/(2 *a**2)
    scale = std/np.sqrt(shape)
    return gp.priors.Gamma(shape,scale)
