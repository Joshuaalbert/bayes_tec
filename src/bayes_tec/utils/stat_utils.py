import numpy as np
from scipy.optimize import fmin,minimize

def log_normal_solve(mean,std):
    mu = np.log(mean) - 0.5*np.log((std/mean)**2 + 1)
    sigma = np.sqrt(np.log((std/mean)**2 + 1))
    return mu, sigma
    

def log_normal_solve_fwhm(a,b,D=0.5):
    assert b > a
    lower = np.log(a)
    upper = np.log(b)
    d = upper - lower #2 sqrt(2 sigma**2 ln(1/D))
    sigma2 = 0.5*(0.5*d)**2/np.log(1./D)
    s = upper + lower #2 (mu - sigma**2)
    mu = 0.5*s + sigma2
    return mu, np.sqrt(sigma2)

def gamma_prior(mode,std):
    """
    In general you should prefer the log_normal prior.
    """
    a = std/mode#sqrt(k)/(k-1)
    shape = (2* a**2 + np.sqrt((4 * a**2 + 1)/a**4) * a**2 + 1)/(2 *a**2)
    scale = std/np.sqrt(shape)
    return gp.priors.Gamma(shape,scale)
