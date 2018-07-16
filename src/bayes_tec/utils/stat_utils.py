import numpy as np
from scipy.optimize import fmin,minimize

def log_normal_solve(mode,uncert):
    def func(x):
        mu,sigma2 = x
        mode_ = np.exp(mu-sigma2)
        var_ = (np.exp(sigma2 ) - 1) * np.exp(2*mu + sigma2)
        return (mode_ - mode)**2 + (var_ - uncert**2)**2
    res = minimize(func,(mode,uncert**2))
#         res = fmin(func, (mode,uncert**2))
    return res.x[0],np.sqrt(res.x[1])

def gamma_prior(mode,std):
    """
    In general you should prefer the log_normal prior.
    """
    a = std/mode#sqrt(k)/(k-1)
    shape = (2* a**2 + np.sqrt((4 * a**2 + 1)/a**4) * a**2 + 1)/(2 *a**2)
    scale = std/np.sqrt(shape)
    return gp.priors.Gamma(shape,scale)
