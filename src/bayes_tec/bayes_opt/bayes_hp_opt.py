import gpflowopt as gpo                                                                                                                                                            
from gpflowopt.domain import ContinuousParameter, UnitCube                                                                                                                         
from .mgf import MomentGeneratingFunctionImprovement

import gpflow as gp                                                                                                                                                                
from gpflowopt.design import LatinHyperCube, FactorialDesign                                                                                                                         
from gpflowopt.acquisition import ExpectedImprovement                                                                                                                              
from gpflowopt import optim, BayesianOptimizer                                                                                                                                     
import h5py                                                                                                                                                                        
import numpy as np                                                                                                                                                                 
import os                                                                                                                                                                          
from ..logging import logging     
from ..utils.stat_utils import log_normal_solve
import pylab as plt
plt.style.use('ggplot')
                                                                                                                                                                                   
class BayesHPOpt(object):                                                                                                                                                          
    def __init__(self, objectivefx, fit_params = {}, folds=3, n_jobs=1, init = None, t=0.5,kw_params=True):
        self.kw_params=kw_params
        self.t=t
        self.folds = 3                                                                                                                                                             
        self.n_jobs = 1                                                                                                                                                            
        self._objectivefx = objectivefx                                                                                                                                            
        self.domain = []                                                                                                                                                           
        self.domain_type = []                                                                                                                                                      
        self.fit_params = fit_params                                                                                                                                               
        if init is not None:                                                                                                                                                       
            with h5py.File(init) as f:                                                                                                                                             
                self.init_X = list(f['X'][...])                                                                                                                                    
                self.init_Y = list(f['Y'][...])                                                                                                                                    
                assert len(self.init_X)==len(self.init_Y)                                                                                                                          
        else:                                                                                                                                                                      
            self.init_X = []                                                                                                                                                       
            self.init_Y = []                                                                                                                                                       
        assert callable(objectivefx)                                                                                                                                               
                                                                                                                                                                                   
    def save(self, save_path, clobber=False):                                                                                                                                      
        """Save to hdf5 file"""                                                                                                                                                    
        if os.path.isfile(save_path) and not clobber:                                                                                                                              
            raise ValueError("{} exists already and not clobber".format(save_path))                                                                                                
                                                                                                                                                                                   
        with h5py.File(save_path, 'w') as f:                                                                                                                                       
            f['X'] = np.stack(self.X, axis=0)# N, D                                                                                                                                
            f['Y'] = np.stack(self.Y, axis=0)# N, 1                                                                                                                                
                                                                                                                                                                                   
    def add_continuous_param(self, name, vmin, vmax, log=False):                                                                                                                   
        """                                                                                                                                                                        
        Add a parameter type for optimization.
        name : str keyword name
        vmin : float lower end of range
        vmax : float upper end of range
        log : bool, whether to use log space (then vmin > 0)
        """
        if vmin > vmax:
            vmax_ = vmin
            vmin = vmax
            vmax = vmax_

        if log:
            assert vmin > 0.
            vmin =  np.log(vmin)
            vmax = np.log(vmax)
            self.domain_type.append('log')
        else:
            self.domain_type.append('linear')

        self.domain.append(ContinuousParameter(name,vmin,vmax))

    def add_integer_param(self, name, vmin, vmax):
        ### TODO replace with nunitcube and parametrized bit sampling
        self.domain_type.append("int")
        self.domain.append(ContinuousParameter(name, int(vmin), int(vmax)))

    def add_binary_param(self, name):
        self.domain_type.append('bin')
        self.domain.append(ContinuousParameter(name,0,1))

    def _objective(self,kwargs):
        if self.kw_params:
            return self._objectivefx(**kwargs)

        return self._objectivefx(kwargs)


    def get_kwargs(self, params):
        params = params.flatten()
        args = {}
        for c,p,t in zip(self.domain, params, self.domain_type):
            if t == 'log':
                args[c.label] = np.exp(p)
            elif t == 'bin':
                args[c.label] = 1. if np.random.uniform() < p else 0.
            elif t == 'int':
                args[c.label] = int(p)
            else:
                args[c.label] = p
        return args

    def objective(self, params):
        logging.warning("=== Starting iteration {} ===".format(self.iter))
        params = params.flatten()
        self.X.append(params)
        if self.kw_params:
            args = {}
            for c,p,t in zip(self.domain, params, self.domain_type):
                if t == 'log':
                    args[c.label] = np.exp(p)
                elif t == 'bin':
                    args[c.label] = 1. if np.random.uniform() < p else 0.
                elif t == 'int':
                    args[c.label] = int(p)
                else:
                    args[c.label] = p
        else:
            args = params
        logging.warning("Running with {}".format(args))
        y = self._objective(args) # scalar
        self.Y.append(np.array([y]))
        self.save(self.save_path, True)
        logging.warning("=== Result {} ===".format(self.iter))
        logging.warning("{} : {}".format(args, y))
        self.iter += 1
        return np.array([[y]]) # 1, 1

    def print_top_k(self,k=5):
        f = self.ei.data[1][:,0]
        x = self.ei.data[0]
        a = np.argsort(f)
        for j, i in enumerate(a[:k]):
            kwargs = self.get_kwargs(x[i])
            logging.warning("{} ({})  : {} -> {}".format(j, i, f[i], kwargs))

        mean_x = x[a[:k]].mean(0)
        std_x = x[a[:k]].std(0)
        mean = self.get_kwargs(mean_x)
        lb = self.get_kwargs(mean_x-std_x)
        ub = self.get_kwargs(mean_x+std_x)
        for key in mean.keys():
            logging.warning("{} : {} < {} < {}".format(key,lb[key],mean[key],ub[key]))

    
    def plot_results(self):
        f = self.ei.data[1][:,0]
        plt.plot(np.arange(0, self.ei.data[0].shape[0]), np.minimum.accumulate(f))
        xmin = np.argmin(f)
        plt.vlines(xmin,np.min(f), np.max(np.minimum.accumulate(f)),linestyles='--',label='fmin')
        plt.ylabel('fmin')
        plt.xlabel('Number of evaluated points')
        plt.legend()
        ylim = plt.ylim()
        plt.scatter(np.arange(0, self.ei.data[0].shape[0]), f, marker='+',c='black')
        plt.ylim(ylim[0],np.max(np.minimum.accumulate(f)) + 0.5*(np.max(np.minimum.accumulate(f)) - np.min(f)))
        plt.show()

    def run(self, save_path, init_design_size=1, n_iter=24, plot=False, likelihood_uncert=1.,design = None):
        self.save_path = save_path
        self.iter = 0
        
        self.X = [] + self.init_X
        self.Y = [] + self.init_Y
        
        domain = np.sum(self.domain)

#        if self.first_tries is not None:
#            Y = [self.objective(x).flatten() for x in self.first_tries]
#            self.Y = self.Y + Y
#            self.X = self.X + list(self.first_tries)

        if init_design_size > 0:
            if design == 'latin':
                design = LatinHyperCube(init_design_size,domain)
            if design == 'fac':
                design = FactorialDesign(2, domain)
            X = list(design.generate())
            Y = [self.objective(x).flatten() for x in X]
            self.Y = self.Y + Y
            self.X = self.X + X
        
        
        self.X, self.Y = np.stack(self.X,axis=0), np.stack(self.Y,axis=0)
        
        u = np.concatenate([self.X,self.Y],axis=1)
        u, idx = np.unique(u,return_index=True,axis=0)
        self.X = list(self.X[idx,:])
        self.Y = list(self.Y[idx,:])
        
        self.burnin = len(self.X)
        
        logging.warning("Beginnig search")
#         logging.warning("Initial solutions\n{}".format(list(zip(self.X, self.Y))))
        with gp.defer_build():
            kern = gp.kernels.Matern52(domain.size, ARD=True)# + gp.kernels.White(domain.size)
            m = gp.models.GPR(np.stack(self.X,axis=0), np.stack(self.Y,axis=0), kern)
            lik_var = log_normal_solve(likelihood_uncert**2, 0.5*likelihood_uncert**2)
            m.likelihood.variance = likelihood_uncert**2#np.exp(lik_var[0])
            m.likelihood.variance.prior = gp.priors.LogNormal(lik_var[0], lik_var[1]**2)
            m.likelihood.variance.trainable = False
            m.compile()
        self.ei = MomentGeneratingFunctionImprovement(m, self.t)
        opt = optim.StagedOptimizer([optim.MCOptimizer(domain, 5000), optim.SciPyOptimizer(domain)])
        optimizer = BayesianOptimizer(domain, self.ei, optimizer=opt, hyper_draws=1)

        #with optimizer.silent():
        result = optimizer.optimize(self.objective, n_iter=n_iter)
        logging.warning(result)
        if plot:
            self.plot_results()
        result = self.get_kwargs(result.x)
        self.print_top_k(5)
        return result

def test():
    def fx(a,b):
        return (a - b + 1)**2 - b*0.1

    bo = BayesHPOpt(fx,init=None)
    bo.add_binary_param('b',0,1)
    bo.add_continuous_param('a',-2,2)
    bo.run('test_save.hdf5', init_design_size=6, n_iter=10, plot=True, likelihood_uncert=0.01)
