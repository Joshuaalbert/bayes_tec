from .solver import Solver
from ..datapack import DataPack
from ..utils.data_utils import calculate_weights, make_data_vec, make_coord_array, define_subsets
from ..utils.stat_utils import log_normal_solve
from ..utils.gpflow_utils import train_with_adam, SendSummary, SaveModel
from ..likelihoods import WrappedPhaseGaussian
from ..frames import ENU
from ..models.homoscedastic_phaseonly_svgp import HomoscedasticPhaseOnlySVGP
from ..logging import logging
import astropy.units as au
from scipy.cluster.vq import kmeans2
import numpy as np
import os
import glob
from gpflow import settings
from gpflow.priors import LogNormal, Gaussian
from gpflow.mean_functions import Constant
from gpflow.kernels import Matern52, White
from gpflow.features import InducingPoints
#from gpflow.training.monitor import (create_global_step, PrintTimingsTask, PeriodicIterationCondition, 
#        CheckpointTask, LogdirWriter, ModelToTensorBoardTask, Monitor)
from gpflow import defer_build
import gpflow.multioutput.kernels as mk
import gpflow.multioutput.features as mf
import tensorflow as tf


class PhaseOnlySolver(Solver):
    def __init__(self, run_dir, datapack):
        super( PhaseOnlySolver,self).__init__(run_dir,datapack)
#        if not isinstance(tabs,(tuple,list)):
#            tabs = [tabs]
#        self.tabs = tabs # ['phase', 'amplitude', ...] etc.
        self.tabs = ['phase']

    def _make_part_model(self, X, Y, Z, minibatch_size=None, eval_freq=140e6, 
            tec_scale=0.01, num_latent=1, priors=None, shared_kernels=True, shared_features=True):
        """
        Create a gpflow model for a selection of data
        X: array (N, Din)
        Y: array (N, 2*num_latent + 1)
            See ..utils.data_utils.make_datavec
        minibatch_size : int 
        Z: array (M, Din)
            The inducing points if desired to set.
        eval_freq: float the freq in Hz where evaluation occurs.
        tec_scale : float default 0.01
        num_latent: int (see Dout)
        priors : dict of priors for the global model
        Returns:
        model : gpflow.models.Model 
        """

        raise NotImplementedError("must subclass")

    
    def _solve(self,X_d, X_t, freqs, X_d_screen, Y, weights, **kwargs):
        """
        Defines the solve steps and runs them.
        It must include:
        1. a model creation
        2. model optimization
        3. posterior inference at input points
        4. posterior inference at screen points
        5. save the model

        Params:
        X_d : array [Nd, 2]
        X_t : array [Nt, 1]
        freqs : array [Nf]
        X_d_screen : array [Nd_scren, 2]
        Y : array  [Npols, Nd, Na, Nf, Nt, Ntabs]
        weights : weights = 1./var of datapoints (estimated from data)
        **kwargs: solver specific args passed from run call
        
        Returns:
        posterior_dtec : array [Npol, Nd, Na, Nt]
        posterior_dtec_var : array [Npol, Nd, Na, Nt]
        posterior_screen_dtec : array [Npol, Nd_screen, Na, Nt]
        posterior_screen_dtec_var : array [Npol, Nd_screen, Na, Nt]
        """
        raise NotImplementedError("must subclass")
        
    def run(self, ant_sel=None, time_sel=None, dir_sel=None, freq_sel=None, pol_sel=None,reweight_obs=True, screen_res=30, **kwargs):

        with self.datapack:
            self.datapack.select(ant=ant_sel,time=time_sel, dir=dir_sel, freq=freq_sel, pol=pol_sel)
            Y = []
            weights = []
            axes = None
            # general for more tabs if desired (though model must change)
            for tab in self.tabs:
                vals, axes = self.datapack.__getattr__(tab)
                # each Npols, Nd, Na, Nf, Nt
                Y.append(vals)
                if reweight_obs:
                    logging.info("Re-calculating weights...")
                    smooth_len = int(2 * 180. / (axes['time'][1] - axes['time'][0]))
                    weights_ = calculate_weights(vals,indep_axis = -1, num_threads = None, N=smooth_len, phase_wrap=True, min_uncert=1e-3)
                    self.datapack.__setattr__("weights_{}".format(tab), weights_)
                    weights.append(weights_)
                else:
                    weights_, _ = self.datapack.__getattr__("weights_{}".format(tab))
                    weights.append(weights_)

            antenna_labels, antennas = self.datapack.get_antennas(axes['ant'])
            patch_names, directions = self.datapack.get_sources(axes['dir'])
            timestamps, times = self.datapack.get_times(axes['time'])
            freq_labels, freqs = self.datapack.get_freqs(axes['freq'])
            pol_labels, pols = self.datapack.get_pols(axes['pol'])
            
            # Npol, Nd, Na, Nf, Nt, Ntabs
            Y = np.stack(Y,axis=-1)
            weights = np.stack(weights,axis=-1)
            
            Npol, Nd, Na, Nf, Nt, Ntabs = Y.shape

            
            ###
            # input coords

            X_d = np.array([directions.ra.rad,directions.dec.rad]).T
            X_t = times.mjd[:,None]*86400.#mjs

            d_min, d_max = np.min(X_d), np.max(X_d)
            X_d_screen = np.array([m.flatten() \
                    for m in np.meshgrid(*([np.linspace(d_min,d_max, screen_res)]*2),indexing='ij')]).T
            Nd_ = screen_res**2
            self.datapack.switch_solset("screen_sol", 
                    array_file = DataPack.lofar_array, 
                    directions = X_d_screen)
            # store variance in tec/weights
            self.datapack.add_freq_indep_tab('tec', times.mjd*86400., pols = pol_labels)
            # output solset
            self.datapack.switch_solset("posterior_sol", 
                    array_file=DataPack.lofar_array, 
                    directions = X_d)
            # store variance in tec/weights
            self.datapack.add_freq_indep_tab('tec', times.mjd*86400., pols = pol_labels)

        ###
        # Call subclass _solve
        # TODO separate into optimize/inference steps

        (posterior_dtec, posterior_dtec_var, 
        posterior_screen_dtec, posterior_screen_dtec_var) = \
                        self._solve(X_d, X_t, freqs, X_d_screen, Y, weights, **kwargs)

        ###
        # Store results
        with self.datapack:
            self.datapack.switch_solset("posterior_sol")
            self.datapack.select(ant=ant_sel,time=time_sel, dir=dir_sel, freq=freq_sel, pol=pol_sel)
            self.datapack.tec = posterior_dtec
            self.datapack.weights_tec = 1./(posterior_dtec_var + 1e-6)

            self.datapack.switch_solset("screen_sol")
            self.datapack.select(ant=ant_sel,time=time_sel, dir=dir_sel, freq=freq_sel, pol=pol_sel)
            self.datapack.tec = posterior_screen_dtec
            self.datapack.weights_tec = 1./(posterior_screen_dtec_var + 1e-6)
