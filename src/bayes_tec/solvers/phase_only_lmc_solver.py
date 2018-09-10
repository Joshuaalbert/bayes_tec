from .phase_only_solver import PhaseOnlySolver
from ..datapack import DataPack
from ..utils.data_utils import calculate_weights, make_data_vec, make_coord_array, define_equal_subsets
from ..utils.stat_utils import log_normal_solve
from ..utils.gpflow_utils import train_with_adam, SendSummary, SaveModel
from ..likelihoods import WrappedPhaseGaussianMulti
from ..kernels import ThinLayer
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
from gpflow.mean_functions import Constant, Zero
from gpflow.kernels import Matern52, Matern32, White
from gpflow.features import InducingPoints
from gpflow import defer_build
import gpflow.multioutput.kernels as mk
import gpflow.multioutput.features as mf
import tensorflow as tf


class LMCPhaseOnlySolver(PhaseOnlySolver):
    def __init__(self, run_dir, datapack):
        super( LMCPhaseOnlySolver,self).__init__(run_dir,datapack)

    def _generate_priors(self, uncert_mean, tec_scale, t_std, d_std):
        """
        Create the global (independent) model priors
        Returns:
        dictionary of priors, each prior is a tuple for LogNormal or Normal priors
        """
        ###
        # Pos params using log-normal priors defined by mode and std

        # Gaussian likelihood log-normal prior
        lik_var = log_normal_solve(uncert_mean, uncert_mean*0.25)
        # TEC mean function prior ensemble mean and variance
        tec_mean_mu, tec_mean_var = 0./tec_scale, (0.005)**2/tec_scale**2
        # TEC kern time lengthscale log-normal prior (seconds)
        tec_kern_time_ls = log_normal_solve(50./t_std, 20./t_std)
        # TEC kern dir lengthscale log-normal prior (degrees)
        tec_kern_dir_ls = log_normal_solve(0.5*np.pi/180./d_std, 0.3*np.pi/180./d_std)
        # TEC kern variance priors
        tec_kern_sigma = 0.005/tec_scale
        tec_kern_var = log_normal_solve(tec_kern_sigma**2,0.1*tec_kern_sigma**2)

        logging.info("likelihood var logGaussian {} median (rad) {}".format(lik_var,np.exp(lik_var[0])))
        logging.info("tec mean Gaussian {} {}".format(tec_mean_mu*tec_scale, tec_mean_var*tec_scale**2))
        logging.info("tec kern var logGaussian {} median (tec) {}".format(tec_kern_var,
                np.sqrt(np.exp(tec_kern_var[0]))*tec_scale))
        logging.info("tec kern time ls logGaussian {} median (sec) {} ".format(tec_kern_time_ls,
                np.exp(tec_kern_time_ls[0])*t_std))
        logging.info("tec kern dir ls logGaussian {} median (rad) {}".format(tec_kern_dir_ls,
                np.exp(tec_kern_dir_ls[0])*d_std))

        priors = {
                "likelihood_var": lik_var,
                "tec_kern_time_ls":tec_kern_time_ls,
                "tec_kern_dir_ls":tec_kern_dir_ls,
                "tec_kern_var":tec_kern_var,
                "tec_mean":(tec_mean_mu, tec_mean_var),
                "tec_scale":tec_scale
                }
        return priors


    def _solve(self,X, screen_X, Y, weights, freqs, jitter=1e-6, learning_rate=1e-3, iterations=1000, 
            minibatch_size=128, dof_ratio=35., tec_scale = 0.001,
            intra_op_threads=0, inter_op_threads=0, overlap = 180., max_block_size = 500, **kwargs):
        """
        Solves and returns posterior mean and variance at the training points.

        Defines the solve steps and runs them.
        It must include:
        1. a model creation
        2. model optimization
        3. posterior inference at input points
        4. posterior inference at screen points
        5. save the model

        Note: Ntabs = 1 because it's phase only

        Params:
        X: array [Nd, Na, Nt, 7] (ra, dec, kz, east, north, up, time)
        screen_X: array [Nd_screen, Na, Nt, 7] as above
        Y: array [Nd, Na, Nf, Nt, Npol*Ntabs]
        weights: array [Nd, Na, Nf, Nt, Npol*Ntabs] 1/variance estimate
        freqs: array [Nf]
        **kwargs: solver specific args passed from run call
        
        Returns:
        posterior_dtec : array [Npol, Nd, Na, Nt]
        posterior_dtec_var : array [Npol, Nd, Na, Nt]
        posterior_screen_dtec : array [Npol, Nd_screen, Na, Nt]
        posterior_screen_dtec_var : array [Npol, Nd_screen, Na, Nt]
        """

        uncert_mean = np.mean(1./(np.sqrt(weights)+1e-6))
        weights /= np.mean(weights)

        Nd, Na, Nf, Nt, Nobs = Y.shape
        Ntabs = 1
        Npol = int(Nobs / Ntabs)
        
        Nd_screen = screen_X.shape[0]
        ## get scaling coeffs

        radec = X[..., 0:2].reshape((-1,2))
        d_mean = radec.mean(0)#(2,)
        radec = radec - d_mean
        d_std = np.sqrt(np.mean(radec**2)) + 1e-6
        radec /= d_std

        X_a = X[..., 3:6].reshape((-1,3))
        a_mean = X_a.mean(0)
        X_a = X_a - a_mean
        a_std = np.sqrt(np.mean(X_a**2)) + 1e-6
        X_a /= a_std
        
        ## apply scaling

        X[..., 0:2] -= d_mean
        X[..., 0:2] /= d_std
        X[..., 3:6] -= a_mean
        X[..., 3:6] /= a_std

        screen_X[..., 0:2] -= d_mean
        screen_X[..., 0:2] /= d_std
        screen_X[..., 3:6] -= a_mean
        screen_X[..., 3:6] /= a_std

        
        X_t = X[0,0,:,6:7]#Nt, 1
        min_overlap = int(np.ceil(overlap/(X_t[1,0] - X_t[0,0])))
        blocks, val_blocks, inv_map = define_equal_subsets(X_t.shape[0], max_block_size, min_overlap, False)
        block_size = blocks[0][1] - blocks[0][0]
        M = int(np.ceil(block_size * Nd * Na / dof_ratio))
        logging.info('Using {} inducing points'.format(M))
        L = len(blocks)
        
        Kuu_size = (L*M**2*8)
        Kuf_size = (L*M*minibatch_size * 8)
        Kff_size = (L*minibatch_size**2 * 8)
        logging.info('Size of Kuu is ({} x {} x {}) [{:.2f} GB]'.format(L, M, M, Kuu_size/(1<<30)))
        logging.info('Size of Kuf is ({} x {} x {}) [{:.2f} GB]'.format(L, M, minibatch_size, Kuf_size/(1<<30)))
        logging.info('Size of Kff is ({} x {} x {}) [{:.2f} GB]'.format(L, minibatch_size, minibatch_size, Kff_size/(1<<30)))
        
        # crop only one blocks required for coordinates
        X = X[:,:,:block_size,:]
        screen_X = screen_X[:,:,:block_size,:]
        
        t_std = (X[0,0,-1,6] - X[0,0,0,6])
        X[...,6] = (X[...,6] - X[:,:,0:1,6])/t_std#(X[:,:,-1:,7] - X[:,:,0:1,7])
        screen_X[...,6] = (screen_X[...,6] - screen_X[:,:,0:1,6]) / t_std#(screen_X[:,:,-1:,7] - screen_X[:,:,0:1,7])
        # Nd*Na*block_size, 7
        X = X.reshape((-1,7))
        # Nd_screen*Na*block_size, 7
        screen_X = screen_X.reshape((-1,7))

        Z = kmeans2(X,M,minit='points')[0] if X.shape[0] < 1e4 \
                else X[np.random.choice(X.shape[0], size=M,replace=False), :]

        q_mu = np.zeros((M,L))
        q_sqrt = np.tile(np.eye(M)[None,:,:],(L,1,1))

        W = np.reshape(np.ones(Nobs)[:,None,None] * np.eye(L)[None,:,:], (Nobs*L,L))

        # data subsets
        # Nd, Na, Nf, Nt, Npol*Ntabs -> Nd, Na, Nf, L, B, Npol*Ntabs
        Y = np.stack([Y[:,:,:,b[0]:b[1],:] for b in blocks],axis=2)
        weights = np.stack([weights[:,:,:,b[0]:b[1],:] for b in blocks],axis=2)
#        Y = Y.reshape((Nd, Na, L, block_size, Nf, Nobs))
#        weights = weights.reshape((Nd, Na, L, block_size, Nf, Nobs))
        # -> Nd*Na*B, L*Npol*Ntabs, Nf
        Y = Y.transpose((0,1,4,3,5,2)).reshape((Nd*Na*block_size,L*Nobs,Nf))
        weights = weights.transpose((0,1,4,3,5,2)).reshape((Nd*Na*block_size,L*Nobs,Nf))

        P = Y.shape[1]#L*Npol*Ntabs       
        
        ###
        # priors from input stats
        priors = self._generate_priors(uncert_mean, tec_scale,t_std, d_std)
        priors['Z_var'] = ((X_t[1,0] - X_t[0,0])*3)**2
        
        graph = tf.Graph()
        sess = self._new_session(graph, intra_op_threads, inter_op_threads)
        summary_id = len(glob.glob(os.path.join(self.summary_dir,"summary_*")))
        summary_folder = os.path.join(self.summary_dir,"summary_{:03d}".format(summary_id))
        os.makedirs(summary_folder,exist_ok=True)
        save_id = len(glob.glob(os.path.join(self.save_dir,"save_*")))
        save_folder = os.path.join(self.save_dir,"save_{:03d}".format(save_id))
        os.makedirs(save_folder,exist_ok=True)

        settings.numerics.jitter = jitter

        with graph.as_default(), sess.as_default(), \
                tf.summary.FileWriter(summary_folder, graph) as writer:
            model = self._make_part_model(X, Y, weights, Z, q_mu, q_sqrt, W,freqs,
                    minibatch_size=minibatch_size, 
                    priors=priors)

            ###
            # batch predict density at X
            def _batch_predict_density(model, X,Y,batch_size):
                _lik= []
                for i in range(0,X.shape[0],batch_size):
                    X_batch = X[i:min(i+batch_size,X.shape[0]),:]
                    Y_batch = Y[i:min(i+batch_size,Y.shape[0]),:,:]
                    lik = model.predict_density(X_batch, Y_batch)
                    _lik.append(lik)
                return np.concatenate(_lik,axis=0)

            
#            pred_lik = np.mean(model.predict_density(X,Y))
#            logging.info("Data var-likelihood before training {}".format(pred_lik))
            
            pred_lik = np.mean(_batch_predict_density(model, X,Y, minibatch_size))
            logging.info("Data var-likelihood before training {}".format(pred_lik))
            
            train_with_adam(model, learning_rate, iterations, [SendSummary(model,writer,write_period=10), SaveModel(save_folder, save_period=1000)])

            pred_lik = np.mean(_batch_predict_density(model, X,Y, minibatch_size))
            logging.info("Data var-likelihood after training {}".format(pred_lik))

            ###
            # predict at X
            def _batch_predict_dtec(model, X,batch_size):
                _ystar, _varstar = [],[]
                for i in range(0,X.shape[0],batch_size):
                    X_batch = X[i:min(i+batch_size,X.shape[0]),:]
                    ystar, varstar = model.predict_dtec(X_batch)
                    _ystar.append(ystar)
                    _varstar.append(varstar)
                return np.concatenate(_ystar,axis=0), np.concatenate(_varstar,axis=0)

            def _unstack_dtec_results(results, out, inv_map):
                Npol,Nd,Na,Nt = out.shape
                idx = 0
                for l, (start, stop) in enumerate(inv_map):
                    width = stop - start
                    out[:,:,:,idx:idx+width] = results[:,:,:,l,start:stop,0]
                return out

            logging.info("Beginning posterior predictions: minibatching {}".format(minibatch_size))
            # Nd*Na*B,L*Npol*Ntabs
            ystar, varstar = _batch_predict_dtec(model, X, minibatch_size)
            logging.info("Finished posterior predictions")

            #Npol, Nd, Na, L, B, Ntabs
            ystar = ystar.reshape((Nd, Na, block_size, L, Npol,Ntabs)).transpose((4,0,1,3,2,5))
            varstar = varstar.reshape((Nd, Na, block_size, L, Npol,Ntabs)).transpose((4,0,1,3,2,5))
            posterior_dtec = np.zeros((Npol, Nd, Na, Nt),dtype=np.float32)
            posterior_dtec_var = np.zeros((Npol, Nd, Na, Nt),dtype=np.float32)
            _unstack_dtec_results(ystar, posterior_dtec, inv_map)
            _unstack_dtec_results(varstar, posterior_dtec_var, inv_map)

            # Nd_screen*Na*B,L*Npol*Ntabs
            logging.info("Beginning posterior screen predictions: minibatching {}".format(minibatch_size))
            ystar, varstar = _batch_predict_dtec(model, screen_X, minibatch_size)
            logging.info("Finished posterior screen predictions")

            #Npol, Nd_screen, Na, L, B, Ntabs
            ystar = ystar.reshape((Nd_screen, Na, block_size, L, Npol,Ntabs)).transpose((4,0,1,3,2,5))
            varstar = varstar.reshape((Nd_screen, Na, block_size, L, Npol,Ntabs)).transpose((4,0,1,3,2,5))
            posterior_screen_dtec = np.zeros((Npol, Nd_screen, Na, Nt),dtype=np.float32)
            posterior_screen_dtec_var = np.zeros((Npol, Nd_screen, Na, Nt),dtype=np.float32)
            _unstack_dtec_results(ystar, posterior_screen_dtec, inv_map)
            _unstack_dtec_results(varstar, posterior_screen_dtec_var, inv_map)

            return posterior_dtec, posterior_dtec_var, posterior_screen_dtec, posterior_screen_dtec_var



    def _make_part_model(self, X, Y, weights, Z, q_mu, q_sqrt, W, freqs, 
            minibatch_size=None, priors=None):
        """
        Create a gpflow model for a selection of data
        X: array (N, Din)
        Y: array (N, P, Nf)
        weights: array like Y the statistical weights of each datapoint
        minibatch_size : int 
        Z: list of array (M, Din)
            The inducing points mean locations.
        q_mu: list of array (M, L)
        q_sqrt: list of array (L, M, M)
        W: array [P,L]
        freqs: array [Nf,] the freqs
        priors : dict of priors for the global model
        Returns:
        model : gpflow.models.Model 
        """
        N, P, Nf = Y.shape
        _, Din = X.shape

        assert priors is not None
        likelihood_var = priors['likelihood_var']
        tec_kern_time_ls = priors['tec_kern_time_ls']
        tec_kern_dir_ls = priors['tec_kern_dir_ls']
        tec_kern_var = priors['tec_kern_var']
        tec_mean = priors['tec_mean']
        Z_var = priors['Z_var']

        P,L = W.shape

        with defer_build():

            
            # Define the likelihood
            likelihood = WrappedPhaseGaussianMulti(tec_scale=priors['tec_scale'],freqs=freqs)
            likelihood.variance = np.exp(likelihood_var[0]) #median as initial
            likelihood.variance.prior = LogNormal(likelihood_var[0],likelihood_var[1]**2)
            likelihood.variance.set_trainable(True)

            def _kern():
                kern_thin_layer = ThinLayer(np.array([0.,0.,0.]), priors['tec_scale'], 
                        active_dims=slice(2,6,1))
                kern_time = Matern32(1,active_dims=slice(6,7,1))
                kern_dir = Matern32(2, active_dims=slice(0,2,1))
                
                ###
                # time kern
                kern_time.lengthscales = np.exp(tec_kern_time_ls[0])
                kern_time.lengthscales.prior = LogNormal(tec_kern_time_ls[0],
                        tec_kern_time_ls[1]**2)
                kern_time.lengthscales.set_trainable(True)

                kern_time.variance = 1.#np.exp(tec_kern_var[0])
                #kern_time.variance.prior = LogNormal(tec_kern_var[0],tec_kern_var[1]**2)
                kern_time.variance.set_trainable(False)#

                ###
                # directional kern
                kern_dir.variance = np.exp(tec_kern_var[0])
                kern_dir.variance.prior = LogNormal(tec_kern_var[0],tec_kern_var[1]**2)
                kern_dir.variance.set_trainable(True)

                kern_dir.lengthscales = np.exp(tec_kern_dir_ls[0])
                kern_dir.lengthscales.prior = LogNormal(tec_kern_dir_ls[0],
                        tec_kern_dir_ls[1]**2)
                kern_dir.lengthscales.set_trainable(True)

                kern = kern_dir*kern_time#(kern_thin_layer + kern_dir)*kern_time
                return kern

            kern = mk.SeparateMixedMok([_kern() for _ in range(L)], W)

            feature_list = []
            for _ in range(L):
                feat = InducingPoints(Z)
                #feat.Z.prior = Gaussian(Z,Z_var)
                feature_list.append(feat)
            feature = mf.MixedKernelSeparateMof(feature_list)


            mean = Zero()


            model = HomoscedasticPhaseOnlySVGP(weights, X, Y, kern, likelihood, 
                        feat = feature,
                        mean_function=mean, 
                        minibatch_size=minibatch_size,
                        num_latent = P, 
                        num_data = N,
                        whiten=False, q_mu = q_mu, q_sqrt=q_sqrt)
            model.compile()
        return model


    
