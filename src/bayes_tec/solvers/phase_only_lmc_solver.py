from .phase_only_solver import PhaseOnlySolver
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
                "tec_mean":(tec_mean_mu, tec_mean_var)
                }
        return priors


    def _solve(self,X_d, X_t, freqs, X_d_screen, Y, weights, jitter=1e-6, learning_rate=1e-3, iterations=10000, minibatch_size=128, 
            eval_freq=140e6, dof_ratio=35., tec_scale = 0.01,
            intra_op_threads=0, inter_op_threads=0, overlap = 180., max_block_size = 500, **kwargs):
        """
        Breaks up the input data into time chunks and solves with a 
        correlated, multi-output, heterotopic Gaussian process using LMC.
        The multi-output is formed of antennas, polarization, and time chunks.
        Each output is assumed to have a kernel:
        K_global[shared by all] + K_a[shared by antennas].
        The mean is Zero by assumption (see _generate_priors).


        Defines the solve steps and runs them.
        It must include:
        1. a model creation
        2. model optimization
        3. posterior inference at input points
        4. posterior inference at screen points
        5. save the model

        Note: Ntabs = 1 because it's phase only

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

#        overlap = kwargs.get('overlap',180.)
#        max_block_size = kwargs.get('max_block_size', 500)
#        time_skip = kwargs.get('time_skip', 2)

        uncert_mean = np.mean(1./(np.sqrt(weights)+1e-6))
        weights /= np.mean(weights)

        Npol, Nd, Na, Nf, Nt, Ntabs = Y.shape
        num_latent = Npol*Na*Ntabs
        P = num_latent
        Nd_screen = X_d_screen.shape[0]

        # Nd, Nt, Nf, Npol*Na*Ntabs
        Y = Y.transpose((1,4,3,0,2,5)).reshape((Nd, Nt, Nf, Npol*Na*Ntabs))
        weights = weights.transpose((1,4,3,0,2,5)).reshape((Nd, Nt, Nf, Npol*Na*Ntabs))

        d_std = X_d.std(0).mean() + 1e-6
        t_std = X_t.std() + 1e-6

        X_t = (X_t - X_t.mean()) / t_std
        d_mean = X_d.mean(0)
        X_d = (X_d - d_mean) / d_std
        X_d_screen = (X_d_screen - d_mean) / d_std
        freq_bar = np.mean(1./freqs)


        # data subsets
        Z_subs = []
        q_mu_subs = []
        q_sqrt_diag_subs = []
        W_subs = []
        l_per_P = 1
        edges,blocks = define_subsets(X_t, overlap / t_std, max_block_size)
        for subset_idx, block in enumerate(blocks):
            start = edges[block[0]]
            stop = edges[block[1]]

            # Nd*Nt_sub, 3
            X_sub = make_coord_array(X_d, X_t[start:stop,:], freqs[:1,None])[:,:-1]
            N = X_sub.shape[0]
            M = int(np.ceil((stop - start)*X_d.shape[0] / dof_ratio))
            logging.info('Using {} inducing points'.format(M))

            idx = np.random.choice(N,size=M,replace=False)
            Z_sub = X_sub[idx,:]


            Y_sub = Y[:,start:stop,:,:]
            # Nd , Nt_sub, Nf,  Npol*Na*Ntabs
            z = np.exp(1j*Y_sub)
            # Nd , Nt_sub,  Npol*Na*Ntabs
            z_bar = np.mean(z,axis=2)
            z_bar = z_bar.reshape((-1,P))[idx,:]
            # Nd * Nt_sub, Npol*Na*Ntabs
            f_mu_sub = np.angle(z_bar) * freq_bar / -8.440e9 / tec_scale
            Re2 = Nf/(Nf - 1) * (z_bar*z_bar.conj() - 1./Nf)
            # Nd * Nt_sub, Npol*Na*Ntabs
            f_sqrt_diag_sub= np.sqrt(-np.log(Re2) * freq_bar**2 / (-8.440e9)**2 / tec_scale**2).real
            f_sqrt_diag_sub = np.maximum(f_sqrt_diag_sub,1e-2)

            # Nd * Nt_sub, Npol*Na*Ntabs
            # f[:M, :P]

            _,_,Wh = np.linalg.svd(f_mu_sub)
            #P, l_per_P
            W=Wh[:l_per_P,:].T
            # M,P @ P,l_per_P -> M,l_per_P
            q_mu_sub = f_mu_sub.dot(W)
            # M, P @ P, l_per_P -> M,l_per_P
            q_sqrt_diag_sub = f_sqrt_diag_sub.dot(W**2)

            W_subs.append(W)
            q_mu_subs.append(f_mu_sub)
            q_sqrt_diag_subs.append(f_sqrt_diag_sub)
            [Z_subs.append(Z_sub) for _ in range(l_per_P)]
        # P,L
        W = np.concatenate(W_subs,axis=1)
        # M, L
        q_mu = np.concatenate(q_mu_subs,axis=1)
        # M, L
        q_sqrt_diag = np.concatenate(q_sqrt_diag_subs,axis=1)
        # L, M, M
        q_sqrt = np.stack([np.diag(d) for d in q_sqrt_diag.T],axis=0)

        print(q_mu.shape, q_sqrt.shape, len(Z_subs), W.shape)
        
        
        # Nd, Nt, Nf, 2 * Npol*Na*Ntabs + 1
        Y = make_data_vec(Y, freqs,weights=weights)
        # Nd * Nt * Nf, 2 * Npol*Na*Ntabs + 1
        Y = Y.reshape((-1, 2*num_latent + 1))
        
        X = make_coord_array(X_d, X_t, freqs[:,None])[:,:-1]
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
            model = self._make_part_model(X, Y, Z_subs, q_mu, q_sqrt, W,
                    minibatch_size=minibatch_size, 
                    eval_freq=eval_freq, tec_scale=tec_scale, num_latent=num_latent, 
                    priors=priors)
            
#            pred_lik = np.mean(model.predict_density(X,Y))
#            logging.info("Data var-likelihood before training {}".format(pred_lik))
            
            train_with_adam(model, learning_rate, iterations, [SendSummary(model,writer,write_period=10), SaveModel(save_folder, save_period=1000)])
            pred_lik = np.mean(model.predict_density(X,Y))
            logging.info("Data var-likelihood after training {}".format(pred_lik))

            ###
            # predict at X
            Xstar = make_coord_array(X_d, X_t, freqs[:1,None])[:,:-1]
            # Nd*Nt,num_latent=Npol*Na*Ntabs
            ystar, varstar = model.predict_dtec(Xstar)

            #Nd, Nt, Npol, Na, Ntabs
            ystar = ystar.reshape([Nd,Nt,Npol,Na,Ntabs])
            #Npols, Nd, Na, Nt, Ntabs
            ystar = ystar.transpose([2,0,3,1,4])
            #Nd, Nt, Npol, Na, Ntabs
            varstar = varstar.reshape([Nd,Nt,Npol,Na,Ntabs])
            #Npols, Nd, Na, Nt, Ntabs
            varstar = varstar.transpose([2,0,3,1,4])
            posterior_dtec = ystar[...,0]
            posterior_dtec_var = varstar[...,0]

            ###
            # predict at X (screen)
            Xstar = make_coord_array(X_d_screen, X_t, freqs[:1,None])[:,:-1]
            # Nd_screen*Nt,num_latent=Npol*Na*Ntabs
            ystar, varstar = model.predict_dtec(Xstar)

            #Nd_screen, Nt, Npol, Na, Ntabs
            ystar = ystar.reshape([Nd_screen,Nt,Npol,Na,Ntabs])
            #Npols, Nd_screen, Na, Nt, Ntabs
            ystar = ystar.transpose([2,0,3,1,4])
            #Nd_screen, Nt, Npol, Na, Ntabs
            varstar = varstar.reshape([Nd_screen,Nt,Npol,Na,Ntabs])
            #Npols, Nd_screen, Na, Nt, Ntabs
            varstar = varstar.transpose([2,0,3,1,4])
            posterior_screen_dtec = ystar[...,0]
            posterior_screen_dtec_var = varstar[...,0]
        return posterior_dtec, posterior_dtec_var, posterior_screen_dtec, posterior_screen_dtec_var




    def _make_part_model(self, X, Y, Z_subs, q_mu, q_sqrt,W, minibatch_size=None, eval_freq=140e6, 
            tec_scale=0.01, num_latent=1, priors=None):
        """
        Create a gpflow model for a selection of data
        X: array (N, Din)
        Y: array (N, 2*P + 1)
            See ..utils.data_utils.make_datavec
        minibatch_size : int 
        Z_subs: list of array (M, Din)
            The inducing points mean locations.
        q_mu: list of array (M, L)
        q_sqrt: list of array (L, M, M)
        W: array [P,L]
        eval_freq: float the freq in Hz where evaluation occurs.
        tec_scale : float default 0.01
        num_latent: int equivalent to P
        priors : dict of priors for the global model
        Returns:
        model : gpflow.models.Model 
        """
        N, Dout = Y.shape
        _, Din = X.shape

        assert priors is not None
        likelihood_var = priors['likelihood_var']
        tec_kern_time_ls = priors['tec_kern_time_ls']
        tec_kern_dir_ls = priors['tec_kern_dir_ls']
        tec_kern_var = priors['tec_kern_var']
        tec_mean = priors['tec_mean']
        Z_var = priors['Z_var']

        P,L = W.shape
        print( X.shape, Y.shape, Z_subs[0].shape, q_mu.shape, q_sqrt.shape,W.shape)

#        P = num_latent
#        L = len(Z_subs)

        with defer_build():

            
            # Define the likelihood
            likelihood = WrappedPhaseGaussian(tec_scale=tec_scale,freq=eval_freq)
            likelihood.variance = np.exp(likelihood_var[0]) #median as initial
            likelihood.variance.prior = LogNormal(likelihood_var[0],likelihood_var[1]**2)
            likelihood.variance.set_trainable(True)

            def _kern():
                kern_time = Matern32(1,active_dims=[2])
                kern_time.lengthscales = np.exp(tec_kern_time_ls[0])
                kern_time.lengthscales.set_trainable(True)
                kern_time.lengthscales.prior = LogNormal(tec_kern_time_ls[0],tec_kern_time_ls[1]**2)#gamma_prior(70./t_std, 50./t_std)
                kern_time.variance = np.exp(tec_kern_var[0])
                kern_time.variance.set_trainable(True)
                kern_time.variance.prior = LogNormal(tec_kern_var[0],tec_kern_var[1]**2)#gamma_prior(0.001, 0.005)

                kern_space = Matern52(2,active_dims=[0,1],variance=1.)
                kern_space.variance.set_trainable(False)
                kern_space.lengthscales = np.exp(tec_kern_dir_ls[0])
                kern_space.lengthscales.set_trainable(True)
                kern_space.lengthscales.prior = LogNormal(tec_kern_dir_ls[0],tec_kern_dir_ls[1]**2)

#                white = White(3)
#                white.variance = 0.0005**2/tec_scale**2
#                white.variance.set_trainable(False) X, Y, Z_subs, q_mu, q_sqrt,W
                kern = kern_space*kern_time
                return kern

            kern = mk.SeparateMixedMok([_kern() for _ in range(L)], W)
#           kern = mk.SeparateIndependentMok([_kern() for _ in range(L)])
#
#            kern = mk.SeparateIndependentMok([
#                mk.SeparateMixedMok([_kern() for _ in range(P)], W) \
#                    + mk.SeparateIndependentMok([_kern() for _ in range(P)]) \
#                    + mk.SharedIndependentMok(_kern(), P)
#                    ])
            feature_list = []
            for Z in Z_subs:
                feat = InducingPoints(Z)
                feat.Z.prior = Gaussian(Z,Z_var)
                feature_list.append(feat)
            feature = mf.MixedKernelSharedMof(feature_list[0])
#            feature = mf.SeparateIndependentMof(feature_list)


            mean = Zero()
#            mean = Constant(0.)#tec_mean_mu)
#            mean.c.set_trainable(False)
#            mean.c.prior = Gaussian(tec_mean[0],tec_mean[1])

            model = HomoscedasticPhaseOnlySVGP(P, X, Y, kern, likelihood, 
                        feat = feature,
                        mean_function=mean, 
                        minibatch_size=minibatch_size,
                        num_latent = L, 
                        num_data = N,
                        whiten=False, q_mu = q_mu, q_sqrt=q_sqrt)
            model.compile()
        return model


    
