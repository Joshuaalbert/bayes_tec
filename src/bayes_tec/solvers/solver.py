from ..datapack import DataPack
from ..utils.data_utils import phase_weights, make_data_vec, make_coord_array, define_subsets
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

class Solver(object):
    def __init__(self,run_dir, datapack):
        run_dir = os.path.abspath(run_dir)
        self.run_id = len(glob.glob(os.path.join(run_dir,"run_*")))
        self.run_dir = os.path.join(run_dir,"run_{:03d}".format(self.run_id))
        self.summary_dir = os.path.join(self.run_dir,"summaries")
        self.save_dir = os.path.join(self.run_dir, "checkpoints")
        os.makedirs(self.run_dir,exist_ok=True)
        os.makedirs(self.summary_dir,exist_ok=True)
        os.makedirs(self.save_dir,exist_ok=True)
        if isinstance(datapack,str):
            datapack = DataPack(datapack)
        self.datapack = datapack
    def run(self,*args, **kwargs):
        """Run the solver"""
        raise NotImplementedError("Must subclass")

    def _new_session(self, graph, intra_op_threads, inter_op_threads):
        os.environ["KMP_BLOCKTIME"] = "1"
        os.environ["KMP_SETTINGS"] = "1"
        os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"
        if intra_op_threads > 0:
            os.environ["OMP_NUM_THREADS"] = str(intra_op_threads)
        config = tf.ConfigProto()
        config.intra_op_parallelism_threads = intra_op_threads
        config.inter_op_parallelism_threads = inter_op_threads
        sess = tf.Session(graph=graph,config=config)
        return sess


class OverlapPhaseOnlySolver(Solver):
    def __init__(self, overlap, run_dir, datapack):
        super(OverlapPhaseOnlySolver,self).__init__(run_dir,datapack)
        self.overlap = float(overlap)# minimum overlap in seconds
#        if not isinstance(tabs,(tuple,list)):
#            tabs = [tabs]
#        self.tabs = tabs # ['phase', 'amplitude', ...] etc.
        self.tabs = ['phase']

    def _make_part_model(self, X, Y, Z, minibatch_size=None, eval_freq=140e6, 
            tec_scale=0.01, num_latent=1, priors=None, shared_kernels=True, shared_features=True):
        """
        Create a gpflow model for a selection of data
        X: array (N, Din)
        Y: array (N, 2*Dout + 1)
            See ..utils.data_utils.make_datavec
        minibatch_size : int 
        Z: array (M, Din)
            The inducing points if desired to set.
        eval_freq: float the freq in Hz where evaluation occurs.
        """
        N, Dout = Y.shape
        _, Din = X.shape

        assert priors is not None
        likelihood_var = priors['likelihood_var']
        tec_kern_time_ls = priors['tec_kern_time_ls']
        tec_kern_dir_ls = priors['tec_kern_dir_ls']
        tec_kern_var = priors['tec_kern_var']
        tec_mean = priors['tec_mean']

        with defer_build():

            
            # Define the likelihood
            likelihood = WrappedPhaseGaussian(tec_scale=tec_scale,freq=eval_freq)
            likelihood.variance = np.exp(likelihood_var[0]) #median as initial
            likelihood.variance.prior = LogNormal(likelihood_var[0],likelihood_var[1]**2)
            likelihood.variance.set_trainable(True)
            def _kern():
                kern_time = Matern52(1,active_dims=[0])
                kern_time.lengthscales = np.exp(tec_kern_time_ls[0])
                kern_time.lengthscales.set_trainable(True)
                kern_time.lengthscales.prior = LogNormal(tec_kern_time_ls[0],tec_kern_time_ls[1]**2)#gamma_prior(70./t_std, 50./t_std)
                kern_time.variance = np.exp(tec_kern_var[0])
                kern_time.variance.set_trainable(True)
                kern_time.variance.prior = LogNormal(tec_kern_var[0],tec_kern_var[1]**2)#gamma_prior(0.001, 0.005)

                kern_space = Matern52(2,active_dims=[1,2],variance=1.)
                kern_space.variance.set_trainable(False)
                kern_space.lengthscales = np.exp(tec_kern_dir_ls[0])
                kern_space.lengthscales.set_trainable(True)
                kern_space.lengthscales.prior = LogNormal(tec_kern_dir_ls[0],tec_kern_dir_ls[1]**2)

                white = White(3)
                white.variance = 0.0005**2/tec_scale**2
                white.variance.set_trainable(False)
                kern = kern_time*kern_space + white
                return kern

            if not shared_kernels:
                kern_list = [_kern() for _ in range(num_latent)]
                kern = mk.SeparateIndependentMok(kern_list)
            else:
                kern = mk.SharedIndependentMok(_kern(),num_latent)

            if not shared_features:
                feature_list = [InducingPoints(Z) for _ in range(num_latent)]
                feature = mf.SeparateIndependentMof(feature_list)
            else:
                feature = mf.SharedIndependentMof(InducingPoints(Z))



            mean = Constant(0.)#tec_mean_mu)
            mean.c.set_trainable(False)
            mean.c.prior = Gaussian(tec_mean[0],tec_mean[1])

            model = HomoscedasticPhaseOnlySVGP(X, Y, kern, likelihood, 
                        feat = feature,
                        mean_function=mean, 
                        minibatch_size=minibatch_size,
                        num_latent = num_latent, 
                        num_data=N,
                        whiten=False)

            model.compile()
        return model


    def run(self, ant_sel=None, time_sel=None, dir_sel=None, freq_sel=None, pol_sel=None,reweight_obs=True, 
            screen_res=30, jitter=1e-6, learning_rate=1e-3, iterations=10000, minibatch_size=128, 
            eval_freq=140e6, dof_ratio=35., max_block_size=800, tec_scale = 0.01, time_skip=3, 
            intra_op_threads=0, inter_op_threads=0, shared_kernels=True, shared_features=True, **kwargs):

        settings.numerics.jitter = jitter

        with self.datapack:
            self.datapack.select(ant=ant_sel,time=time_sel, dir=dir_sel, freq=freq_sel, pol=pol_sel)
            Y = []
            weights = []
            uncert_mean = []
            axes = None
            # general for more tabs if desired (though model must change)
            for tab in self.tabs:
                vals, axes = self.datapack.__getattr__(tab)
                # each Npols, Nd, Na, Nf, Nt
                Y.append(vals)
                if reweight_obs:

                    weights_, uncert_mean_ = phase_weights(vals,indep_axis = -2, num_threads = None,N=200,phase_wrap=True, min_uncert=1e-3)
                    self.datapack.__setattr__("weights_{}".format(tab), weights_)
                    weights.append(weights_)
                    uncert_mean.append(uncert_mean_)
                else:
                    weights_, _ = self.datapack.__getattr__("weights_{}".format(tab))
                    weights.append(weights_)
                    uncert_mean.append(np.nanmean(np.sqrt(1./weights)))
            uncert_mean = np.mean(uncert_mean)

            antenna_labels, antennas = self.datapack.get_antennas(axes['ant'])
            patch_names, directions = self.datapack.get_sources(axes['dir'])
            timestamps, times = self.datapack.get_times(axes['time'])
            freq_labels, freqs = self.datapack.get_freqs(axes['freq'])
            pol_labels, pols = self.datapack.get_pols(axes['pol'])
            
            # Npols, Nd, Na, Nf, Nt, Ntabs
            Y = np.stack(Y,axis=-1)
            weights = np.stack(weights,axis=-1)
            Npol, Nd, Na, Nf, Nt, Ntabs = Y.shape
            # Nd, Npol*Na*Ntabs, Nf, Nt
            Y = Y.transpose((1,0,2,5, 3,4)).reshape((Nd, Npol*Na*Ntabs, Nf, Nt))
            weights = weights.transpose((1,0,2,5, 3,4)).reshape((Nd, Npol*Na*Ntabs, Nf, Nt))
            #Nd, Nt, Nf, Npol*Na*Ntabs
            Y = Y.transpose((0,3,2,1))
            weights = weights.transpose((0,3,2,1))
            num_latents = Npol*Na*Ntabs
            # Nd, Nt, Nf,2 * Npol*Na*Ntabs + 1
            data_vec = make_data_vec(Y, freqs,weights=weights)
            num_latent = Npol*Na*Ntabs
            
            ###
            # input coords

            X_d = np.array([directions.ra.deg,directions.dec.deg]).T
            X_t = times.mjd[:,None]*86400.#mjs
            enu = ENU(obstime=times[0],location=self.datapack.array_center)
            ant_enu = antennas.transform_to(enu)
            X_a = np.array([ant_enu.east.to(au.km).value, ant_enu.north.to(au.km).value]).T

            d_std = X_d.std(0).mean() + 1e-6
            t_std = X_t.std() + 1e-6
            a_std = X_a.std(0).mean() + 1e-6

            X_a = (X_a - X_a.mean(0)) / a_std
            X_t = (X_t - X_t.mean()) / t_std
            d_mean = X_d.mean(0)
            X_d = (X_d - d_mean) / d_std

            d_min, d_max = np.min(X_d), np.max(X_d)
            X_d_ = np.array([m.flatten() \
                    for m in np.meshgrid(*([np.linspace(d_min,d_max, screen_res)]*2),indexing='ij')]).T
            Nd_ = screen_res**2
            directions_ = X_d_*d_std + d_mean
            self.datapack.switch_solset("screen_sol", 
                    array_file = DataPack.lofar_array, 
                    directions = directions_ * np.pi/180.)
            # store variance in tec/weights
            self.datapack.add_freq_indep_tab('tec', times.mjd*86400., pols = pol_labels)
            screen_dtec, _ = self.datapack.tec
            screen_dtec_var = np.zeros_like(screen_dtec)
            # output solset
            self.datapack.switch_solset("posterior_sol", 
                    array_file=DataPack.lofar_array, 
                    directions = np.array([directions.ra.rad, directions.dec.rad]).T)
            # store variance in tec/weights
            self.datapack.add_freq_indep_tab('tec', times.mjd*86400., pols = pol_labels)
            posterior_dtec, _ = self.datapack.tec
            posterior_dtec_var = np.zeros_like(posterior_dtec)

        # data subsets
        edges,blocks = define_subsets(X_t, self.overlap / t_std, max_block_size)
        for subset_idx, block in enumerate(blocks):
            start = edges[block[0]]
            stop = edges[block[1]]

            # N, 3
            X = make_coord_array(X_d, X_t[start:stop,:], freqs[:,None])[:,:-1]
            # Nd, Nt, Nf,2 * Npol*Na*Ntabs + 1
            Y = data_vec[:,start:stop,:,:]
            Y = Y.reshape((-1, Y.shape[-1]))

            N = Y.shape[0]

            ###
            # Pos params using log-normal priors defined by mode and std

            # Gaussian likelihood log-normal prior
            lik_var = log_normal_solve(uncert_mean, uncert_mean*0.25)
            # TEC mean function prior ensemble mean and variance
            tec_mean_mu, tec_mean_var = 0./tec_scale, (0.005)**2/tec_scale**2
            # TEC kern time lengthscale log-normal prior (seconds)
            tec_kern_time_ls = log_normal_solve(50./t_std, 20./t_std)
            # TEC kern dir lengthscale log-normal prior (degrees)
            tec_kern_dir_ls = log_normal_solve(0.5/d_std, 0.3/d_std)
            # TEC kern variance priors
            tec_kern_sigma = 0.005/tec_scale
            tec_kern_var = log_normal_solve(tec_kern_sigma**2,0.1*tec_kern_sigma**2)

            priors = {
                    "likelihood_var": lik_var,
                    "tec_kern_time_ls":tec_kern_time_ls,
                    "tec_kern_dir_ls":tec_kern_dir_ls,
                    "tec_kern_var":tec_kern_var,
                    "tec_mean":(tec_mean_mu, tec_mean_var)
                    }

            logging.info("likelihood var logGaussian {} median (rad) {}".format(lik_var,np.exp(lik_var[0])))
            logging.info("tec mean Gaussian {} {}".format(tec_mean_mu*tec_scale, tec_mean_var*tec_scale**2))
            logging.info("tec kern var logGaussian {} median (tec) {}".format(tec_kern_var,
                    np.sqrt(np.exp(tec_kern_var[0]))*tec_scale))
            logging.info("tec kern time ls logGaussian {} median (sec) {} ".format(tec_kern_time_ls,
                    np.exp(tec_kern_time_ls[0])*t_std))
            logging.info("tec kern dir ls logGaussian {} median (deg) {}".format(tec_kern_dir_ls,
                    np.exp(tec_kern_dir_ls[0])*d_std))

            M = int(np.ceil((stop - start)*X_d.shape[0] / dof_ratio))
            Z = kmeans2(X, M, minit='points')[0] if N < 10000 \
                    else X[np.random.choice(N,size=M,replace=False),:]
            if M is None:
                Z = make_coord_array(X_t[::time_skip,:],X_d[::1,:])
            
            graph = tf.Graph()
            sess = self._new_session(graph, intra_op_threads, inter_op_threads)
            summary_id = len(glob.glob(os.path.join(self.summary_dir,"summary_*")))
            summary_folder = os.path.join(self.summary_dir,"summary_{:03d}".format(summary_id))
            os.makedirs(summary_folder,exist_ok=True)
            save_id = len(glob.glob(os.path.join(self.save_dir,"save_*")))
            save_folder = os.path.join(self.save_dir,"save_{:03d}".format(save_id))
            os.makedirs(save_folder,exist_ok=True)
            
            with graph.as_default(), sess.as_default(), \
                    tf.summary.FileWriter(summary_folder, graph) as writer:
                model = self._make_part_model(X, Y, Z, minibatch_size=minibatch_size, 
                        eval_freq=eval_freq, tec_scale=tec_scale, num_latent=num_latent, priors=priors, shared_kernels=shared_kernels, shared_features=shared_features)
                
                pred_lik = np.mean(model.predict_density(X,Y))
                logging.info("Data var-likelihood before training {}".format(pred_lik))
                
                train_with_adam(model, learning_rate, iterations, [SendSummary(model,writer,write_period=10), SaveModel(save_folder, save_period=1000)])
                pred_lik = np.mean(model.predict_density(X,Y))
                logging.info("Data var-likelihood after training {}".format(pred_lik))
                ###
                # predict at X
                if subset_idx == 0:
                    sub_start = start = edges[block[0]]
                    sub_stop = edges[block[1]-1]
                elif subset_idx == len(blocks)-1:
                    sub_start = start = edges[block[0]+1]
                    sub_stop = edges[block[1]]
                else:
                    sub_start = start = edges[block[0]+1]
                    sub_stop = edges[block[1]-1]
                Nt_sub = sub_stop - sub_start
                Xstar = make_coord_array(X_d, X_t[sub_start:sub_stop,:], freqs[:1,None])[:,:-1]
                # Nd*Nt_sub,3
                ystar, varstar = model.predict_dtec(Xstar)

                #Nd, Nt, Npol, Na, Ntabs
                ystar = ystar.reshape([Nd,Nt_sub,Npol,Na,Ntabs])
                #Npols, Nd, Na, Nt, Ntabs
                ystar = ystar.transpose([2,0,3,1,4])
                #Nd, Nt, Npol, Na, Ntabs
                varstar = varstar.reshape([Nd,Nt_sub,Npol,Na,Ntabs])
                #Npols, Nd, Na, Nt, Ntabs
                varstar = varstar.transpose([2,0,3,1,4])
                posterior_dtec[:,:,:,sub_start:sub_stop] = ystar[...,0]
                posterior_dtec_var[:,:,:,sub_start:sub_stop] = varstar[...,0]

        with self.datapack:
            self.datapack.switch_solset("posterior_sol")
#            self.datapack.select(ant=ant_sel,time=slice(sub_start,sub_stop,1), dir=dir_sel, freq=freq_sel, pol=pol_sel)
            self.datapack.select_all()
            self.datapack.tec = posterior_dtec
            self.datapack.weights_tec = 1./posterior_dtec_var


