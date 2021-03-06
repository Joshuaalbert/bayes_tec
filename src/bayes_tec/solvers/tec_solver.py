from .solver import Solver
from ..datapack import DataPack
from ..logging import logging
import os
import glob
import tensorflow as tf
from gpflow import settings
import gpflow as gp

from ..utils.data_utils import calculate_weights, make_coord_array
import astropy.units as au
import astropy.coordinates as ac
import astropy.time as at
import numpy as np
from ..frames import ENU
import uuid
import h5py
from ..utils.stat_utils import log_normal_solve
from ..utils.gpflow_utils import train_with_bfgs, SendSummary, SaveModel, Reshape, MatrixSquare
from ..likelihoods import GaussianTecHetero
from ..kernels import ThinLayer
from ..frames import ENU
from ..models.heteroscedastic_tec_svgp import HeteroscedasticTecSVGP
from scipy.cluster.vq import kmeans2   
from gpflow.priors import LogNormal, Gaussian
from gpflow.mean_functions import Constant, Zero
from gpflow.kernels import Matern52, Matern32, White
from gpflow.features import InducingPoints
from gpflow import defer_build
import gpflow.multioutput.kernels as mk
import gpflow.multioutput.features as mf
import tensorflow as tf
from ..bayes_opt.maximum_likelihood_tec import solve_ml_tec
from timeit import default_timer
from ..bayes_opt.maximum_likelihood_tec import solve_ml_tec
from ..plotting.plot_datapack import animate_datapack, plot_data_vs_solution, plot_freq_vs_time,plot_solution_residuals

class TecSolver(Solver):
    def __init__(self,run_dir, datapack):
        super(TecSolver, self).__init__(run_dir, datapack)
        self.soltab = 'tec'

    def _finalize(self, datapack, ant_sel=None, time_sel=None, dir_sel=None, freq_sel=None, pol_sel=None,plot_level=-1,**kwargs):
        """
        Final things to run after a solve, such as plotting
        """
        with datapack:
            datapack.switch_solset(self.solset)
            axes = datapack.axes_phase
            _, freqs = datapack.get_freqs(axes['freq'])
            eval_freq = freqs[len(freqs)>>1]

        if plot_level == -1:
            # plot 1D posterior000/tec000 against data
            plot_data_vs_solution(datapack,os.path.join(self.plot_dir,"posterior_phase_1D"), data_solset='sol000', 
                    solution_solset=self.output_solset, show_prior_uncert=False,
                           ant_sel=ant_sel,time_sel=time_sel,dir_sel=dir_sel,freq_sel=slice(len(freqs)>>1, (len(freqs)>>1)+1, 1),pol_sel=pol_sel)
            
            plot_solution_residuals(datapack, os.path.join(self.plot_dir,"posterior_phase_residuals"), data_solset='sol000', solution_solset =self.output_solset, 
                    ant_sel=ant_sel,time_sel=time_sel,dir_sel=dir_sel,freq_sel=freq_sel,pol_sel=pol_sel)


        if plot_level == 0:

            
            # plot 2D posterior000/tec000 to phase at central freq
            animate_datapack(datapack,os.path.join(self.plot_dir,"posterior_phase_2D"),None,
                    ant_sel=ant_sel,time_sel=time_sel,freq_sel=freq_sel,pol_sel=pol_sel,dir_sel=dir_sel,
                    plot_crosses=True, labels_in_radec=True, observable='tec',phase_wrap=True,
                    solset=self.output_solset,tec_eval_freq=eval_freq)
        if plot_level == 1:

            # plot 2D sol000/phase000 at central freq
            animate_datapack(datapack,os.path.join(self.plot_dir,"sol000_phase_2D"),None,
                    ant_sel=ant_sel,time_sel=time_sel,freq_sel=freq_sel,pol_sel=pol_sel,dir_sel=dir_sel,
                    plot_crosses=True, labels_in_radec=True, observable='phase',phase_wrap=True,
                    solset='sol000')
            # plot 2D posterior000/tec000
            animate_datapack(datapack,os.path.join(self.plot_dir,"posterior_tec_2D"),None,
                    ant_sel=ant_sel,time_sel=time_sel,freq_sel=freq_sel,pol_sel=pol_sel,dir_sel=dir_sel,
                    plot_crosses=True, labels_in_radec=True, observable='tec',phase_wrap=False,
                    solset=self.output_solset,vmin=-0.03, vmax=0.03)
            # plot 1D sol000/tec000 against data
            plot_data_vs_solution(datapack,os.path.join(self.plot_dir,"sol000_phase_1D"), data_solset='sol000', 
                    solution_solset='sol000', show_prior_uncert=False,
                           ant_sel=ant_sel,time_sel=time_sel,dir_sel=dir_sel,freq_sel=slice(len(freqs)>>1, (len(freqs)>>1)+1, 1),pol_sel=pol_sel)
            # plot 2D sol000/phase000 at central freq
            animate_datapack(datapack,os.path.join(self.plot_dir,"sol000_phase_2D"),None,
                    ant_sel=ant_sel,time_sel=time_sel,freq_sel=freq_sel,pol_sel=pol_sel,dir_sel=dir_sel,
                    plot_crosses=True, labels_in_radec=True, observable='phase',phase_wrap=True,
                    solset='sol000')

    def _posterior_coords(self, datapack, screen=False, predict_batch_size=1024, ant_sel=None, time_sel=None, dir_sel=None, freq_sel=None, pol_sel=None, **kwargs):
        """
        generator that yeilds batchs of X coords to do posterior prediction at.
        screen: bool if True return screen coords
        """

        if screen:
            X = self._get_soltab_coords(self.output_screen_solset,'tec', ant_sel=ant_sel, time_sel=time_sel, dir_sel=None, freq_sel=freq_sel, pol_sel=pol_sel, no_freq=True)
        else:
            X = self._get_soltab_coords(self.output_solset,'tec', ant_sel=ant_sel, time_sel=time_sel, dir_sel=None, freq_sel=freq_sel, pol_sel=pol_sel, no_freq=True)
        X = X.reshape((-1, X.shape[-1]))

        shape = X.shape
        for i in range(0,shape[0], predict_batch_size):
            X_batch = X[i:min(i+predict_batch_size,shape[0]),:]
            yield X_batch


    def _save_posterior(self, ystar, varstar, datapack, data_shape, screen=False, ant_sel=None, time_sel=None, dir_sel=None, freq_sel=None, pol_sel=None, **kwargs):
        """
        Save the results in the datapack in the appropriate slots
        ystar: array [N, P]
        varstar: array [N, P]
        datapack: the DataPack
        data_shape: tuple of ints where prod(data_shape) == N
        screen: bool if True save in the screen slot
        """
        with datapack:
            if screen:
                datapack.switch_solset(self.output_screen_solset)
                datapack.select(ant=ant_sel, time=time_sel, dir=None, freq=freq_sel, pol=pol_sel)
            else:
                datapack.switch_solset(self.output_solset)
                datapack.select(ant=ant_sel, time=time_sel, dir=None, freq=freq_sel, pol=pol_sel)

            axes = datapack.axes_tec

            antenna_labels, antennas = datapack.get_antennas(axes['ant'])
            patch_names, directions = datapack.get_sources(axes['dir'])
            timestamps, times = datapack.get_times(axes['time'])
            pol_labels, pols = datapack.get_pols(axes['pol'])

            Npol, Nd, Na,  Nt = len(pols), len(directions), len(antennas), len(times)
            
            ystar = ystar.reshape((Nd, Nt, Na, Npol)).transpose((3,0,2,1))
            varstar = varstar.reshape((Nd, Na, Nt, Npol)).transpose((3,0,2,1))

            datapack.tec = ystar
            datapack.weights_tec = 1./varstar

    def _predict_posterior(self, model, X, **kwargs):
        """
        Predict the model at the coords.
        model: gpmodel
        X: array of [N, ndim]
        returns:
        ystar [N, P] predictive mean at coords
        varstar [N, P] predictive variance at the coords
        """
        shape = X.shape
        #Nd*Nt, P
        ystar, varstar = model.predict_dtec(X) 
        return ystar, varstar


    def _train_model(self, model, save_folder, writer, learning_rate=1e-2, iterations=5000, **kwargs):
        """
        Train the model.
        Returns the saved model.
        """
        train_with_bfgs(model, learning_rate, iterations, [SendSummary(model,writer,write_period=10)])#, SaveModel(save_folder, save_period=1000)])
        save_path = os.path.join(save_folder,'model.hdf5')
        logging.info("Saving model {}".format(save_path))
        self._save_model(model, save_path)
        return save_path

    def _load_model(self, model, load_file):
        """
        Load a model given by model path
        """

        vars = {}
        def _gather(name, obj):
            if isinstance(obj, h5py.Dataset):
                vars[name] = obj[...]

        with h5py.File(load_file) as f:
            f.visititems(_gather)

        model.assign(vars)

    def _save_model(self,model, save_file):
        vars = model.read_trainables()
        with h5py.File(save_file) as f:
            for name, value in vars.items():
                f[name] = value

    def _build_kernel(self, kern_dir_ls=0.3, kern_time_ls=50., kern_var=1., include_time=True, include_dir=True, **priors):

        kern_var = 1. if kern_var == 0. else kern_var

        kern_dir = Matern32(2,active_dims=slice(0,2,1))
        kern_dir.variance.trainable = False
        
        kern_dir.lengthscales = kern_dir_ls
        kern_dir_ls = log_normal_solve(kern_dir_ls, 0.5*kern_dir_ls)
        kern_dir.lengthscales.prior = LogNormal(kern_dir_ls[0], kern_dir_ls[1]**2)
        kern_dir.lengthscales.trainable = False#True

        kern_time = Matern32(1,active_dims=slice(2,3,1))
        
        kern_time.variance = kern_var
        kern_var = log_normal_solve(kern_var,0.5*kern_var)
        kern_time.variance.prior = LogNormal(kern_var[0], kern_var[1]**2)
        kern_time.variance.trainable = False#True

        kern_time.lengthscales = kern_time_ls
        kern_time_ls = log_normal_solve(kern_time_ls, 0.5*kern_time_ls)
        kern_time.lengthscales.prior = LogNormal(kern_time_ls[0], kern_time_ls[1]**2)
        kern_time.lengthscales.trainable = False#True

        kern_white = gp.kernels.White(3)
        kern_white.variance = 1.
        kern_time.variance.trainable = False#True

        if include_time:
            if include_dir:
                return kern_dir*kern_time
            return kern_time
        else:
            if include_dir:
                kern_dir.variance.trainable = True
                return kern_dir
            return kern_white

        return kern_dir*kern_time

    def _build_model(self, Y_var, X, Y, Z=None, q_mu = None, q_sqrt = None, M=None, P=None, L=None, W=None, num_data=None, jitter=1e-6, tec_scale=None, W_diag=False, **kwargs):
        """
        Build the model from the data.
        X,Y: tensors the X and Y of data

        Returns:
        gpflow.models.Model
        """

        
        settings.numerics.jitter = jitter

        with gp.defer_build():
            # Define the likelihood
            likelihood = GaussianTecHetero(tec_scale=tec_scale)
 
            q_mu = q_mu/tec_scale #M, L
            q_sqrt = q_sqrt/tec_scale# L, M, M

            kern = mk.SeparateMixedMok([self._build_kernel(kern_var = np.var(q_mu[:,l]), **kwargs.get("priors",{})) for l in range(L)], W)
            if W_diag:
#                kern.W.transform = Reshape(W.shape,(P,L,L))(gp.transforms.DiagMatrix(L)(gp.transforms.positive))
                kern.W.trainable = False
            else:
                kern.W.transform = Reshape(W.shape,(P//L,L,L))(MatrixSquare()(gp.transforms.LowerTriangular(L,P//L)))
                kern.W.trainable = True
            
            feature = mf.MixedKernelSeparateMof([InducingPoints(Z) for _ in range(L)])
            mean = Zero()
            model = HeteroscedasticTecSVGP(Y_var, X, Y, kern, likelihood, 
                        feat = feature,
                        mean_function=mean, 
                        minibatch_size=None,
                        num_latent = P, 
                        num_data = num_data,
                        whiten = False, 
                        q_mu = q_mu, 
                        q_sqrt = q_sqrt)
            for feat in feature.feat_list:
                feat.Z.trainable = True
            model.q_mu.trainable = True
            model.q_sqrt.trainable = True
#            model.q_sqrt.prior = gp.priors.Gaussian(q_sqrt, 0.005**2)
            model.compile()
            tf.summary.image('W',kern.W.constrained_tensor[None,:,:,None])
            tf.summary.image('q_mu',model.q_mu.constrained_tensor[None,:,:,None])
            tf.summary.image('q_sqrt',model.q_sqrt.constrained_tensor[:,:,:,None])
        return model

    def _get_data(self,indices,data_shape, dtype=settings.np_float):
        """
        Return a selection of (Y_var, freqs, X, Y) order gets fed to _build_model
        indices : array of indices that index into data [N, len(data_shape)]
        data_shape : tuple of dim sizes for index raveling
        Returns:
        X array tf.float64 [N, ndims]
        Y array tf.float64 [N, P]
        weights array tf.float64 [N, P]
        """
        idx = np.sort(np.ravel_multi_index(indices.T, data_shape))

        with h5py.File(self.coord_file) as f:
            Y_ = f['/data/Y']
            Y = np.stack([Y_[i,...] for i in idx], axis=0)
            Y_var_ = f['/data/Y_var']
            Y_var = np.stack([Y_var_[i,...] for i in idx], axis=0)
            X_ = f['/data/X']
            X = np.stack([X_[i,...] for i in idx], axis=0)

        return Y_var.astype(dtype), X.astype(dtype), Y.astype(dtype)

    def _get_soltab_coords(self, solset, soltab, ant_sel=None, time_sel=None, dir_sel=None, freq_sel=None, pol_sel=None, no_freq=True, **kwargs):
        """
        Returns:
        array data_shape + [ndim]
            The coordinates for the data
            Nd, (Nf,) Nt, ndim in this case
        """
        logging.info("Calculating coordinates for {}/{}".format(solset, soltab))
        with self.datapack:
            self.datapack.switch_solset(solset)
            self.datapack.select(ant=ant_sel,time=time_sel, dir=dir_sel, freq=freq_sel, pol=pol_sel)
            
            axes = self.datapack.__getattr__("axes_{}".format(soltab))
            
            antenna_labels, antennas = self.datapack.get_antennas(axes['ant'])
            patch_names, directions = self.datapack.get_sources(axes['dir'])
            timestamps, times = self.datapack.get_times(axes['time'])
            if not no_freq:
                freq_labels, freqs = self.datapack.get_freqs(axes['freq'])
                Nf = len(freqs)
            pol_labels, pols = self.datapack.get_pols(axes['pol'])

            Npol, Nd, Na, Nt = len(pols), len(directions), len(antennas), len(times)


            ra = directions.ra.deg
            dec = directions.dec.deg
            X_d = np.stack([ra,dec],axis=1)
            X_t = ((times.mjd - times[0].mjd)*86400.)[:,None]
            if not no_freq:
                return np.tile(make_coord_array(X_d, X_t, flat=False)[:,None,:,:], (1, Nf, 1, 1))
            return make_coord_array(X_d, X_t, flat=False)

    def _get_soltab_data(self,solset, soltab, ant_sel=None, time_sel=None, dir_sel=None, freq_sel=None, pol_sel=None, **kwargs):
        """
        Returns:
        array data_shape + [shape]
            The data with initial data_shape same as returned by coords
            Nd, Nf, Nt, Na, Npol in this case
        """
        logging.info("Calculating data for {}/{}".format(solset, soltab))
        with self.datapack:
            self.datapack.switch_solset(solset)
            self.datapack.select(ant=ant_sel,time=time_sel, dir=dir_sel, freq=freq_sel, pol=pol_sel)

            # Npol, Nd, Na, Nt
            tec, _ = self.datapack.tec
            Npol, Nd, Na, Nt = tec.shape
            weights_tec, _ = self.datapack.weights_tec
            tec_var = 1./weights_tec
            tec_var = np.where(~np.isfinite(tec_var), 0.005**2, tec_var)

            
            # Nd, Nt, Na,Npol
            Y = tec.transpose((1,3,2,0))
            Y_var = tec_var.transpose((1,3,2,0))

            return Y, Y_var

    
    def _prepare_data(self,datapack,ant_sel=None, time_sel=None, dir_sel=None, freq_sel=None, pol_sel=None,reweight_obs=False, recalculate_coords=False, dof_ratio=40., weight_smooth_len=40, screen_res=30, solset='sol000',coord_file=None, posterior_time_sel = None, **kwargs):
        """
        In this case we are solving for phase as a function of antenna location, direction, and time using dtec as a model.
        Prepares the data in the datapack for solving.
        Likely steps:
        1. Determine measurement variance
        2. Calculate data coordinates (new sol-tab is made inside the solset)
        3. Make new hdf5 file linking to X,Y weights
        datapack : DataPack solutions to solve
        Returns:
        data_shape, build_params
        """
        self.solset = solset
        self.soltab = 'tec'
        with self.datapack:
            ###
            # calculate the coordinates for the solve (currently [ra, dec, t]) on the whole dataset
            # creates the posterior and screen tables
            self. _maybe_create_posterior_solsets(self.solset, self.soltab, **kwargs)
            
            self.datapack.switch_solset(solset)                
            self.datapack.select(ant=ant_sel,time=time_sel, dir=dir_sel, pol=pol_sel)
            axes = self.datapack.__getattr__("axes_{}".format(self.soltab))
            antenna_labels, antennas = self.datapack.get_antennas(axes['ant'])
            patch_names, directions = self.datapack.get_sources(axes['dir'])
            timestamps, times = self.datapack.get_times(axes['time'])
            pol_labels, pols = self.datapack.get_pols(axes['pol'])

            Npol, Nd, Na, Nt = len(pols), len(directions), len(antennas),len(times)
            num_data = Nt*Nd
            ndim = 3
            assert dof_ratio >= 1., "Shouldn't model data with more dof than data"
            M = int(np.ceil(Nt * Nd / dof_ratio))
            P = Npol*Na
            L = Na #each soltab over same coordinates can be 1
            minibatch_size = kwargs.get('minibatch_size',None)
            logging.info("Using {} inducing points to sparsely model {} data points".format(M, Nd*Nt))
            logging.info("Number of latents L=Na={}, number of outputs P=Na*Npol={}".format(L,P))
            logging.info("Performing minibatching with {} sized batches".format(minibatch_size))

            logging.info("Kuu: {}x{}x{} [{:.2f} MB]".format(L,M,M,(8*L*M*M)/(1<<20)))
            logging.info("Kuf: {}x{}x{} [{:.2f} MB]".format(L,M,minibatch_size,(8*L*M*minibatch_size)/(1<<20)))
            logging.info("Kff: {}x{}x{} [{:.2f} MB]".format(L,minibatch_size,minibatch_size,(8*L*minibatch_size**2)/(1<<20)))

                       
            #Nd,  Nt, ndim
            X = self._get_soltab_coords(self.solset, self.soltab, ant_sel=ant_sel,time_sel=time_sel, dir_sel=dir_sel, freq_sel=freq_sel, pol_sel=pol_sel)
            #Nd,  Nt, Na, Npol
            Y, Y_var = self._get_soltab_data(self.solset, self.soltab, ant_sel=ant_sel,time_sel=time_sel, dir_sel=dir_sel, freq_sel=freq_sel, pol_sel=pol_sel)
            
             ###
            # input coords not thread safe
            self.coord_file = os.path.join(self.run_dir,"data_source_{}.hdf5".format(str(uuid.uuid4())))
            logging.info("Creating temporary data source file: {}".format(self.coord_file)) 

            with h5py.File(self.coord_file) as f:
                #Nd*Nt, ndim
                f['/data/X'] = X.reshape((-1, ndim))
                f['/data/Y'] = Y.reshape((-1, P))
                f['/data/Y_var'] = Y_var.reshape((-1, P))
            
            # num of input data tensor iterators (Y_var, X, Y)
            self.model_input_num = 3


            logging.info("Initializing sparse conditions: q_mu and q_sqrt")
            self.datapack.switch_solset(self.solset)
            self.datapack.select(ant=ant_sel,time=time_sel, dir=dir_sel, freq=freq_sel, pol=pol_sel)
            #Npol, Nd, Na, Nt
            tec, _ = self.datapack.tec
            tec_weights, _ = self.datapack.weights_tec
            tec_std = np.sqrt(1./tec_weights)
            tec_std = np.where(~np.isfinite(tec_std), 0.005, tec_std)
            #Npol, Nd, Na, Nt -> Nd, Na, Nt -> Nd, Nt, Na -> Nd*Nt, Na
            tec = tec.mean(0).transpose((0,2,1)).reshape((-1, L))
            tec_std = tec_std.max(0).transpose((0,2,1)).reshape((-1,L))

            idx = np.random.choice(Nd*Nt, size=M, replace=False)
            Z = X[:,:,:].reshape((-1,ndim))[idx,:]#M,D
            q_mu = tec[idx,:]#M, Na
            q_sqrt = np.stack([np.diag(tec_std[idx,l]) for l in range(L)],axis=0)#L, M, M

            W = np.reshape(np.ones(Npol)[:,None,None]*np.eye(Na)[None, :,:],(P,L))
                        
            data_shape = (Nd, Nt)
            build_params = {
                    'Z': Z,
                    'W': W,
                    'q_mu': q_mu,
                    'q_sqrt': q_sqrt,
                    'M': M,
                    'P':P,
                    'L':L,
                    'num_data':num_data}

            return data_shape, build_params
    
    def _maybe_create_posterior_solsets(self, solset, soltab, screen_res=30, num_threads = None, remake_posterior_solsets=False, **kwargs):
        with self.datapack:
            if remake_posterior_solsets:
                self.datapack.delete_solset(self.output_solset)
                self.datapack.delete_solset(self.output_screen_solset)

            if not self.datapack.is_solset(self.output_solset) or not self.datapack.is_solset(self.output_screen_solset):
                logging.info("Creating posterior solsets for facets and {}x{} screen".format(screen_res,screen_res))

                self.datapack.switch_solset(solset)
                self.datapack.select(ant=None,time=None, dir=None, freq=None, pol=None)
                axes = self.datapack.__getattr__("axes_{}".format(soltab))
                
                antenna_labels, antennas = self.datapack.get_antennas(axes['ant'])
                patch_names, directions = self.datapack.get_sources(axes['dir'])
                timestamps, times = self.datapack.get_times(axes['time'])
                pol_labels, pols = self.datapack.get_pols(axes['pol'])

                Npol, Nd, Na,  Nt = len(pols), len(directions), len(antennas),  len(times)
                
                screen_ra = np.linspace(np.min(directions.ra.rad) - 0.25*np.pi/180., 
                        np.max(directions.ra.rad) + 0.25*np.pi/180., screen_res)
                screen_dec = np.linspace(max(-90.*np.pi/180.,np.min(directions.dec.rad) - 0.25*np.pi/180.), 
                        min(90.*np.pi/180.,np.max(directions.dec.rad) + 0.25*np.pi/180.), screen_res)
                screen_directions = np.stack([m.flatten() \
                        for m in np.meshgrid(screen_ra, screen_dec, indexing='ij')], axis=1)
                screen_directions = ac.SkyCoord(screen_directions[:,0]*au.rad,screen_directions[:,1]*au.rad,frame='icrs')
                Nd_screen = screen_res**2
                
                self.datapack.switch_solset(self.output_solset, 
                        array_file=DataPack.lofar_array, 
                        directions = np.stack([directions.ra.rad,directions.dec.rad],axis=1), patch_names=patch_names)
                self.datapack.add_freq_indep_tab('tec', times.mjd*86400., pols = pol_labels)   
                
                self.datapack.switch_solset(self.output_screen_solset, 
                        array_file = DataPack.lofar_array, 
                        directions = np.stack([screen_directions.ra.rad,screen_directions.dec.rad],axis=1))
                self.datapack.add_freq_indep_tab('tec', times.mjd*86400., pols = pol_labels)
                
                self.datapack.switch_solset(solset)
