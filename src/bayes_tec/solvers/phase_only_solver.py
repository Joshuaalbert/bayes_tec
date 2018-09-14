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
from ..utils.gpflow_utils import train_with_adam, SendSummary, SaveModel
from ..likelihoods import WrappedPhaseGaussianMulti
from ..kernels import ThinLayer
from ..frames import ENU
from ..models.homoscedastic_phaseonly_svgp import HomoscedasticPhaseOnlySVGP
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
     

class PhaseOnlySolver(Solver):
    def __init__(self,run_dir, datapack):
        super(PhaseOnlySolver, self).__init__(run_dir, datapack)

    def _posterior_coords(self, datapack, screen=False, minibatch_size=128, **kwargs):
        """
        generator that yeilds batchs of X coords to do posterior prediction at.
        screen: bool if True return screen coords
        """

        with h5py.File(self.coord_file) as f:
            if screen:
                # saved flat already
                X = f['/posterior/X/screen'][...]
            else:
                X = f['/posterior/X/facets'][...]
        shape = X.shape
        for i in range(0,X.shape[0],minibatch_size):
            X_batch = X[i:min(i+minibatch_size,X.shape[0]),:]
            yield X_batch


    def _save_posterior(self, ystar, varstar, datapack, data_shape, screen=False, ant_sel=None, time_sel=None, dir_sel=None, freq_sel=None, pol_sel=None, **kwargs):
        """
        Save the results in the datapack in the appropriate slots
        ystar: array [N, P]
        varstar: array [N,P]
        datapack: the DataPack
        data_shape: tuple of ints where prod(data_shape) == N
        screen: bool if True save in the screen slot
        """
        with datapack:
            if screen:
                datapack.switch_solset("screen_sol")
                datapack.select(ant=ant_sel,time=time_sel, dir=None, freq=freq_sel, pol=pol_sel)
            else:
                datapack.switch_solset("posterior_sol")
                datapack.select(ant=ant_sel,time=time_sel, dir=None, freq=freq_sel, pol=pol_sel)

            axes = datapack.axes_tec

            antenna_labels, antennas = datapack.get_antennas(axes['ant'])
            patch_names, directions = datapack.get_sources(axes['dir'])
            timestamps, times = datapack.get_times(axes['time'])
            pol_labels, pols = datapack.get_pols(axes['pol'])

            Npol, Nd, Na,  Nt = len(pols), len(directions), len(antennas), len(times)
            
            ystar = ystar.reshape((Nd, Na, Nt, Npol)).transpose((3,0,1,2))
            varstar = varstar.reshape((Nd, Na, Nt, Npol)).transpose((3,0,1,2))

            datapack.tec = ystar
            datapack.weights_tec = np.where(varstar > 0., 1./varstar, 1e6)        

    def _predict_posterior(self, model, X, **kwargs):
        """
        Predict the model at the coords.
        X: array of [Nd* Na*Nt, ndim]
        returns:
        ystar [Nd* Na* Nt, P] predictive mean at coords
        varstar [Nd* Na* Nt, P] predictive variance at the coords
        """
        shape = X.shape
        #Nd*Na*Nt, P
        ystar, varstar = model.predict_dtec(X) 
        return ystar, varstar


    def _train_model(self, model, save_folder, writer, learning_rate=1e-2, iterations=5000, **kwargs):
        """
        Train the model.
        Returns the saved model.
        """
        train_with_adam(model, learning_rate, iterations, [SendSummary(model,writer,write_period=10)])#, SaveModel(save_folder, save_period=1000)])
        save_path = os.path.join(save_folder,'model.hdf5')
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

    def _build_kernel(self, kern_dir_ls=None, kern_space_ls=None, kern_time_ls=None, kern_var=None, **priors):
        kern_dir = Matern32(2,active_dims=slice(1,3,1))
        kern_dir.variance.trainable = False
        kern_dir.lengthscales.prior = LogNormal(kern_dir_ls[0], kern_dir_ls[1]**2)
        kern_dir.lengthscales = np.exp(kern_dir_ls[0])
        kern_time = Matern32(1,active_dims=slice(6,7,1))
        kern_time.variance.trainable = True
        kern_time.variance.prior = LogNormal(kern_var[0], kern_var[1]**2)
        kern_time.variance = np.exp(kern_var[0])
        kern_time.lengthscales.prior = LogNormal(kern_time_ls[0], kern_time_ls[1]**2)
        kern_time.lengthscales = np.exp(kern_time_ls[0])
        return kern_dir*kern_time



        kern_all = Matern32(7,ARD=True)
        kern_all.lengthscales = np.exp(np.array([kern_dir_ls[0],kern_dir_ls[0],kern_dir_ls[0],kern_space_ls[0],kern_space_ls[0],kern_space_ls[0],kern_time_ls[0]]))
        kern_all.lengthscales.prior = LogNormal(np.array([kern_dir_ls[0],kern_dir_ls[0],kern_dir_ls[0],kern_space_ls[0],kern_space_ls[0],kern_space_ls[0],kern_time_ls[0]]),
                np.array([kern_dir_ls[1],kern_dir_ls[1],kern_dir_ls[1],kern_space_ls[1],kern_space_ls[1],kern_space_ls[1],kern_time_ls[1]])**2)
        kern_all.variance.prior = LogNormal(kern_var[0], kern_var[1]**2)
        kern_all.variance = np.exp(kern_var[0])
        return kern_all


        kern_thin_layer = ThinLayer(np.array([0.,0.,0.]), a=400, b=20., l=10., tec_scale=tec_scale,
                active_dims=[0,3,4,5])
        kern_pert = Matern32(3, active_dims=[1,2,6], ARD=True)
        kern_pert.lengthscales.prior = LogNormal([kern_dir_ls[0], kern_dir_ls[0], kern_time_ls[0]], 
                [kern_dir_ls[1]**2, kern_dir_ls[1]**2, kern_time_ls[1]**2]
                )
        kern_pert.lengthscales = np.exp(np.array([kern_dir_ls[0], kern_dir_ls[0], kern_time_ls[0]]))

        kern_pert.variance.prior = LogNormal(kern_var[0], kern_var[1]**2)
        kern_pert.variance = np.exp(kern_var[0])
        

        kern = kern_thin_layer + kern_pert
        return kern




    def _build_model(self, X, Y, weights=None, freqs=None, Z=None, q_mu = None, q_sqrt = None, M=None, P=None, L=None, num_data=None, jitter=1e-6, tec_scale=0.001, **kwargs):
        """
        Build the model from the data.
        X,Y: tensors the X and Y of data
        weights: statistical weights that may be None

        Returns:
        gpflow.models.Model
        """

        
        settings.numerics.jitter = jitter

        priors = self._generate_priors(tec_scale)
        likelihood_var = priors.get('likelihood_var', None)
        
        
        with gp.defer_build():
            # Define the likelihood
            likelihood = WrappedPhaseGaussianMulti(tec_scale=tec_scale,freqs=freqs)
            likelihood.variance = 20*np.pi/180.#np.exp(likelihood_var[0]) #median as initial
#            likelihood.variance.prior = LogNormal(likelihood_var[0],likelihood_var[1]**2)
#            likelihood.variance.transform = gp.transforms.Rescale(np.pi/180.)(gp.transforms.positive)
            likelihood.variance.set_trainable(False)

                            
            q_mu = q_mu/tec_scale
            q_sqrt = q_sqrt/tec_scale

            W = np.ones((P,L))#np.random.normal(size=[P,L])
            kern = mk.SeparateMixedMok([self._build_kernel(**priors) for _ in range(L)], W)
            kern.W.set_trainable(False)
            feature = mf.MixedKernelSeparateMof([InducingPoints(Z) for _ in range(L)])
            mean = Zero()
            model = HomoscedasticPhaseOnlySVGP(weights, X, Y, kern, likelihood, 
                        feat = feature,
                        mean_function=mean, 
                        minibatch_size=None,
                        num_latent = P, 
                        num_data = num_data,
                        whiten=False, 
                        q_mu = q_mu, 
                        q_sqrt=q_sqrt)
            model.compile()
        return model

    def _generate_priors(self, tec_scale):
        """
        Create the global (independent) model priors
        Returns:
        dictionary of priors, each prior is a tuple for LogNormal or Normal priors
        """

        # Gaussian likelihood log-normal prior
        phase_uncert_deg = 30.
        phase_var_rad = (phase_uncert_deg*np.pi/180.)**2
        lik_var = log_normal_solve(phase_var_rad, 0.1*phase_var_rad)
        # TEC kern time lengthscale log-normal prior (seconds)
        kern_time_ls = log_normal_solve(50., 20.)
        # TEC kern dir lengthscale log-normal prior (radians)
        kern_dir_ls = log_normal_solve(0.5*np.pi/180., 0.2*np.pi/180.)
        # kern space (km)
        kern_space_ls = log_normal_solve(5.,5.)
        # TEC kern variance priors
        kern_sigma = 0.005/tec_scale
        kern_var = log_normal_solve(kern_sigma**2,0.1*kern_sigma**2)

        logging.info("likelihood var logGaussian {} median (deg) {}".format(lik_var,np.exp(lik_var[0])*180./np.pi))
        logging.info("tec kern var logGaussian {} median (tec) {}".format(kern_var,
                np.sqrt(np.exp(kern_var[0]))*tec_scale))
        logging.info("tec kern time ls logGaussian {} median (sec) {} ".format(kern_time_ls,
                np.exp(kern_time_ls[0])))
        logging.info("tec kern dir ls logGaussian {} median (deg) {}".format(kern_dir_ls,
                np.exp(kern_dir_ls[0])*180./np.pi))
        logging.info("tec kern space ls logGaussian {} median () {}".format(kern_space_ls,
                np.exp(kern_space_ls[0])))


        priors = {
                "likelihood_var": lik_var,
                "kern_time_ls":kern_time_ls,
                "kern_dir_ls":kern_dir_ls,
                "kern_space_ls":kern_space_ls,
                "kern_var":kern_var
                }
        return priors


    def _get_data(self,indices,data_shape, dtype=settings.np_float):
        """
        Return a selection of (X,Y,weights/var)
        indices : array of indices that index into data [N, len(data_shape)]
        data_shape : tuple of dim sizes for index raveling
        Returns:
        X array tf.float64 [N, ndims]
        Y array tf.float64 [N, P]
        weights array tf.float64 [N, P]
        """
        dir_sel = indices[:,0]
        ant_sel = indices[:,1]
        time_sel = indices[:,2]

        idx = np.sort(np.ravel_multi_index((dir_sel, ant_sel, time_sel), data_shape))

        with h5py.File(self.coord_file) as f:
            phase_ = f['/data/Y']
            Y = np.stack([phase_[i,:,:] for i in idx],axis=0)
            weights_ = f['/data/weights']
            weights = np.stack([weights_[i,:,:] for i in idx], axis=0)
            X_ = f['/data/X']
            X = np.stack([X_[i,:] for i in idx], axis=0)

        return X.astype(dtype),Y.astype(dtype),weights.astype(dtype)
        


    def _prepare_data(self,datapack,ant_sel=None, time_sel=None, dir_sel=None, freq_sel=None, pol_sel=None,reweight_obs=True, recalculate_coords=False, dof_ratio=40., weight_smooth_len=40, screen_res=30, solset='sol000',coord_file=None, posterior_time_sel = None, **kwargs):
        """
        In this case we are solving for phase as a function of antenna location, direction, and time using dtec as a model.
        Prepares the data in the datapack for solving.
        Likely steps:
        1. Determine measurement variance
        2. Calculate data coordinates (new sol-tab is made inside the solset)
        3. Make new hdf5 file linking to X,Y weights
        datapack : DataPack solutions to solve
        Returns:
        data_shape
        """
        logging.info("Preparing data")
        self.solset = solset
        self.soltab = 'phase'
        with self.datapack:
            self._maybe_fill_coords(self.solset, self.soltab, screen_res=screen_res, recalculate_coords=recalculate_coords, **kwargs)
            
            self.datapack.switch_solset(solset)
            self.datapack.select(ant=ant_sel,time=time_sel, dir=dir_sel, freq=freq_sel, pol=pol_sel)
            
            if reweight_obs:
                logging.info("Re-calculating weights...")
                phase, axes = self.datapack.phase
                weights = calculate_weights(phase,indep_axis = -1, num_threads = None, N=weight_smooth_len, 
                        phase_wrap=True, min_uncert=5*np.pi/180.)
                weights_pre, axes = self.datapack.weights_phase# should be ML estimates
                weights = 1./(1./weights_pre + 1./weights)
                self.datapack.weights_phase = weights

            axes = self.datapack.__getattr__("axes_{}".format(self.soltab))

            antenna_labels, antennas = self.datapack.get_antennas(axes['ant'])
            patch_names, directions = self.datapack.get_sources(axes['dir'])
            timestamps, times = self.datapack.get_times(axes['time'])
            freq_labels, freqs = self.datapack.get_freqs(axes['freq'])
            pol_labels, pols = self.datapack.get_pols(axes['pol'])

            Npol, Nd, Na, Nf, Nt = len(pols), len(directions), len(antennas), len(freqs), len(times)
            num_data = Nt*Nd*Na
            M = int(np.ceil(Nt * Nd * Na / dof_ratio))
            logging.info("Using {} inducing points".format(M))
            P = Npol
            L = 1 #each soltab over same coordinates can be 1

            
            ###
            # input coords not thread safe
            self.coord_file = os.path.join(self.run_dir,"data_source_{}.hdf5".format(str(uuid.uuid4())))
            logging.info("Calculating coordinates into temporary file: {}".format(self.coord_file))

            

            if posterior_time_sel is None:
                posterior_time_sel = time_sel

            self.datapack.switch_solset("X_facets")
            self.datapack.select(ant=ant_sel,time=posterior_time_sel, dir=None, freq=freq_sel, pol=pol_sel)
            #7, Nd, Na, Nt
            X_facets, _ = self.datapack.coords
            X_facets = X_facets.transpose((1,2,3,0)).reshape((-1,7))

            self.datapack.select(ant=ant_sel,time=time_sel, dir=dir_sel, freq=freq_sel, pol=pol_sel)
            #7, Nd, Na, Nt
            X, _ = self.datapack.coords
            X = X.transpose((1,2,3,0)).reshape((-1,7))

            self.datapack.switch_solset("X_screen")
            self.datapack.select(ant=ant_sel,time=posterior_time_sel, dir=None, freq=freq_sel, pol=pol_sel)
            #7, Nd_screen, Na, Nt
            X_screen, _ = self.datapack.coords
            X_screen = X_screen.transpose((1,2,3,0)).reshape((-1,7))
            
            self.datapack.switch_solset(self.solset)
            self.datapack.select(ant=ant_sel,time=time_sel, dir=dir_sel, freq=freq_sel, pol=pol_sel)
            with h5py.File(self.coord_file) as f:
                f['/data/X'] = X
                f['/posterior/X/facets'] = X_facets
                f['/posterior/X/screen'] = X_screen
                # Npol, Nd, Na, Nf, Nt
                phase, _ = self.datapack.phase
                weights, _ = self.datapack.weights_phase
                # Nd, Na, Nt, Npol, Nf -> Nd*Na*Nt, Npol, Nf
                phase = phase.transpose((1,2,4,0,3)).reshape((-1, Npol, Nf))              
                weights = weights.transpose((1,2,4,0,3)).reshape((-1, Npol, Nf))
                f['/data/Y'] = phase
                f['/data/weights'] = weights
            
            self.datapack.switch_solset("posterior_sol")
            self.datapack.select(ant=ant_sel,time=time_sel, dir=dir_sel, freq=freq_sel, pol=pol_sel)
            tec_ml, _ = self.datapack.tec
            tec_ml = tec_ml.transpose((1,2,3,0)).reshape((-1, Npol)).mean(1)#N

            sigma_tec_ml, _ = self.datapack.weights_tec
            np.sqrt(sigma_tec_ml, out=sigma_tec_ml)
            sigma_tec_ml = 1./sigma_tec_ml
            sigma_tec_ml = sigma_tec_ml.transpose((1,2,3,0)).reshape((-1, Npol)).mean(1)#N

            idx = np.random.choice(X.shape[0], size=M, replace=False)
            Z = X[idx,:]#M,D
            q_mu = np.tile(tec_ml[idx,None], (1,L))#M,L
            q_sqrt = np.tile(np.diag(sigma_tec_ml)[None,:,:],(L,1,1))#L,M,M

#            Z = kmeans2(X,M,minit='points')[0] if X.shape[0] < 1e4 \
#                else X[np.random.choice(X.shape[0], size=M,replace=False), :]
                        
            self.datapack.switch_solset(self.solset)

            data_shape = (Nd, Na, Nt)
            build_params = {'freqs': freqs,
                    'Z': Z,
                    'q_mu': q_mu,
                    'q_sqrt': q_sqrt,
                    'M': M,
                    'P':P,
                    'L':L,
                    'num_data':num_data}

            return data_shape, build_params

    def _parallel_coord_transform(self, array_center,time, time0, directions, screen_directions, antennas):
        enu = ENU(location=array_center, obstime=time)
        enu_dirs = directions.transform_to(enu)
        enu_ants = antennas.transform_to(enu)
        east = enu_ants.east.to(au.km).value
        north = enu_ants.north.to(au.km).value
        up = enu_ants.up.to(au.km).value
        kz = enu_dirs.up.value
        ra = directions.ra.rad
        dec = directions.dec.rad
        X = make_coord_array(
                np.stack([kz,ra,dec],axis=-1),
                np.stack([east,north,up],axis=-1), 
                np.array([[time.mjd*86400. - time0]]),flat=False)
        
        enu_dirs = screen_directions.transform_to(enu)
        kz = enu_dirs.up.value
        ra = screen_directions.ra.rad
        dec = screen_directions.dec.rad
        X_screen = make_coord_array(
                np.stack([kz,ra,dec],axis=-1),
                np.stack([east,north,up],axis=-1), 
                np.array([[time.mjd*86400. - time0]]),flat=False)

        return X, X_screen


    def _maybe_fill_coords(self, solset, soltab, screen_res=30, num_threads = None, recalculate_coords=False, **kwargs):
        with self.datapack:
            logging.info("Calculating coordinates")
            if recalculate_coords:
                self.datapack.delete_solset("X_facets")
                self.datapack.delete_solset("X_screen")


            if not self.datapack.is_solset('X_facets') or not self.datapack.is_solset("X_screen"):
                self.datapack.switch_solset(solset)
                self.datapack.select(ant=None,time=None, dir=None, freq=None, pol=None)
                axes = self.datapack.__getattr__("axes_{}".format(soltab))
                
                antenna_labels, antennas = self.datapack.get_antennas(axes['ant'])
                patch_names, directions = self.datapack.get_sources(axes['dir'])
                timestamps, times = self.datapack.get_times(axes['time'])
                freq_labels, freqs = self.datapack.get_freqs(axes['freq'])
                pol_labels, pols = self.datapack.get_pols(axes['pol'])

                Npol, Nd, Na, Nf, Nt = len(pols), len(directions), len(antennas), len(freqs), len(times)
                
                screen_ra = np.linspace(np.min(directions.ra.rad) - 0.25*np.pi/180., 
                        np.max(directions.ra.rad) + 0.25*np.pi/180., screen_res)
                screen_dec = np.linspace(max(-90.*np.pi/180.,np.min(directions.dec.rad) - 0.25*np.pi/180.), 
                        min(90.*np.pi/180.,np.max(directions.dec.rad) + 0.25*np.pi/180.), screen_res)
                screen_directions = np.stack([m.flatten() \
                        for m in np.meshgrid(screen_ra, screen_dec, indexing='ij')], axis=1)
                screen_directions = ac.SkyCoord(screen_directions[:,0]*au.rad,screen_directions[:,1]*au.rad,frame='icrs')
                Nd_screen = screen_res**2

                ###
                # fill them out
                X = np.zeros((Nd,Na,Nt,7),dtype=np.float32)
                X_screen = np.zeros((Nd_screen,Na,Nt,7),dtype=np.float32)
                t0 = default_timer()
                for j,time in enumerate(times):
                    X[:,:,j:j+1,:],X_screen[:,:,j:j+1,:] = _parallel_coord_transform(self.datapack.array_center, time, times[0].mjd*86400., directions, screen_directions, antennas)
                    if (j+1) % (Nt//20) == 0:
                        time_left = (Nt - j - 1) * (default_timer() - t0)/ (j + 1)
                        logging.info("{:.2f}% done... {:.2f} seconds left".format(100*(j+1)/Nt, time_left))

                
                self.datapack.switch_solset("X_facets", 
                        array_file=DataPack.lofar_array, 
                        directions = np.stack([directions.ra.rad,directions.dec.rad],axis=1))
                self.datapack.add_freq_indep_tab('coords', times.mjd*86400., pols = ('kz','ra','dec','east','north','up','time'))
                self.datapack.coords = X.transpose((3,0,1,2))
                
                
                
                self.datapack.switch_solset("X_screen", 
                        array_file=DataPack.lofar_array, 
                        directions = np.stack([screen_directions.ra.rad,screen_directions.dec.rad],axis=1))
                self.datapack.add_freq_indep_tab('coords', times.mjd*86400., pols = ('kz','ra','dec','east','north','up','time'))
                self.datapack.coords = X_screen.transpose((3,0,1,2))        
                
                self.datapack.delete_solset("screen_sol")
                self.datapack.switch_solset("screen_sol", 
                        array_file = DataPack.lofar_array, 
                        directions = np.stack([screen_directions.ra.rad,screen_directions.dec.rad],axis=1))
                self.datapack.add_freq_indep_tab('tec', times.mjd*86400., pols = pol_labels)
                
                logging.info("Solving for maximum likelihood solutions using Bayesian optimization")
                phase, axes = datapack.phase
                phase = phase.transpose((0,1,2,4,3))
                shape = phase.shape
                phase = phase.reshape((-1, Nf))
                tec_ml, sigma_ml = solve_ml_tec(phase, freqs, batch_size=2048, max_tec=0.4, n_iter=23, t=1.)
                tec_ml = tec_ml.reshape((Npol, Nd, Na, Nt))
                sigma_ml = sigma_ml.reshape((Npol, Nd, Na, Nt))
                logging.info("Finished ML solve")

                self.datapack.delete_solset("posterior_sol")
                self.datapack.switch_solset("posterior_sol", 
                        array_file=DataPack.lofar_array, 
                        directions = np.stack([directions.ra.rad,directions.dec.rad],axis=1))
                self.datapack.add_freq_indep_tab('tec', times.mjd*86400., pols = pol_labels)

                self.datapack.tec = tec_ml
                tec_conversion = 8.448e9/freqs
                nu_bar = np.mean(tec_conversion)
                tec_var = sigma_ml*nu_bar
                self.datapack.weights_tec = np.where(tec_var > 0., 1./tec_var, 1.)
                
                self.datapack.switch_solset(solset)

                phase_var = np.tile(np.square(sigma_ml)[:,:,:,None,:], (1,1,1,Nf,1))
                self.datapack.weights_phase = np.where(phase_var > 0., 1./phase_var, 1.)
