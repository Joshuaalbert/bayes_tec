from ..datapack import DataPack
from ..logging import logging
import os
import glob
import tensorflow as tf
from gpflow import settings
import astropy.units as au
import astropy.coordinates as ac
import astropy.time as at
import numpy as np
from ..frames import ENU
from ..utils.data_utils import make_coord_array
from timeit import default_timer
from concurrent import futures
from tensorflow.python import debug as tf_debug


def _parallel_coord_transform(array_center,time, time0, directions, screen_directions, antennas):
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

    def solve(self, load_model=None, **kwargs):
        """Run the solver"""
        data_shape, build_params = self._prepare_data(self.datapack, **kwargs)
        
        graph = tf.Graph()
        sess = self._new_session(graph, **kwargs)

        summary_id = len(glob.glob(os.path.join(self.summary_dir,"summary_*")))
        summary_folder = os.path.join(self.summary_dir,"summary_{:03d}".format(summary_id))
        os.makedirs(summary_folder,exist_ok=True)
        with graph.as_default(), sess.as_default(), \
                tf.summary.FileWriter(summary_folder, graph) as writer:
            _, (X, Y, weights) = self._train_dataset_iterator(data_shape, sess=sess, **kwargs)
            model = self._build_model(X, Y, weights=weights, **build_params, **kwargs)
            if load_model is not None:
                self._load_model(model, load_model)
            # train model
            save_id = len(glob.glob(os.path.join(self.save_dir,"save_*")))
            save_folder = os.path.join(self.save_dir,"save_{:03d}".format(save_id))
            os.makedirs(save_folder,exist_ok=True)
            saved_model = self._train_model(model, save_folder, writer, **kwargs)
            
            # TODO break prediction into another call (i.e. not in solve)
            self._load_model(model, saved_model)
            
            ystar, varstar = [], []
            for X in self._posterior_coords(self.datapack, screen=False, **kwargs):
                _ystar, _varstar = self._predict_posterior(model, X)
                ystar.append(_ystar)
                varstar.append(_varstar)
            ystar = np.concatenate(ystar, axis=0)
            varstar = np.concatenate(varstar, axis=0)
            self._save_posterior(ystar, varstar, self.datapack, data_shape, screen=False, **kwargs)

            ystar, varstar = [], []
            for X in self._posterior_coords(self.datapack, screen=True, **kwargs):
                _ystar, _varstar = self._predict_posterior(model, X)
                ystar.append(_ystar)
                varstar.append(_varstar)
            ystar = np.concatenate(ystar, axis=0)
            varstar = np.concatenate(varstar, axis=0)
            self._save_posterior(ystar, varstar, self.datapack, data_shape, screen=True, **kwargs)

    def _maybe_fill_coords(self, solset, soltab, datapack, screen_res=30, num_threads = None, recalculate_coords=False, **kwargs):
        ### TODO this should be not in Solver but specific to the type of solver.
        # here so that block solver runs simply for now, will need to rearrange
        # probably will be easiest to have a datapack_prep class that either call
        with datapack:
            logging.info("Calculating coordinates")
            if recalculate_coords:
                datapack.delete_solset("X_facets")
                datapack.delete_solset("X_screen")

            if not datapack.is_solset('X_facets') or not datapack.is_solset("X_screen"):
                datapack.switch_solset(solset)
                datapack.select(ant=None,time=None, dir=None, freq=None, pol=None)
                axes = datapack.__getattr__("axes_{}".format(soltab))
                
                antenna_labels, antennas = datapack.get_antennas(axes['ant'])
                patch_names, directions = datapack.get_sources(axes['dir'])
                timestamps, times = datapack.get_times(axes['time'])
                freq_labels, freqs = datapack.get_freqs(axes['freq'])
                pol_labels, pols = datapack.get_pols(axes['pol'])

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
                    X[:,:,j:j+1,:],X_screen[:,:,j:j+1,:] = _parallel_coord_transform(datapack.array_center, time, times[0].mjd*86400., directions, screen_directions, antennas)
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

                self.datapack.switch_solset("screen_sol", 
                        array_file = DataPack.lofar_array, 
                        directions = np.stack([screen_directions.ra.rad,screen_directions.dec.rad],axis=1))
                self.datapack.add_freq_indep_tab('tec', times.mjd*86400., pols = pol_labels)
                
                self.datapack.switch_solset("posterior_sol", 
                        array_file=DataPack.lofar_array, 
                        directions = np.stack([directions.ra.rad,directions.dec.rad],axis=1))
                self.datapack.add_freq_indep_tab('tec', times.mjd*86400., pols = pol_labels)

                datapack.switch_solset(solset)

    def _posterior_coords(self, datapack, screen=False, **kwargs):
        """
        generator that yeilds batchs of X coords to do posterior prediction at.
        screen: bool if True return screen coords
        """
        raise NotImplementedError("Must subclass")

    def _save_posterior(self, ystar, varstar, datapack, data_shape, screen=False, **kwargs):
        """
        Save the results in the datapack in the appropriate slots
        screen: bool if True save in the screen slot
        """
        raise NotImplementedError("Must subclass")

    def _predict_posterior(self, X, **kwargs):
        """
        Predict the model at the coords.
        X: array of [batch, ndim]
        returns:
        ystar [batch, P] predictive mean at coords
        varstar [batch, P] predictive variance at the coords
        """
        raise NotImplementedError("Must subclass")

    def _train_model(self, model, save_folder, **kwargs):
        """
        Train the model.
        Returns the saved model.
        """
        raise NotImplementedError("Must subclass")

    def _load_model(self, model):
        """
        Load a model given by model path
        """
        raise NotImplementedError("Must subclass")


    def _build_model(self, X, Y, weights=None, **kwargs):
        """
        Build the model from the data.
        X,Y: tensors the X and Y of data
        weights: statistical weights that may be None

        Returns:
        gpflow.models.Model
        """
        raise NotImplementedError("Must subclass")

    def _new_session(self, graph, intra_op_threads=0, inter_op_threads=0, debug=False, **kwargs):
        os.environ["KMP_BLOCKTIME"] = "1"
        os.environ["KMP_SETTINGS"] = "1"
        os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"
        if intra_op_threads > 0:
            os.environ["OMP_NUM_THREADS"] = str(intra_op_threads)
        config = tf.ConfigProto()
        config.intra_op_parallelism_threads = intra_op_threads
        config.inter_op_parallelism_threads = inter_op_threads
        sess = tf.Session(graph=graph,config=config)
        if debug:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)            
        return sess

    def _prepare_data(self,datapack,**kwargs):
        """
        Prepares the data in the datapack for solving.
        Likely steps:
        1. Determine measurement variance
        2. Calculate data coordinates (new sol-tab is made inside the solset)
        3. Make new hdf5 file linking to X,Y weights
        datapack : DataPack solutions to solve
        Returns:
        data_shape
        """
        raise NotImplementedError("Must subclass")

    def _get_data(self,indices,data_shape, dtype=settings.np_float):
        """
        Return a selection of (X,Y,weights/var)
        indices : array of indices that index into data 
        data_shape : tuple of dim sizes for index raveling

        Returns:
        X array tf.float64 [ndims]
        Y array tf.float64 [P]
        weights array tf.float64 [P]
        """
        raise NotImplementedError("Must subclass")

    def _train_dataset_iterator(self, data_shape,  sess=None, minibatch_size=128,seed=0, **kwargs):
        """
        Create a dataset iterator.
        Produce synchronized minibatches, and initializes if given a session.
        Will use _get_data asynchronously to pull data as needed.
        data_shape: tuple of size of data axes e.g. (Nd, Na, Nt)
        Returns:
        TF op to init dataset iterator
        tuple of synchrionized next data tensors (X, Y, weights)
        """

        def _random_coords(n):
            return tf.stack([tf.random_uniform((minibatch_size,),0,s,dtype=tf.int64) for s in data_shape],axis=1)

        minibatches = tf.constant([1])
        data = tf.data.Dataset.from_tensor_slices([minibatches])
        data = data.repeat()#repeat and batch forever
        data = data.map(_random_coords)#indices compatible with data_shape
        data = data.map(lambda indices: \
                tuple(tf.py_func(lambda indices: self._get_data(indices,data_shape),[indices], [settings.float_type]*3)))#X, Y, W

        iterator_tensor = data.make_initializable_iterator()
        if sess is not None:
            sess.run(iterator_tensor.initializer)
        return iterator_tensor.initializer, iterator_tensor.get_next()
