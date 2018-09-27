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


class Solver(object):
    def __init__(self,run_dir, datapack):
        run_dir = os.path.abspath(run_dir)
        self.run_id = len(glob.glob(os.path.join(run_dir,"run_*")))
        self.run_dir = os.path.join(run_dir,"run_{:03d}".format(self.run_id))
        self.summary_dir = os.path.join(self.run_dir,"summaries")
        self.save_dir = os.path.join(self.run_dir, "checkpoints")
        self.plot_dir = os.path.join(self.run_dir, "plots")
        os.makedirs(self.run_dir,exist_ok=True)
        os.makedirs(self.summary_dir,exist_ok=True)
        os.makedirs(self.save_dir,exist_ok=True)
        os.makedirs(self.plot_dir,exist_ok=True)
        if isinstance(datapack,str):
            datapack = DataPack(datapack)
        self.datapack = datapack

    def solve(self, output_solset='posterior_sol', load_model=None, **kwargs):
        """Run the solver"""
        logging.info("Starting solve")
        self.output_solset = output_solset
        self.output_screen_solset = "screen_{}".format(output_solset)
        
        logging.info("Preparing data...")
        data_shape, build_params = self._prepare_data(self.datapack, **kwargs)
        
        graph = tf.Graph()
        sess = self._new_session(graph, **kwargs)

        summary_id = len(glob.glob(os.path.join(self.summary_dir,"summary_*")))
        summary_folder = os.path.join(self.summary_dir,"summary_{:03d}".format(summary_id))
        os.makedirs(summary_folder,exist_ok=True)
        with graph.as_default(), sess.as_default(), \
                tf.summary.FileWriter(summary_folder, graph) as writer:
            logging.info("Constructing dataset iterators")
            _, data_tensors = self._train_dataset_iterator(data_shape, sess=sess, **kwargs)
            logging.info("Building model")
            model = self._build_model(*data_tensors, **build_params, **kwargs)
            if load_model is not None:
                logging.info("Loading previous model state")
                self._load_model(model, load_model)
            else:
                logging.info("No model to load -> train from scratch")
            # train model
            save_id = len(glob.glob(os.path.join(self.save_dir,"save_*")))
            save_folder = os.path.join(self.save_dir,"save_{:03d}".format(save_id))
            os.makedirs(save_folder,exist_ok=True)
            logging.info("Starting model training")
            saved_model = self._train_model(model, save_folder, writer, **kwargs)
            
            # TODO break prediction into another call (i.e. not in solve)
            logging.info("Loading trained model")
            self._load_model(model, saved_model)
            
            logging.info("Predicting posterior at data coords")
            ystar, varstar = [], []
            for X in self._posterior_coords(self.datapack, screen=False, **kwargs):
                _ystar, _varstar = self._predict_posterior(model, X)
                ystar.append(_ystar)
                varstar.append(_varstar)
            ystar = np.concatenate(ystar, axis=0)
            varstar = np.concatenate(varstar, axis=0)
            logging.info("Saving posterior at data coords")
            self._save_posterior(ystar, varstar, self.datapack, data_shape, screen=False, **kwargs)
            
            logging.info("Predicting posterior over screen")
            ystar, varstar = [], []
            for X in self._posterior_coords(self.datapack, screen=True, **kwargs):
                _ystar, _varstar = self._predict_posterior(model, X)
                ystar.append(_ystar)
                varstar.append(_varstar)
            ystar = np.concatenate(ystar, axis=0)
            varstar = np.concatenate(varstar, axis=0)
            logging.info("Saving posterior over screen")
            self._save_posterior(ystar, varstar, self.datapack, data_shape, screen=True, **kwargs)

            logging.info("Finalize routines")
            self._finalize(self.datapack, **kwargs)
            logging.info("Done")

    def _finalize(self, datapack, **kwargs):
        """
        Final things to run after a solve, such as plotting
        """
        raise NotImplementedError("Must subclass")

    
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
        tuple of synchrionized next data tensors (Y_var, freqs, X, Y) (order is how they get fed to model build)
        """

        def _random_coords(n):
            return tf.stack([tf.random_uniform((minibatch_size,),0,s,dtype=tf.int64) for s in data_shape],axis=1)

        minibatches = tf.constant([1])
        data = tf.data.Dataset.from_tensor_slices([minibatches])
        data = data.repeat()#repeat and batch forever
        data = data.map(_random_coords)#indices compatible with data_shape
        data = data.map(lambda indices: \
                tuple(tf.py_func(lambda indices: self._get_data(indices,data_shape),[indices], [settings.float_type]*4)))#Y_var, freqs, X, Y

        iterator_tensor = data.make_initializable_iterator()
        if sess is not None:
            sess.run(iterator_tensor.initializer)
        return iterator_tensor.initializer, iterator_tensor.get_next()
