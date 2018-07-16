from ..datapack import DataPack
from ..utils.data_utils import phase_weights, make_data_vec, make_coord_array, define_subsets
from ..frames import ENU
from scipy.cluster.vq import kmeans2
import numpy as np
import os
from gpflow import settings

class Solver(object):
    def __init__(self,run_dir, datapack):
        self.run_dir = os.path.abspath(run_dir)
        os.makedirs(self.run_dir,exist_ok=True)
        if isinstance(datapack,str):
            datapack = DataPack(datapack,readonly=True)
        self.datapack = datapack
    def run(self,*args, **kwargs):
        """Run the solver"""
        raise NotImplementedError("Must subclass")


class OverlapSolver(object):
    def __init__(self, tabs, overlap, run_dir, datapack):
        super(OverlapSolver,self).__init__(run_dir,datapack)
        self.overlap = float(overlap)# minimum overlap in seconds
        if not isinstance(tabs),(tuple,list)):
            tabs = [tabs]
        self.tabs = tabs # ['phase', 'amplitude', ...] etc.

    def _make_part_model(self, X, Y, M=None, minibatch_size=None, Z=None, eval_freq=140e6):
        """
        Create a gpflow model for a selection of data
        X: array (N, Din)
        Y: array (N, 2*Dout + 1)
            See ..utils.data_utils.make_datavec
        M: int, the number of inducing points (optional)
        minibatch_size : int 
        Z: array (M, Din)
            The inducing points if desired to set.
        eval_freq: float the freq in Hz where evaluation occurs.
        """
        pass

    def run(self, ant_sel=None, time_sel=None, dir_sel=None, freq_sel=None, pol_sel=None, 
            screen_res=30, jitter=1e-6, learning_rate=1e-3, iterations=10000, minibatch_size=128, 
            eval_freq=140e6, dof_ratio=35.):

        settings.numerics.jitter = jitter

        with self.datapack:
            self.datapack.select(ant=ant_sel,time=time_sel, dir=dir_sel, freq=freq_sel, pol=pol_sel)
            Y = []
            axes = None
            for tab in self.tabs:
                vals, axes = self.datapack.__getattr__(tab)
                Y.append(vals)

            antenna_labels, antennas = self.datapack.get_antennas(axes['ant'])
            patch_names, directions = self.datapack.get_sources(axes['dir'])
            timestamps, times = self.datapack.get_times(axes['time'])
            freq_labels, freqs = self.datapack.get_freqs(axes['freq'])
            pol_labels, pols = self.datapack.get_pols(axes['pol'])

            # Npols, Nd, Na, Nf, Nt, Ntabs
            Y = np.stack(Y,axis=-1)
            Npols, Nd, Na, Nf, Nt, Ntabs = Y.shape
            # Nd, Npol*Na*Ntabs, Nf, Nt
            Y = Y.transpose((1,0,2,5, 3,4)).reshape((Nd, Npol*Na*Ntabs, Nf, Nt))
            weights = phase_weights(Y,N=200,phase_wrap=True, min_uncert=1e-3)
            #Nd, Nt, Nf, Npol*Na*Ntabs
            Y = Y.transpose((0,3,2,1))
            weights = weights.transpose((0,3,2,1))
            num_latents = Npol*Na*Ntabs
            data_vec = make_data_vec(Y, freqs,weights=weights)
            
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
            X_d_ = np.array([m.flatten() for m in np.meshgrid(*([np.linspace(d_min,d_max, screen_res)]*2),indexing='ij')]).T
            Nd_ = screen_res**2
            directions_ = X_d_*d_std + d_mean
            self.datapack.switch_solset("screen_sol", 
                    array_file=DataPack.lofar_array, 
                    directions = directions_ * np.pi/180.)
        # data subsets
        edges = define_subsets(X_t, self.overlap / t_std)
        
        
        
        # N, 3
        X = make_coord_array(X_d, X_t, freqs[:,None])[:,:-1]
        M = int(np.ceil(



            
                

