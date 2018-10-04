from bayes_tec.solvers.phase_only_solver import PhaseOnlySolver
from bayes_tec.datapack import DataPack
import numpy as np
    
from bayes_tec.solvers.phase_only_solver import PhaseOnlySolver
from bayes_tec.datapack import DataPack
import numpy as np
import pylab as plt
    
def test_get_coords():
    
    datapack = DataPack('../../scripts/data/killms_datapack_3.hdf5', readonly=True)
    with datapack:
        datapack.select(time=slice(0,1000,1),
                        ant='RS210HBA',
                        pol=slice(0,1,1))

        phase, axes = datapack.phase

        _,times = datapack.get_times(axes['time'])
        _, directions = datapack.get_sources(axes['dir'])
        _, freqs = datapack.get_freqs(axes['freq'])
        Nt, Nd, Nf = len(times), len(directions), len(freqs)
        
        indices = np.array([np.random.randint(Nd,size=1000),
                    np.random.randint(Nf,size=1000),
                   np.random.randint(Nt,size=1000)]).T

        ra = directions.ra.deg[indices[:,0]]
        dec = directions.dec.deg[indices[:,0]]
        time = times.mjd[indices[:,2]]*86400. - times[0].mjd*86400.
        freq = freqs[indices[:,1]]
        
        phase = phase[0,indices[:,0],0,indices[:,1], indices[:,2]][...,None]
        
        
    
    
    solver = PhaseOnlySolver('run_dir_diagnostic', datapack)
    kwargs = {'ant_sel':"RS210HBA",
              'time_sel':slice(0,1000,1),
              'pol_sel':slice(0,1,1), 
              'reweight_obs':False, 
              'coord_file':"coords.hdf5",
              'minibatch_size':32, 
              'tec_scale':0.005}
    solver.output_solset = 'posterior_sol'
    solver.output_screen_solset = 'screen_sol'
    data_shape, build_params = solver._prepare_data(datapack,**kwargs)
    yv, f, x, y = solver._get_data(indices, [Nd, Nf, Nt])
    
    assert np.isclose(ra, x[:,0]).all()
    assert np.isclose(dec, x[:,1]).all()
    assert np.isclose(time, x[:,2]).all()
    assert np.isclose(phase, y).all()
    assert np.isclose(freq, f[:,0]).all()
    assert np.all(yv < 2*np.pi)
    
    
    
    
test_get_coords()
    
if __name__=='__main__':
    test_new_solver()
