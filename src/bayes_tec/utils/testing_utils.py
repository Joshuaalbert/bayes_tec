from ..datapack import DataPack
from ..logging import logging
import numpy as np

def make_example_datapack(Nd,Nf,Nt,time_corr=50.,tec_scale=0.01,name='test.hdf5'):
    datapack = DataPack(name)
    with datapack:
        datapack.add_antennas()

        datapack.add_sources(np.random.normal(np.pi/4.,np.pi/180.*2.5,size=[Nd,2]))
        times = np.linspace(-Nt*4,Nt*4,Nt)
        freqs = np.linspace(120,160,Nf)*1e6

        tec_conversion = -8.4e9/freqs
        x = (times[:,None] - times[None,:])/time_corr
        x2 = x*x
        K = tec_scale**2 * np.exp(-0.5*x2)
        L = np.linalg.cholesky(K+1e-6*np.eye(Nt))
        phase = []
        for d in range(Nd):
            Z = np.random.normal(size=(Nt,))
            # Nt
            tec = np.einsum("ab,b->a",L,Z)
            # Nf,Nt
            phase.append(tec[None,:] * tec_conversion[:,None])

        # Nd, Nf, Nt
        phase = np.stack(phase,axis=0)
        phase = np.tile(phase[:,None,:,:],(1,len(datapack.antennas[0]), 1, 1) )
        phase += 0.3*np.random.normal(size=phase.shape)
        print(phase.shape)
        
        datapack.add_freq_dep_tab('phase',times=times,freqs=freqs,vals=phase)
        logging.warning(str(datapack.H))

        return datapack


