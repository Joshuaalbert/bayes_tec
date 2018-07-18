from ..datapack import DataPack
from ..logging import logging
import numpy as np

def make_example_datapack(Nd,Nf,Nt,pols=None, time_corr=50.,tec_scale=0.01,name='test.hdf5'):
    logging.info("=== Creating example datapack ===")

    datapack = DataPack(name)
    with datapack:
        datapack.add_antennas()
        datapack.add_sources(np.random.normal(np.pi/4.,np.pi/180.*2.5,size=[Nd,2]))

        times = np.linspace(-Nt*4,Nt*4,Nt)
        freqs = np.linspace(120,160,Nf)*1e6
        if pols is not None:
            use_pols = True
            assert isinstance(pols,(tuple,list))
        else:
            use_pols = False
            pols = ['XX']

        tec_conversion = -8.440e9/freqs
        x = (times[:,None] - times[None,:])/time_corr
        x2 = x*x
        K = tec_scale**2 * np.exp(-0.5*x2)
        L = np.linalg.cholesky(K+1e-6*np.eye(Nt))
        phase = []
        for d in range(Nd):
            Z = np.random.normal(size=(Nt,len(pols)))
            # Nt, Npols
            tec = np.einsum("ab,bc->ac",L,Z)
            # Nf,Nt,Npols
            phase.append(tec[None,:,:] * tec_conversion[:,None,None])

        # Nd, Nf, Nt, Npols
        phase = np.stack(phase,axis=0)
        # Nd, Na, Nf, Nt, Npols
        phase = np.tile(phase[:,None,:,:,:],(1,len(datapack.antennas[0]), 1, 1, 1) )
        phase += 0.3*np.random.normal(size=phase.shape)

        phase = phase.transpose((4,0,1,2,3))
        if not use_pols:
            phase = phase[0,...]
            pols = None
        
        datapack.add_freq_dep_tab('phase',times=times,freqs=freqs,pols=pols,vals=phase)

        return datapack


