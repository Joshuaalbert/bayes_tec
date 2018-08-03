from ..datapack import DataPack
from ..logging import logging
from .data_utils import make_coord_array
import numpy as np
import os
import astropy.time as at

def make_example_datapack(Nd,Nf,Nt,pols=None, time_corr=50.,dir_corr=0.5*np.pi/180.,tec_scale=0.02,tec_noise=1e-3,name='test.hdf5',clobber=False):
    logging.info("=== Creating example datapack ===")
    name = os.path.abspath(name)
    if os.path.isfile(name) and clobber:
        os.unlink(name)

    datapack = DataPack(name,readonly=False)
    with datapack:
        datapack.add_antennas()
        datapack.add_sources(np.random.normal(np.pi/4.,np.pi/180.*2.5,size=[Nd,2]))
        _, directions = datapack.sources
        _, antennas = datapack.antennas
        ref_dist = np.linalg.norm(antennas - antennas[0:1,:],axis=1)[None,None,:,None]#1,1,Na,1

        times = at.Time(np.linspace(0,Nt*8,Nt)[:,None],format='gps').mjd*86400.#mjs
        freqs = np.linspace(120,160,Nf)*1e6
        if pols is not None:
            use_pols = True
            assert isinstance(pols,(tuple,list))
        else:
            use_pols = False
            pols = ['XX']
            
        tec_conversion = -8.440e9/freqs #Nf
        
        X = make_coord_array(directions/dir_corr, times/time_corr)# Nd*Nt, 3
        X2 = np.sum((X[:,:,None] - X.T[None,:,:])**2, axis=1)#N,N
        K = tec_scale**2 * np.exp(-0.5*X2)
        L = np.linalg.cholesky(K + 1e-6*np.eye(K.shape[0]))#N,N
        Z = np.random.normal(size=(K.shape[0],len(pols)))#N,npols
        tec = np.einsum("ab,bc->ac",L,Z)#N,npols
        tec = tec.reshape((Nd,Nt,len(pols))).transpose((2,0,1))#Npols,Nd,Nt
        tec = tec[:,:,None,:]*(0.2+ref_dist/np.max(ref_dist))#Npols,Nd,Na,Nt
#         print(tec)
        tec += tec_noise*np.random.normal(size=tec.shape)
        phase = tec[:,:,:,None,:]*tec_conversion[None,None,None,:,None]##Npols,Nd,Na,Nf,Nt
#         print(phase)
        phase = np.angle(np.exp(1j*phase))
        
        if not use_pols:
            phase = phase[0,...]
            pols = None
        datapack.add_freq_dep_tab('phase',times=times,freqs=freqs,pols=pols,vals=phase)
        datapack.phase = phase
        return datapack

