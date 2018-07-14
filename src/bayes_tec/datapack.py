
from .logging import logging
from losoto.h5parm import h5parm
import os
import numpy as np
import astropy.units as au
import astropy.time as at
import astropy.coordinates as ac
import sys

class DataPack(object):
    """
    We use losoto.h5parm as the data holder.
    """
    _arrays = os.path.dirname(sys.modules["bayes_tec"].__file__)
    lofar_array = os.path.join(_arrays,'arrays/lofar.hba.antenna.cfg')
    lofar_cycle0_array = os.path.join(_arrays,'arrays/lofar.cycle0.hba.antenna.cfg')
    gmrt_array = os.path.join(_arrays,'arrays/gmrtPos.csv')

    def __init__(self,filename,readonly=False,solset='sol000'):
        self.filename = os.path.abspath(filename)
        self.readonly=readonly
        self.solset = solset
        # create if required
        H = h5parm(self.filename, readonly=False)
        if self.solset not in H.getSolsetNames():
            #adds the antenna, directions tables
            H.makeSolset(solsetName=self.solset,addTables=True)
            logging.warning("Created {}".format(str(H)))
        H.close()
        self.H = None
        self._contexts_open = 0

    def __enter__(self):
        if self._contexts_open == 0:
            self.H = h5parm(self.filename, readonly=self.readonly)
        self._contexts_open += 1
        return self.H

    def __exit__(self,exc_type, exc_val, exc_tb):
        if self.H:
            if self._contexts_open == 1:
                self.H.close()
                self.H = None
            self._contexts_open -= 1

    @property
    def _solset(self):
        with self:
            return self.H.getSolset(self.solset)

    def _load_array_file(self,array_file):
        '''Loads a csv where each row is x,y,z in geocentric ITRS coords of the antennas'''
        
        try:
            types = np.dtype({'names':['X','Y','Z','diameter','station_label'],
                             'formats':[np.double,np.double,np.double,np.double,'S16']})
            d = np.genfromtxt(array_file,comments = '#',dtype=types)
            diameters = d['diameter']
            labels = np.array(d['station_label'].astype(str))
            locs = ac.SkyCoord(x=d['X']*au.m,y=d['Y']*au.m,z=d['Z']*au.m,frame='itrs')
            Nantenna = int(np.size(d['X']))
        except:
            d = np.genfromtxt(array_file,comments = '#',usecols=(0,1,2))
            locs = ac.SkyCoord(x=d[:,0]*au.m,y=d[:,1]*au.m,z=d[:,2]*au.m,frame='itrs')
            Nantenna = d.shape[0]
            labels = np.array([b"ant{:02d}".format(i) for i in range(self.Nantenna)])
            diameters = None
        return np.array(labels).astype(np.str_), locs.cartesian.xyz.to(au.m).value.transpose()

    def save_array_file(self,array_file):
        import time
        with self:
            ants = self._solset.getAnt()
            labels = []
            locs = []
            for label, pos in ants.items():
                labels.append(label)
                locs.append(pos)
            Na = len(labels)
        with open(array_file,'w') as f:
            f.write('# Created on {0} by Joshua G. Albert\n'.format(time.strftime("%a %c",time.localtime())))
            f.write('# ITRS(m)\n')
            f.write('# X\tY\tZ\tlabels\n')
            i = 0
            while i < Na:
                f.write('{0:1.9e}\t{1:1.9e}\t{2:1.9e}\t{3:d}\t{4}'.format(locs[i][0],locs[i][1],locs[i][2],labels[i]))
                if i < Na-1:
                    f.write('\n')
                i += 1

    def add_antennas(self, array_file = None):
        if array_file is None:
            array_file = self.lofar_array
        labels, pos = self._load_array_file(array_file)
        antennaTable = self._solset.obj._f_get_child('antenna')
        for lab,p in zip(labels,pos):
            if lab not in antennaTable.cols.name[:].astype(type(lab)):
                antennaTable.append([(lab,p)])

    def add_sources(self, directions, patch_names=None):
        Nd = len(directions)
        if patch_names is None:
            patch_names = []
            for d in range(Nd):
                patch_names.append("patch_{:03d}".format(d))

        sourceTable = self._solset.obj._f_get_child('source')
        for lab,p in zip(patch_names,directions):
            if lab not in sourceTable.cols.name[:].astype(type(lab)):
                sourceTable.append([(lab,p)])

    @property
    def _antennas(self):
        with self:
            return self._solset.obj.antenna
        
    @property
    def antennas(self):
        antenna_labels, pos = [],[]
        for a in self._antennas:
            antenna_labels.append(a['name'])
            pos.append(a['position'])
        return antenna_labels, pos

    @property
    def _sources(self):
        with self:
            return self._solset.obj.source
        
    @property
    def sources(self):
        patch_names = []
        dirs = []
        for s in self._sources:
            patch_names.append(s['name'])
            dirs.append(s['dir'])
        return patch_names, dirs
    
    def add_freq_indep_tab(self, name, times, pols = None, ants = None, dirs = None, vals=None):
        with self:
            #pols = ['XX','XY','YX','YY']
            if dirs is None:
                dirs,_ = self.sources
            if ants is None:
                ants,_ = self.antennas
            if pols is not None:
                Npol = len(pols)
            Nd = len(dirs)
            Na = len(ants)
            Nt = len(times)
            if pols is not None:
                if vals is None:
                    phase = np.zeros([Npol,Nd,Na,Nt])
                self._solset.makeSoltab(name, axesNames=['pol','dir','ant','time'],
                        axesVals=[pols, dirs, ants, times],vals=vals, weights=np.ones_like(vals))
            else:
                if vals is None:
                    phase = np.zeros([Nd,Na,Nt])
                self._solset.makeSoltab(name, axesNames=['dir','ant','time'],
                        axesVals=[dirs, ants, times],vals=vals, weights=np.ones_like(vals))
    

    def add_freq_dep_tab(self, name, times, freqs, pols = None, ants = None, dirs = None, vals=None):
        with self:
            #pols = ['XX','XY','YX','YY']
            if dirs is None:
                dirs,_ = self.sources
            if ants is None:
                ants,_ = self.antennas
            if pols is not None:
                Npol = len(pols)
            Nd = len(dirs)
            Na = len(ants)
            Nt = len(times)
            Nf = len(freqs)
            if pols is not None:
                if vals is None:
                    vals = np.zeros([Npol,Nd,Na,Nf,Nt])
                self._solset.makeSoltab(name, axesNames=['pol','dir','ant','freq','time'],
                        axesVals=[pols, dirs, ants, freqs, times],vals=vals, weights=np.ones_like(vals))
            else:
                if vals is None:
                    vals = np.zeros([Nd,Na,Nf,Nt])
                self._solset.makeSoltab(name, axesNames=['dir','ant','freq','time'],
                        axesVals=[dirs, ants, freqs, times],vals=vals, weights=np.ones_like(vals))
    
    def __getattr__(self, tab):
        """
        Links any attribute with an "axis name" to getValuesAxis("axis name")
        also links val and weight to the relative arrays.
        Parameter
        ----------
        axis : str
            The axis name.
        """
        if tab == 'phase':
            with self:
                return self._solset.getSoltab('phase000')
        elif tab == 'amplitude':
            with self:
                return self._solset.getSoltab('amplitude000')
        elif tab == 'tec':
            with self:
                return self._solset.getSoltab('tec000')
        elif tab == 'variance_phase':
            with self:
                return self._solset.getSoltab('variance_phase000')
        elif tab == 'variance_amplitude':
            with self:
                return self._solset.getSoltab('variance_amplitude000')
        elif tab == 'variance_tec':
            with self:
                return self._solset.getSoltab('variance_tec000')
        else:
            return object.__getattribute__(self, tab)
