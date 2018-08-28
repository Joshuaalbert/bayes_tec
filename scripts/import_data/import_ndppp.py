#!/usr/bin/env python

import argparse
from bayes_tec.datapack import DataPack
import h5py
import numpy as np
import os
import sys
from bayes_tec.logging import logging

#TECU = 1e16
tec_conversion = -8.4480e9# rad Hz/tecu

def _wrap(x):
    return np.angle(np.exp(1j*x))

def import_data(ndppp_dd_sols, out_datapack, clobber,ant_sel, time_sel, freq_sel, pol_sel, dir_sel):
    """Create a datapack from the direction dependent NDPPP solutions.
    
    """
    if os.path.exists(out_datapack):
        logging.info("{} exists".format(out_datapack))
        if clobber:
            logging.info("Deleting old datapack")
            os.unlink(out_datapack)
        else:
            raise ValueError("{} already exists and non clobber".format(out_datapack))
        
    with DataPack(ndppp_dd_sols,readonly=True) as f_dd:
        f_dd.select(ant=ant_sel,time=time_sel,freq=freq_sel,dir=dir_sel,pol=pol_sel)
        freqs = np.array([120.,130.,140.,150.,160.])*1e6

        with DataPack(out_datapack) as out:
            patch_names, directions = f_dd.sources
            antenna_labels, antennas = f_dd.antennas
            out.add_antennas()#default is lofar
            out.add_sources(directions, patch_names=patch_names)
            
            tec,axes = f_dd.tec#(npol), nt, na, nd,1
            scalarphase,axes = f_dd.scalarphase#(npol), nt, na, nd,1
            
            if 'pol' in axes.keys():#(1,3595,62,1,42,1)
                tec = tec[...,0].transpose((0,3,2,1))#npol,nd,na,nt
                scalarphase = scalarphase[...,0].transpose((0,3,2,1))#npol,nd,na,nt
                phase = tec_conversion*tec[:,:,:,None,:]/freqs[None,None,None,:,None] + scalarphase[:,:,:,None,:]
            else:
                tec = tec[...,0].transpose((2,1,0))#nd,na,nt
                scalarphase = scalarphase[...,0].transpose((2,1,0))#nd,na,nt
                phase = tec_conversion*tec[None,:,:,None,:]/freqs[None,None,None,:,None] + scalarphase[None,:,:,None,:]
                axes['pol'] = ['XX']
                
            out.add_freq_dep_tab('phase', axes['time'], freqs, pols=axes['pol'], ants = axes['ant'], 
                                 dirs = axes['dir'], vals=_wrap(phase))
    logging.info("Done importing data")

def add_args(parser):
    def _time_sel(s):
        if s.lower() == 'none':
            return None
        elif '/' in s:#slice
            s = s.split("/")
            assert len(s) == 3, "Proper slice notations is 'start/stop/step'"
            return slice(int(s[0]) if s[0].lower() != 'none' else None, 
                    int(s[1]) if s[1].lower() != 'none' else None, 
                    int(s[2])if s[2].lower() != 'none' else None)
        else:
            return s

    def _ant_sel(s):
        if s.lower() == 'none':
            return None
        elif '/' in s:#slice
            s = s.split("/")
            assert len(s) == 3, "Proper slice notations is 'start/stop/step'"
            return slice(int(s[0]) if s[0].lower() != 'none' else None, 
                    int(s[1]) if s[1].lower() != 'none' else None, 
                    int(s[2])if s[2].lower() != 'none' else None)
        else:
            return s

    def _dir_sel(s):
        if s.lower() == 'none':
            return None
        elif '/' in s:#slice
            s = s.split("/")
            assert len(s) == 3, "Proper slice notations is 'start/stop/step'"
            return slice(int(s[0]) if s[0].lower() != 'none' else None, 
                    int(s[1]) if s[1].lower() != 'none' else None, 
                    int(s[2])if s[2].lower() != 'none' else None)
        else:
            return s

    def _pol_sel(s):
        if s.lower() == 'none':
            return None
        elif ',' in s:
            s = s.split(',')
            return list(s)
        elif '/' in s:#slice
            s = s.split("/")
            assert len(s) == 3, "Proper slice notations is 'start/stop/step'"
            return slice(int(s[0]) if s[0].lower() != 'none' else None, 
                    int(s[1]) if s[1].lower() != 'none' else None, 
                    int(s[2])if s[2].lower() != 'none' else None)
        else:
            return s
    def _freq_sel(s):
        if s.lower() == 'none':
            return None
        elif '/' in s:#slice
            s = s.split("/")
            assert len(s) == 3, "Proper slice notations is 'start/stop/step'"
            return slice(int(s[0]) if s[0].lower() != 'none' else None, 
                    int(s[1]) if s[1].lower() != 'none' else None, 
                    int(s[2])if s[2].lower() != 'none' else None)
        else:
            return s


    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.register("type", "time_sel", _time_sel)
    parser.register("type", "ant_sel", _ant_sel)
    parser.register("type", "dir_sel", _dir_sel)
    parser.register("type", "pol_sel", _pol_sel)
    parser.register("type", "freq_sel", _freq_sel)

    optional = parser._action_groups.pop() # Edited this line
    parser._action_groups.append(optional) # added this line

    required = parser.add_argument_group('required arguments')



    # remove this line: optional = parser...

    required.add_argument("--ndppp_dd_sols", type=str,
            help="""NDPPP direction-dep. sols in a losoto h5parm.""", required=True)
    required.add_argument("--out_datapack", type=str,
            help="""The name of output datapack.""", required=True)


    # network
    optional.add_argument("--ant_sel", type="ant_sel", default=None, 
            help="""The antennas selection: None, regex RS*, or slice format <start>/<stop>/<step>.\n""")
    optional.add_argument("--time_sel", type="time_sel", default=None, 
                        help="""The antennas selection: None, or slice format <start>/<stop>/<step>.\n""")
    optional.add_argument("--dir_sel", type="dir_sel", default=None, 
                        help="""The direction selection: None, regex patch_???, or slice format <start>/<stop>/<step>.\n""")
    optional.add_argument("--pol_sel", type="pol_sel", default=None, 
                        help="""The polarization selection: None, list XX,XY,YX,YY, regex X?, or slice format <start>/<stop>/<step>.\n""")
    optional.add_argument("--freq_sel", type="freq_sel", default=None, 
                        help="""The channel selection: None, or slice format <start>/<stop>/<step>.\n""")
    optional.add_argument("--clobber", type="bool", default=False, 
                        help="""Whether to overwrite output datapack.\n""")
    
if __name__=='__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_args(parser)
    flags, unparsed = parser.parse_known_args()
    import_data(**vars(flags))
