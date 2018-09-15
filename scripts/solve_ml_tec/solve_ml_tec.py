#!/usr/bin/env python
from bayes_tec.logging import logging
import argparse
from bayes_tec.datapack import DataPack
from bayes_tec.bayes_opt.maximum_likelihood_tec import solve_ml_tec
import numpy as np

def run_solve(flags):

    with DataPack(flags.datapack) as datapack:
        datapack.switch_solset(flags.solset)
        datapack.select(ant=flags.ant_sel, dir=flags.dir_sel, pol=flags.pol_sel, time=flags.time_sel, freq=flags.freq_sel)
        phase,axes = datapack.phase
        timestamps, times = datapack.get_times(axes['time'])
        pol_labels, pols = datapack.get_pols(axes['pol'])
        _, freqs = datapack.get_freqs(axes['freq'])
        Npol, Nd, Na, Nf, Nt = phase.shape
        phase = phase.transpose((0,1,2,4,3))
        phase = phase.reshape((-1, Nf))
        tec_ml, sigma_ml = solve_ml_tec(phase, freqs, batch_size=flags.batch_size,max_tec=flags.max_tec, n_iter=flags.n_iter, t=flags.t,num_proposal=flags.num_proposal, verbose=True)
        datapack.add_freq_indep_tab('tec', times.mjd*86400., pols = pol_labels)
        tec_ml = tec_ml.reshape((Npol, Nd, Na, Nt))
        sigma_ml = sigma_ml.reshape((Npol, Nd, Na, Nt))
        datapack.tec = tec_ml
        datapack.weights_tec = np.where(sigma_ml > 0., 1./np.square(sigma_ml), 0.)#weights are 1/var

    
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
    sim_args = parser.add_argument_group('simulation arguments')



    # remove this line: optional = parser...

    required.add_argument("--datapack", type=str, 
            help="Input datapack")
    optional.add_argument("--solset", type=str, default='sol000',
            help="The solset with input phase, solutions will go into tec tab and phase/weights")

    optional.add_argument("--batch_size", type=int, default=10000,
                       help="""The batch size of solve.""")
    optional.add_argument("--max_tec", type=float, default=0.3,
                       help="""Max TEC abs TEC scale > 0.""")
    optional.add_argument("--n_iter", type=int, default=23, 
                      help="How many iterations to run")
    optional.add_argument("--num_proposal", type=int, default=100, 
                      help="How many proposals per iteration")
    optional.add_argument("--t", type=float, default=1.,
                       help="""Exploration parameter > 0, higher is more exploratory.""")
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
 



if __name__=='__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_args(parser)
    flags, unparsed = parser.parse_known_args()
    run_solve(flags)







            
                

