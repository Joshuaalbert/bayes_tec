#!/usr/bin/env python
from bayes_tec.logging import logging
import argparse
from bayes_tec.datapack import DataPack
from bayes_tec.bayes_opt.maximum_likelihood_tec import solve_ml_tec
import numpy as np
from bayes_tec.utils.data_utils import calculate_weights

def run_solve(flags):

    with DataPack(flags.datapack) as datapack:
        datapack.switch_solset(flags.solset)
        datapack.select_all()
        axes = datapack.axes_phase
        timestamps, times = datapack.get_times(axes['time'])
        pol_labels, pols = datapack.get_pols(axes['pol'])
        datapack.delete_soltab('tec')
        datapack.add_freq_indep_tab('tec', times.mjd*86400., pols = pol_labels)

        datapack.select(ant=flags.ant_sel, dir=flags.dir_sel, pol=flags.pol_sel, time=flags.time_sel, freq=flags.freq_sel)


        phase_,axes = datapack.phase
        _, freqs = datapack.get_freqs(axes['freq'])

        Npol, Nd, Na, Nf, Nt = phase_.shape
        std = np.sqrt(0.5*calculate_weights(phase_,indep_axis=-2,N=len(freqs)//4,min_uncert=0.001) + 0.5*calculate_weights(phase_,indep_axis=-1,N=10,min_uncert=0.001))
        std = np.exp(np.median(np.log(std),axis=-2,keepdims=True))

        phase = phase_.transpose((0,1,2,4,3))
        phase = phase.reshape((-1, Nf))

        std = std.transpose((0,1,2,4,3))
        std = std.reshape((-1, 1))

    
    tec_ml, sigma_ml = solve_ml_tec(phase, freqs, batch_size=flags.batch_size,max_tec=flags.max_tec, n_iter=flags.n_iter, 
            t=flags.t,num_proposal=flags.num_proposal, lik_sigma = std, verbose=True)
    
    # fill in zeros
    tec_r = tec_ml.reshape((-1,Nt))
    t = times.mjd*86400. - times[0].mjd*86400.
    tec_in = []
    for i in range(tec_r.shape[0]):
        mask = tec_r[i,:] == 0.
        if not np.any(mask):
            tec_in.append(tec_r[i,:])
            continue
        nmask = np.bitwise_not(mask)
        if not np.any(nmask):
            tec_in.append(tec_r[i,:])
            continue
        tec_in.append(np.interp(t, t[nmask], tec_r[i,nmask]))
    tec_ml = np.stack(tec_in,axis=0).reshape((Npol,Nd,Na,Nt))


    ###
    # essentially removed noise, any wraps should be trivial to solve
    tec_ml = tec_ml.reshape((Npol, Nd, Na, Nt))
    # Npol, Nd, Na, Nf, Nt
    phase_pred = np.unwrap(tec_ml[...,None,:]*(-8.4480e9/freqs[:,None]), axis=-1)
    
    tec_post = phase_pred*(freqs[:,None]/-8.4480e9)
    # average out frequencies
    # Npol, Nd, Na ,Nt
    tec_mu = np.mean(tec_post,axis=-2)
    tec_var = np.var(tec_post,axis=-2)

    #Npol,Nd, Na, Nf, Nt
    def _wrap(phi):
        return np.angle(np.exp(1j*phi))
    phase_var = np.square(_wrap(_wrap(phase_)-_wrap(phase_pred)))

    
    with datapack:
        
        datapack.tec = tec_mu
        datapack.weights_tec = 1./tec_var
        datapack.weight_phase = 1./phase_var

    
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

    optional.add_argument("--batch_size", type=int, default=100000,
                       help="""The batch size of solve.""")
    optional.add_argument("--max_tec", type=float, default=0.25,
                       help="""Max TEC abs TEC scale > 0.""")
    optional.add_argument("--n_iter", type=int, default=25, 
                      help="How many iterations to run")
    optional.add_argument("--num_proposal", type=int, default=100, 
                      help="How many proposals per iteration")
    optional.add_argument("--t", type=float, default=0.05,
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







            
                

