#!/usr/bin/env python

from bayes_tec.solvers.phase_only_lmc_solver import LMCPhaseOnlySolver
from bayes_tec.utils.testing_utils import make_example_datapack
from bayes_tec.logging import logging
import numpy as np
import argparse
import os

def run_solve(flags):

    output_folder = os.path.abspath(flags.output_dir)
    os.makedirs(output_folder, exist_ok=True)
    filename = os.path.join(output_folder, 
            "datapack_{:.1e},_{:.1e}_{:.1e}_{:.1e}.hdf5".format(flags.sim_tec_scale, flags.sim_tec_noise, flags.sim_time_corr, flags.sim_dir_corr))
    datapack = make_example_datapack(flags.sim_Nd,flags.sim_Nf,flags.sim_Nt,pols=['XX'], time_corr=flags.sim_time_corr,dir_corr=flags.sim_dir_corr*np.pi/180.,tec_scale=flags.sim_tec_scale,tec_noise=flags.sim_tec_noise,name=filename,clobber=flags.sim_clobber)
    logging.info(datapack)
    solver = LMCPhaseOnlySolver(output_folder, datapack)
    solver.run(**vars(flags))

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

    required.add_argument("--output_dir", type=str,
            help="""Output directory.""", required=True)

    sim_args.add_argument("--sim_tec_scale", type=float,default=0.01,
            help="""Scale of TEC deviations.""")
    sim_args.add_argument("--sim_tec_noise", type=float,default=1e-3,
            help="""TEC measurement noise.""")
    sim_args.add_argument("--sim_time_corr", type=float,default=50,
            help="""Time scale in seconds on which TEC changes.""")
    sim_args.add_argument("--sim_dir_corr", type=float,default=0.5,
            help="""Directional scale in degrees on which TEC changes.""")
    sim_args.add_argument("--sim_clobber", type="bool", default=False, 
            help="""Whether to overwrite results.\n""")

    sim_args.add_argument("--sim_Nd", type=int, default=10, 
            help="""Number of directions randomly scattered in 2.5deg FOV.\n""")
    sim_args.add_argument("--sim_Nt", type=int, default=100, 
            help="""Number of time stamps of 8 seconds each.\n""")
    sim_args.add_argument("--sim_Nf", type=int, default=10, 
            help="""Number of freqs between 120e6 and 160e6 Hz.\n""")

    optional.add_argument("--dof_ratio", type=float, default=40.,
                       help="""The ratio of temporal-spatial coordinates to degrees of freedom.""")
    optional.add_argument("--minibatch_size", type=int, default=256,
                      help="Size of minibatch")
    optional.add_argument("--learning_rate", type=float, default=1e-3,
                      help="learning rate")
    optional.add_argument("--plot", type="bool", default=True, const=True,nargs='?',
                      help="Whether to plot results")
    optional.add_argument("--run_dir", type=str, default='./run_dir', 
                      help="Where to run the solve")
    optional.add_argument("--iterations", type=int, default=10000, 
                      help="How many iterations to run")
    optional.add_argument("--jitter", type=float, default=1e-6, 
                      help="Jitter for stability")
    optional.add_argument("--eval_freq", type=float, default=144e6, 
                      help="Eval frequency")
    optional.add_argument("--reweight_obs", type="bool", default=True, 
                      help="Whether to re-calculate the weights down the frequency axis. Otherwise use /weight table.")
    optional.add_argument("--inter_op_threads", type=int, default=0,
                       help="""The max number of concurrent threads""")
    optional.add_argument("--intra_op_threads", type=int, default=0,
                       help="""The number threads allowed for multi-threaded ops.""")
    optional.add_argument("--tec_scale", type=float, default=0.01,
                       help="""The relative tec scale used for scaling the GP model for computational stability.""")
    optional.add_argument("--max_block_size", type=int, default=500,
                       help="""Maximum number of timestamps per block solve.""")
    required.add_argument("--overlap", type=float, default=160.,
            help="""Temporal overlap in seconds.""",required=True)
    optional.add_argument("--time_skip", type=int, default=2,
                      help="Time skip")

    



if __name__=='__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_args(parser)
    flags, unparsed = parser.parse_known_args()
    run_solve(flags)







            
                

