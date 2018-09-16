#!/usr/bin/env python
from bayes_tec.datapack import DataPack
from bayes_tec.plotting.plot_datapack import animate_datapack
from bayes_tec.logging import logging
import argparse

def run_plot(datapack, output_folder, num_processes, **kwargs):
    animate_datapack(datapack,output_folder, num_processes, **kwargs)

def add_args(parser):
    def _time_sel(s):
        logging.info("Parsing {}".format(s))
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
        logging.info("Parsing {}".format(s))
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
        logging.info("Parsing {}".format(s))
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
        logging.info("Parsing {}".format(s))
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
        logging.info("Parsing {}".format(s))
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

    required.add_argument("--datapack", type=str,
            help="""Datapack input, a losoto h5parm.""", required=True)

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

    optional.add_argument("--plot_crosses", type="bool", default=True,
                      help="Plot crosses in facets")
    optional.add_argument("--labels_in_radec", type="bool", default=True,
                      help="Labels in RA/DEC")
    optional.add_argument("--plot_screen", type="bool", default=False,
                      help="Whether to plot screen. Expects properly shaped array.")


    optional.add_argument("--num_processes", type=int, default=1,
                      help="Number of parallel plots")
    optional.add_argument("--output_folder", type=str, default="./figs",
                       help="""The output folder.""")
    optional.add_argument("--observable", type=str, default="phase",
                       help="""The soltab to plot""")
    optional.add_argument("--phase_wrap", type="bool", default=True,
                       help="""Whether to wrap the observable""")
    optional.add_argument("--solset", type=str, default="sol000",
                       help="""The solset to plot""")

    optional.add_argument("--vmin", type=float, default=None,
                       help="""The min value if phase_wrap is False""")
    optional.add_argument("--vmax", type=float, default=None,
                       help="""The max value if phase_wrap is False""")


if __name__=='__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_args(parser)
    flags, unparsed = parser.parse_known_args()
    logging.info(vars(flags))
    run_plot(**vars(flags))

