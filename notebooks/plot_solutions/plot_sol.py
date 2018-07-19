from bayes_tec.datapack import DataPack
from bayes_tec.plotting.plot_datapack import animate_datapack
import argparse

def run_plot(datapack, output_folder, num_processes, ant_sel, time_sel, freq_sel, pol_sel, dir_sel):
    animate_datapack(datapack,output_folder, num_processes, labels_in_radec=True, plot_crosses=False, ant=ant_sel, dir=dir_sel, freq=freq_sel, pol=pol_sel)

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

    optional.add_argument("--num_processes", type=int, default=1,
                      help="Number of parallel plots")
    optional.add_argument("--output_folder", type=str, default="./figs",
                       help="""The output folder.""")

if __name__=='__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_args(parser)
    flags, unparsed = parser.parse_known_args()
    run_plot(**vars(flags))

