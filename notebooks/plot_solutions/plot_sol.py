from bayes_tec.datapack import DataPack
from bayes_tec.plotting.plot_datapack import animate_datapack
import argparse

def run(datapack,output_folder, num_processes,antenna):
    animate_datapack(datapack,output_folder, num_processes, labels_in_radec=True, plot_crosses=False, ant=antenna)



def add_args(parser):
    parser.register("type", "bool", lambda v: v.lower() == "true")

    # network
    parser.add_argument("--antenna", type=str, default="*", 
                        help="""The antennas selection using regex.\n{}""")
    parser.add_argument("--datapack", type=str,
                       help="""Datapack input, a losoto h5parm.""")
    parser.add_argument("--output_folder", type=str, default='./figs',
            help="""The output folder for figures""")
    parser.add_argument('--num_processes', type=int, default=1,
            help="""The number of plotting processes""")

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    add_args(parser)
    flags, unparsed = parser.parse_known_args()
    run(**vars(flags))







            
                

