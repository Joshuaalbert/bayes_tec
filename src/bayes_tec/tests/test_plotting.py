import matplotlib
matplotlib.use('agg')
from ..utils.testing_utils import make_example_datapack
from ..plotting.plot_datapack import DatapackPlotter, animate_datapack, plot_phase_vs_time,plot_data_vs_solution
import os

def test():
    # 2 timestamps
    datapack = make_example_datapack(45,4,2,name='plotting_test_datapack.hdf5')
#    dp = DatapackPlotter(datapack=datapack)
#    dp.plot(time=slice(0,10),labels_in_radec=True,show=True)

#    animate_datapack('plotting_test_datapack.hdf5',
#            'output_folder_test_plotting',num_processes=1,ant_sel="RS*",time_sel=slice(0,10,1),observable='phase',labels_in_radec=True,show=False)

    plot_data_vs_solution(datapack = '/net/lofar1/data1/albert/git/bayes_tec/scripts/data/killms_datapack.hdf5',
        output_folder = "./killms_1D_RS210",
        show_prior_uncert=False,
        ant_sel = 'RS210HBA',
        time_sel = slice(0,100,1),
        dir_sel = None,
        freq_sel = slice(0,1,1),
        pol_sel = None)
#    os.unlink('plotting_test_datapack.hdf5')
