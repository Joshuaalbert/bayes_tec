import matplotlib
matplotlib.use('agg')
from ..utils.testing_utils import make_example_datapack
from ..plotting.plot_datapack import DatapackPlotter, animate_datapack
import os

def test():
    # 2 timestamps
    datapack = make_example_datapack(45,4,2,name='plotting_test_datapack.hdf5')
#    dp = DatapackPlotter(datapack=datapack)
#    dp.plot(time=slice(0,10),labels_in_radec=True,show=True)

    animate_datapack('plotting_test_datapack.hdf5',
            'output_folder_test_plotting',num_processes=1,time_sel=slice(0,10,2),observable='phase',labels_in_radec=True,show=False)
    os.unlink('plotting_test_datapack.hdf5')



