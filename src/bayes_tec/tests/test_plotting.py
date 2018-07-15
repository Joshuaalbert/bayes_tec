from ..utils.testing_utils import make_example_datapack
from ..plotting.plot_datapack import DatapackPlotter

def test():
    datapack = make_example_datapack(45,4,100,name='plotting_test_datapack.hdf5')
    dp = DatapackPlotter(datapack=datapack)
    dp.plot(time=slice(0,10),labels_in_radec=True,show=True)

#    animate_datapack('../data/rvw_datapack_full_phase_dec27_smooth.hdf5',
#            'test_output',num_processes=1,observable='phase',labels_in_radec=True,show=True)



