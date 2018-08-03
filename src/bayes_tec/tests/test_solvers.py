from ..utils.testing_utils import make_example_datapack
from ..solvers.solver import OverlapPhaseOnlySolver
from ..solvers.phase_only_lmc_solver import LMCPhaseOnlySolver
import os

#def test_overlap_solver():
#    datapack = make_example_datapack(45,5,10,pols=['XX'], name='datapack_test_solver.hdf5')
#    solver = OverlapPhaseOnlySolver('./run_dir_test_solver', datapack)
#    solver.run(iterations=50,kernels_shared=False,ant_sel="RS210HBA")
#    os.unlink('datapack_test_solver.hdf5')

def test_lmc_solver():
    datapack = make_example_datapack(45,10,200,pols=['XX'], name='datapack_test_solver.hdf5')
    solver = LMCPhaseOnlySolver('./run_dir_test_solver', datapack)
    solver.run(iterations=5000,ant_sel="RS210HBA", learning_rate=1e-2, overlap=32., max_block_size=200)
    os.unlink('datapack_test_solver.hdf5')
