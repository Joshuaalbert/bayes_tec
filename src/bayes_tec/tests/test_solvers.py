from ..utils.testing_utils import make_example_datapack
from ..solvers.solver import OverlapPhaseOnlySolver
import os

def test_overlap_solver():
    datapack = make_example_datapack(45,5,10,pols=['XX'], name='datapack_test_solver.hdf5')
    solver = OverlapPhaseOnlySolver(160., './run_dir_test_solver', datapack)
    solver.run(iterations=50,kernels_shared=False,ant_sel="RS210HBA")
    os.unlink('datapack_test_solver.hdf5')
