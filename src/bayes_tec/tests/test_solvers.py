from ..utils.testing_utils import make_example_datapack
from ..solvers.solver import OverlapPhaseOnlySolver
import os

def test_overlap_solver():
    datapack = make_example_datapack(10,4,100,pols=['XX'], name='datapack_test_solver.hdf5')
    solver = OverlapPhaseOnlySolver(32., './run_dir_test_solver', datapack)
    solver.run(iterations=10)
    os.unlink('datapack_test_solver.hdf5')
