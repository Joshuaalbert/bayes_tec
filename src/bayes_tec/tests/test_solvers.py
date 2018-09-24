from ..utils.testing_utils import make_example_datapack
from ..solvers.phase_only_solver import PhaseOnlySolver
from ..solvers.block_solver import BlockSolver
import os


def test_new_solver():

    run_dir = "run_dir_killms_10_Wdiag"
    datapack = '/net/lofar1/data1/albert/git/bayes_tec/scripts/data/killms_datapack_3.hdf5'
    solver = PhaseOnlySolver(run_dir, datapack)
    solver.solve(solset='sol000', recalculate_coords=False, jitter=1e-6, tec_scale=0.005, screen_res=30, weight_smooth_len=40, reweight_obs=False, 
            learning_rate=1e-2,iterations=2000, minibatch_size=128, dof_ratio=10.,intra_op_threads=0, inter_op_threads=0, ant_sel=slice(51,55,1),
            time_sel=slice(0,100,1),pol_sel=slice(0,1,1),debug=False, W_diag=True)

    run_dir = "run_dir_killms_10_chol"
    datapack = '/net/lofar1/data1/albert/git/bayes_tec/scripts/data/killms_datapack_3.hdf5'
    solver = PhaseOnlySolver(run_dir, datapack)
    solver.solve(solset='sol000', recalculate_coords=False, jitter=1e-6, tec_scale=0.005, screen_res=30, weight_smooth_len=40, reweight_obs=False, 
            learning_rate=1e-2,iterations=2000, minibatch_size=128, dof_ratio=10.,intra_op_threads=0, inter_op_threads=0, ant_sel=slice(51,55,1),
            time_sel=slice(0,100,1),pol_sel=slice(0,1,1),debug=False, W_diag=False)

    run_dir = "run_dir_ndppp_10_Wdiag"
    datapack = '/net/lofar1/data1/albert/git/bayes_tec/scripts/data/ndppp_datapack.hdf5'
    solver = PhaseOnlySolver(run_dir, datapack)
    solver.solve(solset='sol000', recalculate_coords=False, jitter=1e-6, tec_scale=0.005, screen_res=30, weight_smooth_len=40, reweight_obs=False, 
            learning_rate=1e-2,iterations=2000, minibatch_size=128, dof_ratio=10.,intra_op_threads=0, inter_op_threads=0, ant_sel=slice(51,55,1),
            time_sel=slice(0,100,1),pol_sel=slice(0,1,1),debug=False, W_diag=True)

    run_dir = "run_dir_ndppp_10_chol"
    datapack = '/net/lofar1/data1/albert/git/bayes_tec/scripts/data/killms_datapack.hdf5'
    solver = PhaseOnlySolver(run_dir, datapack)
    solver.solve(solset='sol000', recalculate_coords=False, jitter=1e-6, tec_scale=0.005, screen_res=30, weight_smooth_len=40, reweight_obs=False, 
            learning_rate=1e-2,iterations=2000, minibatch_size=128, dof_ratio=10.,intra_op_threads=0, inter_op_threads=0, ant_sel=slice(51,55,1),
            time_sel=slice(0,100,1),pol_sel=slice(0,1,1),debug=False, W_diag=False)

if __name__=='__main__':
    test_new_solver()
