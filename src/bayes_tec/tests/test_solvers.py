from ..utils.testing_utils import make_example_datapack
from ..solvers.phase_only_solver import PhaseOnlySolver
from ..solvers.block_solver import BlockSolver
import os

#def test_overlap_solver():
#    datapack = make_example_datapack(45,5,10,pols=['XX'], name='datapack_test_solver.hdf5')
#    solver = OverlapPhaseOnlySolver('./run_dir_test_solver', datapack)
#    solver.run(iterations=50,kernels_shared=False,ant_sel="RS210HBA")
#    os.unlink('datapack_test_solver.hdf5')

#def test_lmc_solver():
#    datapack = make_example_datapack(45,10,200,pols=['XX'], name='datapack_test_solver.hdf5')
#    solver = LMCPhaseOnlySolver('./run_dir_test_solver', datapack)
#    solver.run(iterations=5000,ant_sel="RS210HBA", learning_rate=1e-2, overlap=32., max_block_size=200)
#    os.unlink('datapack_test_solver.hdf5')

#def test_multi_seperable_solver():
#    datapack = make_example_datapack(45,10,20,pols=['XX'], name='datapack_test_solver.hdf5')
#    solver = LMCPhaseOnlySolver('./run_dir_test_solver_killms_40dof', '/net/lofar1/data1/albert/git/bayes_tec/scripts/data/killms_datapack.hdf5')
#    solver.run(iterations=5000,ant_sel="RS210HBA",
#            learning_rate=1e-2, overlap=160., max_block_size=400, dof_ratio=40.)
#    #os.unlink('datapack_test_solver.hdf5')

def test_new_solver():
#    datapack = make_example_datapack(45,10,20,pols=['XX'], name='datapack_test_solver.hdf5')
#    solver_cls = PhaseOnlySolver
    run_dir = "run_dir_killms"
    datapack = '/net/lofar1/data1/albert/git/bayes_tec/scripts/data/killms_datapack.hdf5'
#    num_threads = 1
    solver = PhaseOnlySolver(run_dir, datapack)
    solver.solve(solset='sol000', recalculate_coords=True, jitter=1e-6, tec_scale=0.005, screen_res=30, weight_smooth_len=40, reweight_obs=True, 
            learning_rate=1e-2,iterations=5000, minibatch_size=128, dof_ratio=30.,intra_op_threads=0, inter_op_threads=32, ant_sel='RS210HBA',
            time_sel=slice(0,100,1),debug=False)#,load_model='/home/albert/git/bayes_tec/src/bayes_tec/tests/run_dir_killms_new/run_021/checkpoints/save_000/model.hdf5')

#    solver = PhaseOnlySolver('./run_dir_test_solver_killms_new', '/net/lofar1/data1/albert/git/bayes_tec/scripts/data/killms_datapack.hdf5')#'./run_dir_test_solver', 'datapack_test_solver.hdf5')
#    
#    solver.solve(iterations=5000,ant_sel="RS210HBA", time_sel=slice(1,100,1), pol_sel=slice(0,1,1), jitter=1e-6, tec_scale=0.001,screen_res=30, weight_smooth_len=40,solset='sol000',reweight_obs=True,learning_rate=1e-2, minibatch_size=128, dof_ratio=40.,intra_op_threads=0, inter_op_threads=0)
    #os.unlink('datapack_test_solver.hdf5')

