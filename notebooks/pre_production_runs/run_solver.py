from bayes_tec.solvers.phase_only_solver import PhaseOnlySolver
from bayes_tec.utils.data_utils import define_equal_subsets
from bayes_tec.logging import logging
import numpy as np
from timeit import default_timer

def test_new_solver():

#    opt = {'initial_learning_rate': 0.0469346965745387, 'learning_rate_steps': 2.3379450095649053, 'learning_rate_decay': 2.3096977604598385, 'minibatch_size': 257, 'dof_ratio': 15.32485312998133, 'gamma_start': 1.749795137201838e-05, 'gamma_add': 0.00014740343452076625, 'gamma_mul': 1.0555893705407017, 'gamma_max': 0.1063958902418518, 'gamma_fallback': 0.15444066000616663}
    
    opt = {'initial_learning_rate': 0.03, 'learning_rate_steps': 2.39, 'learning_rate_decay': 2.66, 'minibatch_size': 1024, 'dof_ratio': 7., 'gamma_start': 5e-05, 'gamma_add': 1e-4, 'gamma_mul': 1.04, 'gamma_max': 0.14, 'gamma_fallback': 0.1}
#    opt['priors'] = {'kern_time_ls': 42.20929516497659, 'kern_dir_ls': 0.36789336277387313}


    datapack = '/net/lofar1/data1/albert/git/bayes_tec/scripts/data/killms_datapack_4.hdf5'
    run_dir='run_dir_killms_gains'
    output_solset = "posterior_sol_gains_Wcon"

    solver = PhaseOnlySolver(run_dir, datapack)

    
    solve_slices, set_slices, subset_slices = define_equal_subsets(3600,200,20)
    for solve_slice, set_slice, subset_slice in zip(solve_slices, set_slices, subset_slices):
        time_sel = slice(*solve_slice,1)
        opt['posterior_save_settings'] = {'save_time_sel':slice(*set_slice,1), 'subset_slice':slice(*subset_slice,1)}
        logging.debug(time_sel)
        logging.debug(opt['posterior_save_settings'])

#    for start in range(1,3600, 100):
#        stop = min(3600, start + 100)
#
#        time_sel = slice(start,stop,1)

        solver.solve(output_solset=output_solset, solset='sol000', jitter=1e-6, tec_scale=0.005, screen_res=30, remake_posterior_solsets=False,
                    iterations=400,intra_op_threads=0, inter_op_threads=0, ant_sel="RS*", dir_sel=slice(None,None,1), time_sel=time_sel,pol_sel=slice(0,1,1),debug=False, 
                    W_trainable=True, freq_sel=slice(0,48,1), plot_level=-1, **opt)
    
#        solver.solve(output_solset=output_solset, solset='sol000', jitter=1e-6, tec_scale=0.005, screen_res=30, remake_posterior_solsets=False,
#                   iterations=500, intra_op_threads=0, inter_op_threads=0, ant_sel="CS*", time_sel=time_sel,pol_sel=slice(0,1,1),debug=False, 
#                   W_diag=False, freq_sel=slice(0,48,1), **opt)


#    W_diag = False
#    dof_ratio = 20.
#
#    run_dir = "run_dir_killms_notime_{}_{}".format(int(dof_ratio),'diag' if W_diag else 'chol')
#    output_solset = "posterior_sol_notime_{}_{}".format(int(dof_ratio),'diag' if W_diag else 'chol')
#    solver = PhaseOnlySolver(run_dir, datapack)
#    solver.solve(output_solset=output_solset, solset='sol000', jitter=1e-6, tec_scale=0.005, screen_res=30, remake_posterior_solsets=False,
#                initial_learning_rate=1e-2, final_learning_rate=1e-3, iterations=2000, minibatch_size=128, dof_ratio=dof_ratio,
#                intra_op_threads=0, inter_op_threads=0, ant_sel=ant_sel, time_sel=time_sel,pol_sel=slice(0,1,1),debug=False, W_diag=W_diag, freq_sel=slice(0,48,1))







#    ###
#    # RS
#    for i in range(18):
#        time_sel = slice(i*200,min(3600,(i+1)*200),1)
#        solver.solve(output_solset=output_solset, solset='sol000', jitter=1e-6, tec_scale=0.005, screen_res=30, remake_posterior_solsets=False,
#                learning_rate=1e-2,iterations=2000, minibatch_size=128, dof_ratio=20.,intra_op_threads=0, inter_op_threads=0, ant_sel="RS*",
#                time_sel=time_sel,pol_sel=slice(0,1,1),debug=False, W_diag=True, freq_sel=slice(0,48,1))
#
#    ###
#    # CS
#    for i in range(18):
#        time_sel = slice(i*200,min(3600,(i+1)*200),1)
#        solver.solve(output_solset=output_solset, solset='sol000', jitter=1e-6, tec_scale=0.005, screen_res=30, remake_posterior_solsets=False,
#                learning_rate=1e-2,iterations=2000, minibatch_size=128, dof_ratio=20.,intra_op_threads=0, inter_op_threads=0, ant_sel="CS*",
#                time_sel=time_sel,pol_sel=slice(0,1,1),debug=False, W_diag=True, freq_sel=slice(0,48,1))



#    run_dir = "run_dir_killms_10_Wdiag"
#    datapack = '/net/lofar1/data1/albert/git/bayes_tec/scripts/data/killms_datapack_3.hdf5'
#    solver = PhaseOnlySolver(run_dir, datapack)
#    solver.solve(solset='sol000', recalculate_coords=False, jitter=1e-6, tec_scale=0.005, screen_res=30, weight_smooth_len=40, reweight_obs=False, 
#            learning_rate=1e-2,iterations=2000, minibatch_size=128, dof_ratio=10.,intra_op_threads=0, inter_op_threads=0, ant_sel="RS*",
#            time_sel=slice(100,200,1),pol_sel=slice(0,1,1),debug=False, W_diag=True)

#    run_dir = "run_dir_killms_10_chol"
#    datapack = '/net/lofar1/data1/albert/git/bayes_tec/scripts/data/killms_datapack_3.hdf5'
#    solver = PhaseOnlySolver(run_dir, datapack)
#    solver.solve(solset='sol000', recalculate_coords=False, jitter=1e-6, tec_scale=0.005, screen_res=30, weight_smooth_len=40, reweight_obs=False, 
#            learning_rate=1e-2,iterations=2000, minibatch_size=128, dof_ratio=10.,intra_op_threads=0, inter_op_threads=0, ant_sel="RS*",
#            time_sel=slice(100,200,1),pol_sel=slice(0,1,1),debug=False, W_diag=False)
#
#    run_dir = "run_dir_ndppp_10_Wdiag"
#    datapack = '/net/lofar1/data1/albert/git/bayes_tec/scripts/data/ndppp_datapack.hdf5'
#    solver = PhaseOnlySolver(run_dir, datapack)
#    solver.solve(solset='sol000', recalculate_coords=False, jitter=1e-6, tec_scale=0.005, screen_res=30, weight_smooth_len=40, reweight_obs=False, 
#            learning_rate=1e-2,iterations=2000, minibatch_size=128, dof_ratio=10.,intra_op_threads=0, inter_op_threads=0, ant_sel="RS*",
#            time_sel=slice(100,200,1),pol_sel=slice(0,1,1),debug=False, W_diag=True)
#
#    run_dir = "run_dir_ndppp_10_chol"
#    datapack = '/net/lofar1/data1/albert/git/bayes_tec/scripts/data/killms_datapack.hdf5'
#    solver = PhaseOnlySolver(run_dir, datapack)
#    solver.solve(solset='sol000', recalculate_coords=False, jitter=1e-6, tec_scale=0.005, screen_res=30, weight_smooth_len=40, reweight_obs=False, 
#            learning_rate=1e-2,iterations=2000, minibatch_size=128, dof_ratio=10.,intra_op_threads=0, inter_op_threads=0, ant_sel="RS*",
#            time_sel=slice(100,200,1),pol_sel=slice(0,1,1),debug=False, W_diag=False)

if __name__ == '__main__':
    test_new_solver()
