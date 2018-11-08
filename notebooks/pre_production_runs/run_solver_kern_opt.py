from bayes_tec.solvers.phase_only_solver import PhaseOnlySolver
from bayes_tec.utils.data_utils import define_equal_subsets
from bayes_tec.logging import logging
import numpy as np
from timeit import default_timer
import gpflow as gp
from bayes_tec.utils.stat_utils import log_normal_solve, log_normal_solve_fwhm
from gpflow.priors import LogNormal

def create_kern(name):
    kerns = {'rbf':gp.kernels.RBF,'m12':gp.kernels.Matern12, 'm32':gp.kernels.Matern32, 'm52':gp.kernels.Matern52}
    s = name.split("_")
    k_time = kerns[s[1].lower()]
    k_dir = kerns[s[2].lower()]
    if s[0].lower() == 'sum':
        return _sum(k_time,k_dir)
    elif s[0].lower() == 'product':
        return _product(k_time,k_dir)

def _product(kern_time_, kern_dir_):
    def _kern(kern_ls_lower=0.75, kern_ls_upper=1.25, kern_dir_ls=0.5, kern_time_ls=50., kern_var=1., include_time=True, include_dir=True, **priors):
        kern_dir = kern_dir_(2,active_dims=slice(0,2,1))
        kern_time = kern_time_(1,active_dims=slice(2,3,1))
        kern = kern_dir*kern_time

        kern_var = 1. if kern_var == 0. else kern_var
        kern_dir.variance.trainable = False 
        kern_dir.lengthscales = kern_dir_ls
        kern_dir_ls = log_normal_solve_fwhm(kern_dir_ls*kern_ls_lower, kern_dir_ls*kern_ls_upper, D=0.1)    
        kern_dir.lengthscales.prior = LogNormal(kern_dir_ls[0], kern_dir_ls[1]**2)
        kern_dir.lengthscales.trainable = True
        kern_time.variance = kern_var
        kern_var = log_normal_solve_fwhm(kern_var*kern_ls_lower, kern_var*kern_ls_upper, D=0.1)
        kern_time.variance.prior = LogNormal(kern_var[0], kern_var[1]**2)
        kern_time.variance.trainable = True
        kern_time.lengthscales = kern_time_ls
        kern_time_ls = log_normal_solve_fwhm(kern_time_ls*kern_ls_lower, kern_time_ls*kern_ls_upper, D=0.1)
        kern_time.lengthscales.prior = LogNormal(kern_time_ls[0], kern_time_ls[1]**2)
        kern_time.lengthscales.trainable = True
        return kern
    return _kern

def _sum(kern_time_, kern_dir_):
    def _kern(kern_ls_lower=0.75, kern_ls_upper=1.25, kern_dir_ls=0.5, kern_time_ls=50., kern_var=1., include_time=True, include_dir=True, **priors):
        kern_dir = kern_dir_(2,active_dims=slice(0,2,1))
        kern_time = kern_time_(1,active_dims=slice(2,3,1))
        kern = kern_dir + kern_time

        kern_var = 1. if kern_var == 0. else kern_var
        kern_var = log_normal_solve_fwhm(kern_var*kern_ls_lower, kern_var*kern_ls_upper, D=0.1)
        kern_dir.variance.prior = LogNormal(kern_var[0], kern_var[1]**2)
        kern_dir.variance.trainable = True
        kern_dir.variance = np.exp(kern_var[0])
        kern_dir.lengthscales = kern_dir_ls
        kern_dir_ls = log_normal_solve_fwhm(kern_dir_ls*kern_ls_lower, kern_dir_ls*kern_ls_upper, D=0.1)    
        kern_dir.lengthscales.prior = LogNormal(kern_dir_ls[0], kern_dir_ls[1]**2)
        kern_dir.lengthscales.trainable = True

        kern_time.variance.prior = LogNormal(kern_var[0], kern_var[1]**2)
        kern_time.variance = np.exp(kern_var[0])
        kern_time.variance.trainable = True
        kern_time.lengthscales = kern_time_ls
        kern_time_ls = log_normal_solve_fwhm(kern_time_ls*kern_ls_lower, kern_time_ls*kern_ls_upper, D=0.1)
        kern_time.lengthscales.prior = LogNormal(kern_time_ls[0], kern_time_ls[1]**2)
        kern_time.lengthscales.trainable = True
        return kern
    return _kern





def test_new_solver():

#    opt = {'initial_learning_rate': 0.0469346965745387, 'learning_rate_steps': 2.3379450095649053, 'learning_rate_decay': 2.3096977604598385, 'minibatch_size': 257, 'dof_ratio': 15.32485312998133, 'gamma_start': 1.749795137201838e-05, 'gamma_add': 0.00014740343452076625, 'gamma_mul': 1.0555893705407017, 'gamma_max': 0.1063958902418518, 'gamma_fallback': 0.15444066000616663}
    
    opt = {'initial_learning_rate': 0.030035792298837113, 'learning_rate_steps': 2.3915384159241064, 'learning_rate_decay': 2.6685242978751798, 'minibatch_size': 128, 'dof_ratio': 10., 'gamma_start': 6.876944103773131e-05, 'gamma_add': 1e-4, 'gamma_mul': 1.04, 'gamma_max': 0.14, 'gamma_fallback': 0.1, 'priors' : {'kern_time_ls': 50., 'kern_dir_ls': 0.80}}


    datapack = '/net/lofar1/data1/albert/git/bayes_tec/scripts/data/killms_datapack_2.hdf5'
    run_dir='run_dir_killms_kern_opt'
    output_solset = "posterior_sol_kern_opt"

    time_sel = slice(50,150,1)
    ant_sel = "RS210HBA"
    
    
    import itertools
    res = []
    for s in itertools.product(['product','sum'],['rbf','m32','m52'],['rbf','m32','m52']):
        name = "_".join(s)
        logging.info("Running {}".format(name))
        solver = PhaseOnlySolver(run_dir, datapack)
        solver._build_kernel = create_kern(name)

        lik = solver.solve(output_solset=output_solset, solset='sol000', jitter=1e-6, tec_scale=0.005, screen_res=30, remake_posterior_solsets=False,
                    iterations=500,intra_op_threads=0, inter_op_threads=0, ant_sel=ant_sel, time_sel=time_sel,pol_sel=slice(0,1,1),debug=False, 
                    W_diag=True, freq_sel=slice(0,48,1), plot_level=-1, return_likelihood=True, num_likelihood_samples=100, **opt)
        res.append([name,-lik[0]/1e6,lik[1]/1e6])
        logging.info("{} results {}".format(name,res))
        with open("kern_opt_res.csv", 'a') as f:
            f.write("{}\n".format(str(res[-1]).replace('[','').replace(']','') ))
        

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
