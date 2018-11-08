from bayes_tec.bayes_opt.bayes_hp_opt import BayesHPOpt
from bayes_tec.solvers.phase_only_solver import PhaseOnlySolver
from concurrent import futures
import numpy as np



opt = {'initial_learning_rate': 0.03, 
        'learning_rate_steps': 2.39, 
        'learning_rate_decay': 2.66, 
        'minibatch_size': 128, 
        'dof_ratio': 20., 
        'gamma_start': 5e-05, 
        'gamma_add': 1e-4, 
        'gamma_mul': 1.04, 
        'gamma_max': 0.14, 
        'gamma_fallback': 0.1}

def _run(kwargs):
    datapack = '../../scripts/data/killms_datapack_2.hdf5'
    ant_sel='RS*'
    time_sel=slice(0,100,1)
    freq_sel=slice(0,48,1)
    pol_sel=slice(0,1,1)
    iterations=100
    run_dir='run_dir_hp_opt_SM_itoh'
    output_solset='posterior_sol_hp_opt_SM'

    opt['priors'] = {
            'time_period_uncert':[kwargs['t_uncert{}'.format(i)] for i in range(3)],
            'dir_period_uncert':[kwargs['d_uncert{}'.format(i)] for i in range(3)],
            'time_periods':[kwargs['t_period{}'.format(i)] for i in range(3)],            
            'dir_periods':[kwargs['d_period{}'.format(i)] for i in range(3)],
            'w_init':[kwargs['w{}'.format(i)] for i in range(3)]
            }

    solver = PhaseOnlySolver(run_dir, datapack)
    m,s = solver.solve(output_solset=output_solset, solset='sol000',
                        jitter=1e-6,tec_scale=0.005,screec_res=30,
                        iterations=iterations,
                        remake_posterior_solsets=False, inter_op_threads=0,
                        intra_op_threads=0,ant_sel=ant_sel, time_sel=time_sel,
                        pol_sel=pol_sel, freq_sel=freq_sel,debug=False,
                        W_diag=False,return_likelihood=True,num_likelihood_samples=100,
                        plot_level=-3, compute_posterior=False,  **opt)
    return -m/1e6

def objective(**kwargs):
    return _run(kwargs)

    with futures.ThreadPoolExecutor(max_workers=2) as exe:
        jobs = exe.map(_run, [kwargs])
        res = list(jobs)
    return np.mean(res)

#1.326511211345239 -> {'initial_learning_rate': 0.08672814094012078, 'learning_rate_steps': 3.845691869451716, 'learning_rate_decay': 2.2338225170518045, 'minibatch_size': 343, 'dof_ratio': 17.483912391131362, 'gamma_start': 1.8893702113878085e-05, 'gamma_add': 0.00025304970643971796, 'gamma_mul': 1.1673530952703717, 'gamma_max': 0.21196916296812654, 'gamma_fallback': 0.15811131579133963}
#WARNING:root:1 (43)  : 1.4169010382169323 -> {'initial_learning_rate': 0.04693469657453876, 'learning_rate_steps': 2.3379450095649053, 'learning_rate_decay': 2.309697760459837, 'minibatch_size': 257, 'dof_ratio': 15.324853129981337, 'gamma_start': 1.7497951372018477e-05, 'gamma_add': 0.00024740343452076625, 'gamma_mul': 1.1955893705407017, 'gamma_max': 0.34639589024185186, 'gamma_fallback': 0.15444066000616663}
#WARNING:root:2 (27)  : 1.5081948998999013 -> {'initial_learning_rate': 0.009944268827823587, 'learning_rate_steps': 2.7228499570724916, 'learning_rate_decay': 1.268929681705544, 'minibatch_size': 484, 'dof_ratio': 15.793002501207107, 'gamma_start': 1.3162914446789919e-05, 'gamma_add': 0.0014083695974122102, 'gamma_mul': 1.1920515053318887, 'gamma_max': 0.08734702837532575, 'gamma_fallback': 0.21598310688240693}
#WARNING:root:3 (51)  : 1.5183867590113769 -> {'initial_learning_rate': 0.09929641010035925, 'learning_rate_steps': 3.760297282474147, 'learning_rate_decay': 1.9596598257348894, 'minibatch_size': 381, 'dof_ratio': 19.712394557961836, 'gamma_start': 5.7113644372202535e-05, 'gamma_add': 0.00039745743579932673, 'gamma_mul': 1.0104099384398493, 'gamma_max': 0.49512123114366735, 'gamma_fallback': 0.2273128821926654}
#WARNING:root:4 (41)  : 1.5421102537039924 -> {'initial_learning_rate': 0.03999651253015149, 'learning_rate_steps': 2.7655606636091004, 'learning_rate_decay': 2.252062714633563, 'minibatch_size': 257, 'dof_ratio': 19.897864384533356, 'gamma_start': 2.2467224826890863e-05, 'gamma_add': 0.00048298906787098023, 'gamma_mul': 1.0293807927120147, 'gamma_max': 0.45511367853454426, 'gamma_fallback': 0.22026128808845857}



bo = BayesHPOpt(objective,init='hp_opt_results_SM_itoh_td_yvar.hdf5',t=20.)
#     initial_learning_rate=0.1,learning_rate_steps=2,
#               learning_rate_decay=1.5,
#               minibatch_size=128, dof_ratio=30,
#              gamma_start=1e-5,gamma_add=1e-3,gamma_mul=1.1,
#              gamma_max=0.15,gamma_fallback=1e-1):
#bo.add_continuous_param('initial_learning_rate',1e-3,1e-1,log=True)
#bo.add_continuous_param('learning_rate_steps',1,4)
#bo.add_continuous_param('learning_rate_decay',1.,3.)
#bo.add_integer_param('minibatch_size',16,512)
#bo.add_continuous_param('dof_ratio',14,21)
#bo.add_continuous_param('gamma_start',1e-7,1e-4,log=True)
#bo.add_continuous_param('gamma_add',1e-5,1e-2,log=True)
#bo.add_continuous_param('gamma_mul',1.01,1.3,log=True)
#bo.add_continuous_param('gamma_max',0.01,0.5,log=True)
#bo.add_continuous_param('gamma_fallback',1e-3,5e-1,log=True)
#bo.add_continuous_param('kern_time_ls',25,100,log=True)
#bo.add_continuous_param('kern_dir_ls',0.1,1.2,log=True)
bo.add_continuous_param('t_period0',15,200.,log=False)
bo.add_continuous_param('t_period1',15,200.,log=False)
bo.add_continuous_param('t_period2',15,200.,log=False)
bo.add_continuous_param('d_period0',0.3, 2.,log=False)
bo.add_continuous_param('d_period1',0.3, 2.,log=False)
bo.add_continuous_param('d_period2',0.3, 2.,log=False)
bo.add_continuous_param('t_uncert0',1.,100.,log=True)
bo.add_continuous_param('t_uncert1',1.,100.,log=True)
bo.add_continuous_param('t_uncert2',1.,100.,log=True)
bo.add_continuous_param('d_uncert0',0.05,2.,log=True)
bo.add_continuous_param('d_uncert1',0.05,2.,log=True)
bo.add_continuous_param('d_uncert2',0.05,2.,log=True)
bo.add_continuous_param('w0',0.1, 2.,log=False)
bo.add_continuous_param('w1',0.1, 2.,log=False)
bo.add_continuous_param('w2',0.1, 2.,log=False)


bo.run('hp_opt_results_SM_itoh_td_yvar.hdf5',init_design_size=0,n_iter=0,plot=True,likelihood_uncert=0.1)                                            
