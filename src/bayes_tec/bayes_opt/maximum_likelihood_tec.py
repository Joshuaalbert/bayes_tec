from ..logging import logging
from timeit import default_timer
import tensorflow as tf
import numpy as np
from collections import namedtuple
float_type = tf.float64

def scaled_square_dist_batched(X, X2, lengthscales):
    """
    X: tensor B, N, D
    X2: tensor B, M, D (or 1, M, D) and will be broadcast to B, M ,D
    Return:
    tensor B, N, M
    """
    # Clipping around the (single) float precision which is ~1e-45.
    X = X / lengthscales
    Xs = tf.reduce_sum(tf.square(X), axis=2)#B,N

    if X2 is None:
        dist = -2.*tf.matmul(X,X,transpose_b=True)
        dist += Xs[:,:,None] + Xs[:,None,:]
        return tf.maximum(dist, 1e-40)

    X2 = X2 / lengthscales# B (1), M, D
    X2s = tf.reduce_sum(tf.square(X2), axis=2)# B (1), M 
    dist = -2 * tf.matmul(X, X2, transpose_b=True)
    dist += Xs[:,:,None] + X2s[:,None,:]
    return tf.maximum(dist, 1e-40)
    
def kern(X, X2, T1, l):
    r2 = scaled_square_dist_batched(X,X2, l)
    r = tf.sqrt(scaled_square_dist_batched(X,X2, T1))
    return tf.cos(r) * tf.exp(-0.5*r2)

def conditional(Knm, Knn_sigma, Kmm, Y, jitter):
    """
    Return mean and variance of fstar | X, y, Xstar
    * we don't need cov so we don't compute it.
    See p.16 of RW.pdf
    Knm : B, N, M
    Knn_sigma: B, N, N
    Kmm: B, M (diag)
    Y: B, N, 1
    P = 1 because only one objective
    Returns:
    fmean: tensor (B, M)
    fvar: tensor (B, M)
    """
    Ln = tf.cholesky(Knn_sigma + jitter * tf.eye(tf.shape(Knn_sigma)[1],dtype=float_type)[None,:,:])#B, N, N
    A = tf.matrix_triangular_solve(Ln, Knm, lower=True)#B, N, M
    fvar = Kmm - tf.reduce_sum(tf.square(A),axis=1)#B, M
    #fvar = tf.tile(fvar[None, :], [P, 1])
    A = tf.matrix_triangular_solve(tf.transpose(Ln, (0, 2, 1)), A,lower=False)#B, N, M
    fmean = tf.matmul(A,Y, transpose_a=True)[:,:,0]#B, M
    return fmean, fvar

def mgf_acquisistion(fmean, fvar, t, fmin, jitter):
    """
    For reference see:
    @INPROCEEDINGS{8122656,
    author={H. Wang and B. van Stein and M. Emmerich and T. Back},
    booktitle={2017 IEEE International Conference on Systems, Man, and Cybernetics (SMC)},
    title={A new acquisition function for Bayesian optimization based on the moment-generating function},
    year={2017},
    pages={507-512},
    doi={10.1109/SMC.2017.8122656},
    month={Oct},}

    Get the acquistion function and the evaluated Xcand points.
    fmean and fvar are posterior at Xcand.
    fmean: tensor B, M
    fvar: tensor B,M
    t: tensor scalar exploration param
    fmin: B
    Returns: 
    Score: tensor B, M
    """
    fvar = tf.maximum(fvar, jitter)

    s = tf.sqrt(fvar)

    mp = fmean - fvar * t

    # Compute the acquisition
    normal = tf.distributions.Normal(mp, s, allow_nan_stats=False)# for debug don't allow
    t1 = normal.cdf(fmin) #B, M
    t2 = tf.exp((fmin - fmean - 1.)*t + 0.5 * fvar * t)# B, M
    return tf.multiply(t1, t2, name="mgf_acquisition")  # B, M

def wrap(phi):
    return tf.atan2(tf.sin(phi),tf.cos(phi))

def log_normal_solve(mean,std):
    mu = np.log(mean) - 0.5*np.log((std/mean)**2 + 1)
    sigma = np.sqrt(np.log((std/mean)**2 + 1))
    return mu, sigma

def likelihood(tec, phase, tec_conversion, lik_sigma, K = 2):
    """
    Get the likelihood of the tec given phase data and lik_var variance.
    tec: tensor B, 1
    phase: tensor B, Nf
    tec_conversion: tensor Nf
    lik_sigma: tensor B, 1
    Returns:
    log_prob: tensor (B,1)
    """
    mu = wrap(tec*tec_conversion[None,:])# B, Nf
    phase = wrap(phase)
    #K, B, Nf
    d = tf.stack([tf.distributions.Normal(mu + tf.convert_to_tensor(k*2*np.pi,float_type), 
                                          lik_sigma).log_prob(phase) for k in range(-K,K+1,1)], axis=0)
    
    #B, Nf -> B
    log_lik = tf.reduce_sum(tf.reduce_logsumexp(d, axis=0), axis=1)
    
    
    # B, 1
    tec_prior = tf.distributions.Normal(
        tf.convert_to_tensor(0.,dtype=float_type),
        tf.convert_to_tensor(0.5,dtype=float_type)).log_prob(tec)
    
    sigma_priors = log_normal_solve(0.2,0.1)
    #B, 1
    sigma_prior = tf.distributions.Normal(
        tf.convert_to_tensor(sigma_priors[0],dtype=float_type), 
        tf.convert_to_tensor(sigma_priors[1],dtype=float_type)).log_prob(tf.log(lik_sigma)) - tf.log(lik_sigma)
    
    #B, 1
    log_prob = log_lik[:,None] + tec_prior + sigma_prior
    
    return -log_prob

def init_population(phase, tec_conversion, max_tec=0.3, lik_sigma=0.3, N=5):
    """
    phase: B, Nf
    N: int num of samples
    Returns:
    X: B, N, D
    Y: B, N, 1
    """
    tec_conversion = tf.convert_to_tensor(tec_conversion, dtype=float_type)
    max_tec = tf.convert_to_tensor(max_tec, dtype=float_type)
    lik_sigma = tf.convert_to_tensor(lik_sigma, dtype=float_type)
    lik_log_sigma = tf.zeros(shape=tf.concat([tf.shape(phase)[0:1], tf.constant([1])],axis=0), dtype=float_type)
    lik_sigma = lik_sigma*tf.exp(lik_log_sigma)
    # initial population of X, Y
    X_init = tf.cast(tf.linspace(-max_tec, max_tec,N),float_type) # N
    X_init = tf.tile(X_init[:, None, None], tf.concat([tf.constant([1]), tf.shape(phase)[0:1], tf.constant([1])],axis=0))
    #N, B, 1
    Y_init = tf.map_fn(lambda X: likelihood(X, phase, tec_conversion, lik_sigma), X_init)
    return tf.transpose(X_init,(1,0,2)), tf.transpose(Y_init,(1,0,2))
    
    
BayesOptReturn = namedtuple('BayesOptReturn',['X', 'Y', 'aq', 'fmean', 'fvar'])

def bayes_opt_iter(phase, tec_conversion, X, Y, jitter = 1e-6, num_proposal=100, 
                   max_tec=0.3,t = 0.5, lik_sigma=0.3, l=0.195):
    """
    phase: tensor, B Nf
    X: tensor B N D
    Y: tensor B N 1
    """
    tec_conversion = tf.convert_to_tensor(tec_conversion, dtype=float_type)
    jitter = tf.convert_to_tensor(jitter, dtype=float_type)
    t = tf.convert_to_tensor(t, dtype=float_type)
    max_tec = tf.convert_to_tensor(max_tec, dtype=float_type)
    lik_sigma = tf.convert_to_tensor(lik_sigma, dtype=float_type)
    num_proposal = tf.convert_to_tensor(num_proposal, dtype=tf.int64)
    # based on bandwidth
    T1 = tf.reduce_max(tf.abs(1./tec_conversion))# ~ 0.11/(2*np.pi)
    l = 2*np.pi*T1# ~ 0.11
    # increase if failure, equiv to measurement uncert in -log_prob normalized
    f_sigma = tf.convert_to_tensor(0.005,dtype=float_type)

    # proposal array
    tec_array = tf.cast(tf.linspace(-max_tec, max_tec, num_proposal), dtype=float_type)
    grid_size = tf.cast(2*max_tec/(num_proposal-1),dtype=float_type)
    tec_array += tf.random_uniform((),tf.constant(0., dtype=float_type), grid_size, dtype=float_type)
    
    lik_log_sigma = tf.zeros(shape=tf.concat([tf.shape(phase)[0:1], tf.constant([1])],axis=0), dtype=float_type)
    lik_sigma = lik_sigma*tf.exp(lik_log_sigma)

    Xstar = tf.tile(tec_array[None,:,None], tf.concat([tf.shape(X)[0:1], tf.constant([1, 1])],axis=0))# B, M, D
    # standardise Y
    Y_norm = Y - tf.reduce_mean(Y,axis=1, keepdims=True) #B, N, 1
    Y_norm = Y_norm / (tf.sqrt(tf.reduce_mean(tf.square(Y_norm),axis=1,keepdims=True)) + 1e-6)
    fmin = tf.reduce_max(Y_norm, axis=1)# B, 1
    # get GPR ystar and varstar
    Knm = kern(X, Xstar, T1, l)#B, N, M
    Knn = kern(X, None, T1, l) #B, N, N
    Knn_sigma = Knn + tf.square(f_sigma)*tf.eye(tf.shape(X)[1], dtype=float_type)[None,:,:]

    Kmm = tf.ones(tf.shape(Xstar)[1:2],dtype=float_type)# M   #tf.diag_part(kern(Xstar, None, T1, T2, l))
    Kmm = tf.tile(Kmm[None,:], tf.concat([tf.shape(X)[0:1], tf.constant([1])],axis=0))# B, M
    fmean, fvar = conditional(Knm, Knn_sigma, Kmm, Y_norm, jitter)# B M, B M
    # Get acquisition function
    aq = mgf_acquisistion(fmean, fvar, t, fmin, jitter)# B, M

    idx_next = tf.argmax(aq, axis=1, name='idx_next')# B,
    Xnext = tf.gather(Xstar[0,:,:],idx_next, axis=0)# B, D
    #get value at next point
    Ynext = likelihood(Xnext, phase, tec_conversion, lik_sigma)# (B, 1)
    
    X = tf.concat([X, Xnext[:, None, :]], axis = 1)
    Y = tf.concat([Y, Ynext[:, None, :]], axis = 1)
    
    return BayesOptReturn(X, Y, aq, fmean, fvar)

def solve_ml_tec(phase, freqs, batch_size=1000, max_tec=0.3, num_proposal=100, n_iter = 21, init_pop = 5, t=0.5, verbose=False):
    tec_conversion = -8.448e9/freqs
    with tf.Session(graph=tf.Graph()) as sess:
        phase_pl = tf.placeholder(float_type)
        tec_conversion_pl = tf.placeholder(float_type)
        t_pl = tf.placeholder(float_type)

        X_init, Y_init = init_population(phase_pl,tec_conversion_pl,N=init_pop)
        Xcur, Ycur = X_init, Y_init
        X_,Y_,aq_,fmean_,fvar_ = [],[],[],[],[]
        for i in range(n_iter):
            res = bayes_opt_iter(phase_pl, tec_conversion_pl, Xcur, Ycur,
                                 num_proposal=num_proposal, t = t_pl,max_tec = max_tec)
            X_.append(res.X)
            Y_.append(res.Y)
            aq_.append(res.aq)
            fmean_.append(res.fmean)
            fvar_.append(res.fvar)
            Xcur = res.X
            Ycur = res.Y
        indices = tf.stack([tf.range(tf.cast(tf.shape(Ycur)[0],tf.int64),dtype=tf.int64),
                   tf.argmin(Ycur[:,:,0],axis=1),
                   tf.zeros(tf.shape(Ycur)[0], dtype=np.int64)],axis=1)
        tec_min = tf.gather_nd(Xcur,indices) #B,
        mu = tec_min[:,None]*tec_conversion_pl[None,:]
        error = wrap(wrap(mu) - wrap(phase_pl))#B,Nf
        phase_sigma = tf.sqrt(tf.reduce_mean(tf.square(error),axis=1))

        out_tec, out_sigma = [],[]
        for i in range(0,phase.shape[0], batch_size):
            phase_batch = phase[i:min(i+batch_size,phase.shape[0]), :]
            # get results
            if verbose:
                t0 = default_timer()
                logging.info("Starting batch {}".format(i))

            _tec, _sigma = sess.run([tec_min, phase_sigma],
                                             feed_dict={t_pl:t,
                                                        phase_pl:phase_batch,
                                                        tec_conversion_pl:tec_conversion})
            if verbose:
                t1 = default_timer()
                dt = t1-t0
                perc = 100.*float(i)/phase.shape[0]
                s =phase_batch.shape[0]
                logging.info("Finished batch {} {}% [{} {} samples/seconds {} ms/sample]".format(i,perc, dt, s/dt, dt*1000/s))

            out_tec.append(_tec)
            out_sigma.append(_sigma)

    return np.concatenate(out_tec, axis=0), np.concatenate(out_sigma, axis=0)
