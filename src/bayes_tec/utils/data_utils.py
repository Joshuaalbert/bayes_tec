import numpy as np
from ..logging import logging
from scipy.ndimage.filters import convolve
from concurrent import futures
import tensorflow as tf
from collections import namedtuple
from scipy.signal import stft

def wrap(x):
    return np.arctan2(np.sin(x),np.cos(x))

def make_coord_array(*X,flat=True):
    """
    Return the design matrix from coordinates.
    """
    def add_dims(x,where,sizes):
        shape = []
        tiles = []
        for i in range(len(sizes)):
            if i not in where:
                shape.append(1)
                tiles.append(sizes[i])
            else:
                shape.append(-1)
                tiles.append(1)
        return np.tile(np.reshape(x,shape),tiles)
    N = [x.shape[0] for x in X]
    X_ = []

    for i,x in enumerate(X):
        for dim in range(x.shape[1]):
            X_.append(add_dims(x[:,dim],[i], N))
    X = np.stack(X_,axis=-1)
    if not flat:
        return X 
    return np.reshape(X,(-1,X.shape[-1]))

def _parallel_shift(arg):
    position, dY2, axis = arg
    return np.roll(dY2, position, axis=axis)

def calculate_weights(Y,indep_axis=-1, N=00,phase_wrap=True, min_uncert=1e-3, num_threads=None):
    """
    Get a weight matrix for each datapoint in Y using moving average of TD.
    The values must [... , Nt], Nt is uncorrelated axis
    Y: array shape [... , Nt], independent axis
    indep_axis: int the axis to do TD down
    N : int the window size
    phase_wrap : bool whether to phase wrap differences
    min_uncert : float the minimum allowed uncertainty
    num_threads: the number of threads to use, None used ncpu*5
    Returns:
    weights [..., Nt] of same shape and dtype as Y
    """
    if phase_wrap:
        z = np.exp(1j*Y)
        args = []
        for i in range(-(N>>1),N-(N>>1)):
            args.append((i, z, indep_axis))

        with futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            jobs = executor.map(_parallel_shift,args)
            results = list(jobs)# each is N, Nt but rolled
        for r in results[1:]:
            results[0] = results[0] + r
        results[0] = results[0]/N
        z_mean = results[0]
        R2 = z_mean * z_mean.conj()
        Re2 = N/(N-1)*(R2 - 1./N)
        var = -np.log(Re2).astype(Y.dtype)
    else:
        args = []
        for i in range(-(N>>1),N-(N>>1)):
            args.append((i, Y,indep_axis))
        with futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            jobs = executor.map(_parallel_shift,args)
            results = list(jobs)# each is N, Nt but rolled
        mean = results[0].copy()
        for r in results[1:]:
            mean += r
        mean /= N
        var = (results[0] - mean)**2
        for r in results[1:]:
            var += np.abs(r - mean)**2
        var /= N
    var = np.maximum(min_uncert**2, var)
    return var
    
def make_data_vec(Y,freqs,weights=None):
    """
    Stacks weights, and repeats the freqs and puts at the end of Y so that 
    output[...,0:N] = Y
    output[...,N:2N] = weights
    output[...,2N:2N+1] = freqs in the proper order.
    Y : array (..., Nf, N)
    freqs : array (Nf,)
    weights: array same shape as Y optional
        Weights for Y if available (else use ones)
    Returns:
    array (..., Nf, 2*N+1)
    """
    shape = Y.shape
    for _ in range(len(shape)-2):
        freqs = freqs[None,...]
    freqs = freqs[...,None]
    # freqs is now [1,1,...,1,Nf,1]
    tiles = list(shape)
    # [..., 1, 1]
    tiles[-1] = 1
    tiles[-2] = 1
    # [..., Nf, 1]
    freqs = np.tile(freqs,tiles)
    if weights is None:
        weights = np.ones_like(Y)
    # ..., Nf, 2*N+1
    return np.concatenate([Y, weights, freqs],axis=-1)



def define_equal_subsets(N,max_block_size, min_overlap,verbose=False):
    """
    Break an abscissa into equal overlaping regions using modular arithmetic.
    Args:
    N : int length of abscisa
    max_block_size : int maximal size of partitions
    min_overlap : int minimum overlap in units of elements
    verbose: bool print the options

    Returns:
    blocks, val_blocks which are lists of (start,end) tuples that can be used to 
    construct slices of time blocks.
    """

    def xgcd(b, a):
        x0, x1, y0, y1 = 1, 0, 0, 1
        while a != 0:
            q, b, a = b // a, a, b % a
            x0, x1 = x1, x0 - q * x1
            y0, y1 = y1, y0 - q * y1
        return  b, x0, y0

    def mulinv(b, n):
        g, x, _ = xgcd(b, n)
        if g == 1:
            return x % n

    res = []
    for n in range(1,N):
        a = -2*n
        b = n+1
        ainv = mulinv(a,b)
        if ainv is None:
            continue
        O = (ainv * N) % b
        B = (N - a*O)//b
        if B <= max_block_size and O >= min_overlap and B - 2*O > 0:
            res.append((n,B,O))

    if len(res) == 0:
        raise ValueError("Incompatible max blocksize and min overlap. Try raising or lowering respectively.")
    possible = np.array(res)

    ##
    # selection
    min_n = np.argmin(possible[:,0])
    res = possible[min_n,:]
    if verbose:
        verb = "\n".join(["  {:3d}|{:3d}|{:3d}".format(*r) if not np.all(r == res) else ">>{:3d}|{:3d}|{:3d}".format(*r) for r in possible])
        logging.warning("Available configurations:\n  ( n|  B| overlap )\n{}".format(verb))

    blocks, val_blocks, inv_map = [],[],[]
    start=0
    i = 0
    n,B,O = res
    while i <= n:
        blocks.append((i*B - i*2*O, (i+1)*B - i*2*O))
        if i == 0:
            val_blocks.append((blocks[-1][0], blocks[-1][1]-O))
            inv_map.append((0,B-O))
        elif i == n:
            val_blocks.append((blocks[-1][0] + O, blocks[-1][1]))
            inv_map.append((O,B))
        else:
            val_blocks.append((blocks[-1][0] + O, blocks[-1][1] - O))
            inv_map.append((O,B-O))
        i += 1
    return blocks, val_blocks, inv_map


def define_subsets(X_t, overlap, max_block_size):
    """
    Define the subsets of X_t with minimum overlap size blocks, 
    as a set of edges.
    X_t :array (N,1)
        times
    overlap : float
    max_block_size : int
        The max number of points per block
    Returns:
    list of int, The edges
    """
    assert overlap < X_t[-1,0] - X_t[0,0]
    dt = X_t[1,0] - X_t[0,0]
    max_block_size = int(max_block_size)
    
    M = int(np.ceil(X_t.shape[0]/max_block_size))

    edges = np.linspace(X_t[0,0],X_t[-1,0],M+1)
    
    edges_idx = np.searchsorted(X_t[:,0],edges)
    starts = edges_idx[:-1]
    stops = edges_idx[1:]
    for s1,s2 in zip(starts,stops):
        assert X_t[s2,0] - X_t[s1,0] >= 3*overlap, "Overlap ({}) -> {} and max_block_size ({}) incompatible".format(overlap,overlap/dt,max_block_size)
    return starts,stops

def _old_define_subsets(X_t, overlap, max_block_size):
    """
    Define the subsets of X_t with minimum overlap size blocks, 
    as a set of edges.
    X_t :array (N,1)
        times
    overlap : float
    max_block_size : int
        The max number of points per block
    Returns:
    list of int, The edges
    """
    max_block_size = int(max_block_size)
    dt = X_t[1,0] - X_t[0,0]
    T = X_t[-1,0] - X_t[0,0]
    N = int(np.ceil(overlap / dt))
    assert N < max_block_size, "overlap ({}) larger than max_block_size ({})".format(N, max_block_size)
    assert N < X_t.shape[0], "overlap requested ({}) requested larger than full time range ({})".format(N,X_t.shape[0])
    edges = list(range(0,X_t.shape[0],N))
    edges[-1] = X_t.shape[0]-1

    block_size = edges[1] - edges[0]
    max_blocks = max_block_size // block_size
    if max_blocks < 3:
        return define_subsets(X_t, overlap + 1, max_block_size)
    block_start = 0
    blocks = []
    while block_start + max_blocks < len(edges):
        blocks.append((block_start, block_start + max_blocks))
        block_start = block_start + max_blocks - 1
    blocks.append(blocks[-1])
#    blocks.append((block_start, len(edges) -1))
    for b in blocks:
        if b[1] - b[0] < 3:
            return define_subsets(X_t, overlap+1,max_block_size)
    return edges, blocks


def calculate_empirical_W(f, axis):
    """Calculate the empirical coupling matrix for LMC from f down axis"""
    shape = f.shape
    f = np.swapaxes(f, axis, -1)
    f = f.reshape((-1, shape[axis]))
    cov = np.cov(f,rowvar=False)
    d, u = np.linalg.eigh(cov)
    W = u.dot(np.diag(np.sqrt(d)))
    return W

    std = np.maximum(np.sqrt(np.diag(cov)), 1e-6)
    cor = cov / (std[:, None] * std[None, :])
    return cor

def solve_spectral_mixture(lag, K, Fs, mu_prior=0.2, kern_sigma = 0.01, N=10000, Q=3):


    def get_starting_point(lag,K,N=N):
        "Iteratively, solve for each Q with random search"

        def _iter(w, mu, v, N):
            #Q,N
            if w is None:
                w = np.ones((1, N))
            else:
                w = np.concatenate( [np.tile(w[:, None], (1, N)),
                                     np.random.uniform(0., 1., size=[1, N])], axis=0)
                w /= np.sum(w,axis=0,keepdims=True)
            if mu is None:
                mu = np.random.uniform(0., 0.05*0.5*Fs, size=[1, N])
            else:
                mu = np.concatenate([np.tile(mu[:, None], (1, N)),
                                     np.random.uniform(0., mu_prior*0.5*Fs, size=[1, N])], axis=0)
            if v is None:
                v = 1/(2*np.pi*np.random.uniform(np.percentile(lag,1), np.max(lag), size=[1, N]))**2
            else:
                v = np.concatenate([np.tile(v[:, None], (1, N)),
                                    1/(2*np.pi*np.random.uniform(np.percentile(lag,1), np.max(lag), size=[1, N]))**2],axis=0)

            #N, T
            r = np.sum(w[:,:, None]*np.cos(2*np.pi*lag[None,None,:]*mu[:,:, None])*np.exp(-2*np.pi**2*lag[None,None,:]**2*v[:,:,None]),axis=0)
            r = np.mean(np.abs(r - K[None,:])/kern_sigma,axis=1)# N
            a = np.argmin(r)
            return r[a], w[:, a], mu[:,a], v[:, a]
        w,mu,v = None, None, None
        for q in range(Q):
            loss, w, mu, v = _iter(w, mu, v, N)

        return w, mu, v

    # TF solve this point
    def forward_logistic(x, a, b):
        ex = tf.exp(-x)
        return a + (b - a) / (1. + ex)

    def backward_logistic(y, a, b):
        return -np.log((b - a) / (y - a) - 1.)

    def optimize_kern(w_init, mu_init, v_init, epochs = 1000):

        if w_init is not None:
            w_init = np.log(w_init)
        if mu_init is not None:
            mu_init = backward_logistic(mu_init, -1e-6, Fs/2.)
        if v_init is not None:
            v_init = np.log(v_init)


        def tf_func(x, w, mu,v):
            mu = forward_logistic(mu, 0., Fs/2.)
            r = tf.reduce_sum(tf.exp(w)[:, None] * tf.cos(2*np.pi*x[None, :] * mu[:, None]) *\
                             tf.exp(-2*np.pi**2 * x**2 * tf.exp(v[:, None])), axis= 0)
            return r

        with tf.Session(graph=tf.Graph()) as sess:

            x_pl = tf.placeholder(tf.float64)
            y_pl = tf.placeholder(tf.float64)
            kern_var_pl = tf.placeholder(tf.float64)
            w = tf.Variable(w_init, name='w',dtype=tf.float64)
            mu = tf.Variable(mu_init, name='mu',dtype=tf.float64)
            v = tf.Variable(v_init, name='v',dtype=tf.float64)

            r = tf_func(x_pl, w, mu, v)

            dy = y_pl - r
            dy = tf.square(dy)/(kern_var_pl)
            loss = 0.5*tf.reduce_mean(dy)# + 0.1*tf.reduce_mean(tf.abs(tf.exp(w[:, None]) / tf.reduce_sum(tf.exp(w))))

            opt = tf.train.AdamOptimizer(1e-3).minimize(loss)

            sess.run(tf.global_variables_initializer())

            for i in range(epochs):
                _, w_, mu_, v_, loss_ = sess.run([opt, w, mu, v, loss],
                                                 feed_dict = {x_pl:lag, y_pl:K,
                                                              kern_var_pl:kern_sigma**2})
#            print("{} loss {} w {} mu {} v {}".format(i, loss_,
#                                                      np.exp(w_), -1e-6 + (Fs/2. + 1e-6) / ( 1 + np.exp(-mu_)), np.exp(v_)))

        return np.exp(w_), -1e-6 + (Fs/2. + 1e-6) / ( 1 + np.exp(-mu_)), np.exp(v_)

    w,mu,v = get_starting_point(lag,K,N=10000)
    w, mu, v = optimize_kern(w, mu, v, 100)
    return w, mu, v

KernParams = namedtuple('KernParams',['w', 'mu', 'v'])

def calculate_empirical_spectral_kern(tec, times, directions):
    """Get the spectral mixture kernel params from the empirical kernel.
    tec : array [Npol, Nd, Na, Nt]
    times: array [Nt]
    directions  : array [Nd,2]
    Returns:
        List of KernParams, one entry per Na
    """
    Nd, _ = directions.shape
    Nt = times.shape[0]
    Na = tec.shape[2]
    Fs_time = len(times)/(np.max(times) - np.min(times))
    Fs_ra = np.sqrt(directions.shape[0]) / (np.max(directions[:,0]) - np.min(directions[:,0]))
    Fs_dec = np.sqrt(directions.shape[0]) / (np.max(directions[:,1]) - np.min(directions[:,1]))

    # time
    def get_empirical_time_kernel(y, Fs, block_size=None):
        """
        Calcualgte the empirical kernel in time for blocks.
        Args:
            y : array [Nd, Na, Nt]
        Returns:
            lag (Nlag,), K (Na, Nlag, Nblocks) if block_size is not None else (Na, Nlag)

        """
        #Nd, Na, Nt
        y = y - y.mean(-1,keepdims=True)
        k = np.zeros([Na,Nt])
        ks = np.zeros([Na,Nt])
        #Na, Nt, Nt
        C = np.mean(y[:,:, :, None]*y[:,:,None,:], axis=0)
        for j in range(Na):
            for i in range(100):
                k[j,i] = np.mean(np.diag(C[j,:,:],i))
                ks[j,i] = np.std(np.diag(C[j,:,:],i))
            where = ks[j,:] == 0.
            ks[j,where] = np.mean(ks[j,~where])

        ks /= k[:,0:1]
        k /= k[:,0:1]
        return times - times[0], k , ks

#        _bs = False
#        if block_size is None:
#            _bs = True
#            block_size = y.shape[-1]
#        y = y# - y.mean(axis=-1, keepdims=True)
#        f,t,sp = stft(y, Fs, nperseg = block_size,
#                  noverlap=0,boundary='constant',axis=-1)
#
#        S = np.exp(np.log(np.maximum(np.abs(sp), 1e-20)**2).mean(0))
#        K = np.fft.hfft(S,axis=-2)[:,:S.shape[1], :]
#        K /= K[:,0:1,:]#Na, Nlag, Nblocks
#        lag = np.linspace(0.,K.shape[1]/Fs, K.shape[1])
#        if _bs:
#            return lag, K[..., -1]
#        return lag, K

    def get_empirical_dir_kernel(y, Fs):
        """
        Calcualgte the empirical kernel in time for blocks.
        Args:
            y : array [Nd, Na, Nt]
        Returns:
            lag (Nlag,), K_ra (Na, Nlags),  K_ra (Na, Nlags)

        """
        y = y# - y.mean(axis=-1, keepdims=True)

        fourier_freqs = np.linspace(-Fs/2., Fs/2., directions.shape[0])
        ra = directions[:, 0]
        W = np.exp(-2j*np.pi*(fourier_freqs[:, None]*ra[None,:]))
        xf = np.log(np.maximum(np.abs(np.einsum('ab,bcd->acd',W,y)), 1e-20)).mean(2)#Ns, Na
        dir_freqs = np.linspace(0, 2.5, 45)
        Wf = np.exp(2j*np.pi*(dir_freqs[:, None]*fourier_freqs[None,:]))

        K = np.einsum("ab,bc->ac",Wf, np.exp(xf)).real#Nlags, Na
        K_ra = K / K[0:1,:]

        dec = directions[:, 1]
        W = np.exp(-2j*np.pi*(fourier_freqs[:, None]*ra[None,:]))
        xf = np.log(np.maximum(np.abs(np.einsum('ab,bcd->acd',W,y)), 1e-20)).mean(2)#Ns, Na

        K = np.einsum("ab,bc->ac",Wf, np.exp(xf)).real#Nlags, Na
        K_dec = K/ K[0:1,:]


        return dir_freqs, K_ra.T, K_dec.T


    lag_time, K_time, K_time_sigma = get_empirical_time_kernel(tec[0, ...], Fs_time)
#    lag_dir, K_ra, K_dec = get_empirical_dir_kernel(tec[0, ...], np.sqrt(45)/2.5)
    kern_params = []
    for i in range(K_time.shape[0]):
        w_time, mu_time, v_time = solve_spectral_mixture(lag_time[:], K_time[i,:], 1./8., mu_prior=1., kern_sigma = K_time_sigma[i,:], N=30000, Q=3)
#        w_ra, mu_ra, v_ra = solve_spectral_mixture(lag_dir, K_ra[i,:], np.sqrt(45)/2.5, mu_prior=1., kern_sigma = 0.01, N=10000, Q=3)
#        w_dec, mu_dec, v_dec = solve_spectral_mixture(lag_dir, K_dec[i,:], np.sqrt(45)/2.5, mu_prior=1., kern_sigma = 0.01, N=10000, Q=3)
#        w = np.stack([w_ra, w_dec, w_time], axis=1)
#        mu = np.stack([mu_ra, mu_dec, mu_time], axis=1)
#        v = np.stack([v_ra, v_dec, v_time], axis=1)
        kern_params.append(KernParams(w_time,mu_time,v_time))

    return kern_params

    

