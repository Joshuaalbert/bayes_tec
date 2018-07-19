import numpy as np
from concurrent import futures

def wrap(x):
    return np.arctan2(np.sin(x),np.cos(x))

def make_coord_array(*X):
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
    
    return np.reshape(X,(-1,X.shape[-1]))

def weights_and_mean_uncert(Y,N=200,phase_wrap=True, min_uncert=1e-3):
    """
    Get a weight matrix for each datapoint in Y using moving average of TD.
    The values must be uncorrelated
    Y: array shape [Nt], independent axis
    N : int the window
    Returns:
    weights [Nt] and mean_uncert float
    """
    if phase_wrap:
        dY = wrap(wrap(Y[:-1]) - wrap(Y[1:]))
    else:
        dY = Y[:-1] - Y[1:]
    dY = np.pad(dY,(0,N),mode='symmetric')
    uncert = np.sqrt(np.convolve(dY**2, np.ones((N,))/N, mode='valid',))
    weights = uncert
    weights = np.maximum(min_uncert,weights)
    mean_uncert = max(min_uncert,np.mean(weights))
    weights = 1./weights**2
    weights /= np.mean(weights)
    weights[np.isnan(weights)] = 1.
    return weights, mean_uncert

def _parallel_phase_weights(arg):
    phase, kwargs = arg
    w,u = weights_and_mean_uncert(phase,**kwargs)
    return w,u


def phase_weights(phase,indep_axis=-1,num_threads=None,**kwargs):
    """
    returns the weight matrix for phase
    phase: array [Nd, Na, Nf, Nt] or [Nd, Nf, Nt]
    Returns:
    array same shape as phase
    """
    shape = phase.shape
    roll_axis=False
    if indep_axis not in [-1, len(shape)-1]:
        roll_axis=True
        phase = np.swapaxes(phase,indep_axis,len(shape)-1)

    shape_ = phase.shape
    phase = phase.reshape((-1, shape_[-1]))
    N = phase.shape[0]
    args = []
    for i in range(N):
        args.append((phase[i,:], kwargs))

    with futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        jobs = executor.map(_parallel_phase_weights,args)
        results = list(jobs)

    weights = np.zeros_like(phase)
    uncert_mean = []
    for i in range(N):
        w,u = results[i]
        weights[i,:] = w
        uncert_mean.append(u)
    weights = weights.reshape(shape_)
    if roll_axis:
        return np.swapaxes(weights,indep_axis,len(shape)-1), np.mean(uncert_mean)
    else:
        return weights, np.mean(uncert_mean)
    
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
    dt = X_t[1,0] - X_t[0,0]
    T = X_t[-1,0] - X_t[0,0]
    N = int(np.ceil(overlap / dt))
    edges = list(range(0,X_t.shape[0],N))
    edges[-1] = X_t.shape[0] - 1

    block_size = edges[1] - edges[0]
    max_blocks = max_block_size // block_size
    if max_blocks < 3:
        return define_subsets(X_t, overlap + 1, max_block_size)
    block_start = 0
    blocks = []
    while block_start + max_blocks < len(edges):
        blocks.append((block_start, block_start + max_blocks))
        block_start = block_start + max_blocks - 1
    blocks.append((block_start, len(edges) -1))
    for b in blocks:
        if b[1] - b[0] < 3:
            return define_subsets(X_t, overlap+1,max_block_size)
    return edges, blocks

