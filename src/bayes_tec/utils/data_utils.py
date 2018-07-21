import numpy as np
from ..logging import logging
from scipy.ndimage.filters import convolve

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

def _parallel_weights(arg):
    position, dY2 = arg
    return np.roll(dY2, position, axis=1)

def calculate_weights(Y,indep_axis=-1, N=200,phase_wrap=True, min_uncert=1e-3, num_threads=None):
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
    shape = Y.shape
    roll_axis=False
    if indep_axis not in [-1, len(shape)-1]:
        roll_axis=True
        Y = np.swapaxes(Y,indep_axis,len(shape)-1)
    shape_ = Y.shape
    #N, Nt
    Y = Y.reshape((-1, shape_[-1]))

    if phase_wrap:
        dY2 = wrap(wrap(Y[:,1:]) - wrap(Y[:,:-1]))
    else:
        dY2 = Y[:,1:] - Y[:,:-1]

    dY2 = np.pad(dY2,((0,0),(0,1)),mode='symmetric')
    dY2 *= dY2

    args = []
    for i in range(-(N>>1),N>>1):
        args.append((i, dY2))

    with futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        jobs = executor.map(_parallel_weights,args)
        results = list(jobs)# each is N, Nt but rolled
    
    var = np.mean(np.stack(results, axis=0),axis=0)
    var = np.maximum(min_uncert**2, var)
    weights = 1./var


    weights = weights.reshape(shape_)
    if roll_axis:
        return np.swapaxes(weights,indep_axis,len(shape)-1)
    else:
        return weights
    
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

