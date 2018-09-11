import numpy as np
from ..logging import logging
from scipy.ndimage.filters import convolve
from concurrent import futures

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
    if phase_wrap:
        z = np.exp(1j*Y)
        args = []
        for i in range(-(N>>1),N-(N>>1)):
            args.append((i, z))

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
            args.append((i, Y))
        with futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            jobs = executor.map(_parallel_shift,args)
            results = list(jobs)# each is N, Nt but rolled
        mean = results[0].copy()
        for r in results[1:]:
            mean += r
        mean /= N
        var = (results[0] - mean)**2
        for r in results[1:]:
            var += (r - mean)**2
        var /= N
    var = np.maximum(min_uncert**2, var)
    return 1./var
    
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
            inv_map.append((O,B-O))
        else:
            val_blocks.append((blocks[-1][0] + O, blocks[-1][1] - O))
            inv_map.append((O,B))
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

