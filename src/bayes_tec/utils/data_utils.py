import numpy as np

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
    Y: array shape [Nd, Nt]
    N : int the window
    Returns:
    weights [Nd,Nt] and mean_uncert float
    """
    weights = []
    for k in range(Y.shape[0]):
        dY = Y[k,:]
        if phase_wrap:
            dY = wrap(wrap(dY[:-1]) - wrap(dY[1:]))
        else:
            dY = dY[:-1] - dY[1:]
        dY = np.pad(dY,(0,N),mode='symmetric')
        uncert = np.sqrt(np.convolve(dY**2, np.ones((N,))/N, mode='valid',))
        weights.append(uncert)
    weights = np.stack(weights,axis=0)#uncert
    weights = np.maximum(min_uncert,weights)
    mean_uncert = max(min_uncert,np.mean(weights))
    weights = 1./weights**2
    weights /= np.mean(weights)
    weights[np.isnan(weights)] = 1.
    return weights, mean_uncert

def phase_weights(phase,**kwargs):
    """
    returns the weight matrix for phase
    phase: array [Nd, Na, Nf, Nt] or [Nd, Nf, Nt]
    Returns:
    array same shape as phase
    """
    shape = phase.shape
    uncert_mean = []
    if len(shape) == 3:
        Nd, Nf, Nt = shape
        weights = []
        for l in range(Nf):
            w,u = weights_and_mean_uncert(phase[:,l,:],**kwargs)
            weights.append(w)
            uncert_mean.append(u)
        return np.stack(weights, axis=1), np.mean(uncert_mean)
    elif len(shape) == 4:
        Nd, Na, Nf, Nt = shape
        weights = []
        for i in range(Na):
            weights_ = []
            for l in range(Nf):
                w,u = weights_and_mean_uncert(phase[:,i,l,:],**kwargs)
                weights_.append(w)
                uncert_mean.append(u)
            weights.append(np.stack(weights_,axis=1))
        return np.stack(weights, axis=1), np.mean(uncert_mean)
    else:
        raise ValueError("wrong shape {}".format(shape))

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

