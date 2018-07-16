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
    if len(shape) == 3:
        Nd, Nf, Nt = shape
        weights = []
        for l in range(Nf):
            weights.append(weights_and_mean_uncert(phase[:,l,:],**kwargs))
        return np.stack(weights, axis=1)
    elif len(shape) == 4:
        Nd, Na, Nf, Nt = shape
        weights = []
        for i in range(Na):
            weights_ = []
            for l in range(Nf):
                weights_.append(weights_and_mean_uncert(phase[:,i,l,:],**kwargs))
            weights.append(np.stack(weights_,axis=1))
        return np.stack(weights, axis=1)
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


