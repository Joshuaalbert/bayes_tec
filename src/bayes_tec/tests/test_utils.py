from ..utils.testing_utils import make_example_datapack
from ..utils.data_utils import define_subsets, calculate_weights
import numpy as np
import os

def test_define_subsets():
    X_t = np.linspace(0,800,100)[:,None]
    overlap = 32
    max_block_size = 20
    edges, blocks = define_subsets(X_t, overlap, max_block_size)
    for b in blocks:
        assert b[1] - b[0] >= 3
    assert len(edges)-1 in [b[1] for b in blocks]

def test_make_example_datapack():
    datapack = make_example_datapack(10,4, 100, name='datapack_test_utils.hdf5')
    with datapack:
        assert datapack.phase[0].shape==(10, 62, 4, 100)
    os.unlink('datapack_test_utils.hdf5')

def test_weights():
    Y = 5*np.random.normal(size=[10,1000])
    weights = calculate_weights(Y,N=100,phase_wrap=True)
    assert not np.any(np.isnan(weights))
    assert np.all(weights>0)
    weights = calculate_weights(Y,N=100,phase_wrap=False)
    assert not np.any(np.isnan(weights))
    assert np.all(weights>0)

