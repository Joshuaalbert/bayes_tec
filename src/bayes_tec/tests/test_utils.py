from ..utils.testing_utils import make_example_datapack
from ..utils.data_utils import define_subsets
import numpy as np

def test_define_subsets():
    X_t = np.linspace(0,800,100)[:,None]
    overlap = 32
    max_block_size = 20
    edges, blocks = define_subsets(X_t, overlap, max_block_size)
    for b in blocks:
        assert b[1] - b[0] >= 2

def test_make_example_datapack():
    make_example_datapack(10,20, 100)

