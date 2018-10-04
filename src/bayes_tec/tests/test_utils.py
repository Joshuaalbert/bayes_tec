from ..utils.testing_utils import make_example_datapack
import numpy as np
import os

#def test_define_subsets():
#    X_t = np.linspace(0,800,100)[:,None]
#    overlap = 32
#    max_block_size = 20
#    starts, stops = define_subsets(X_t, overlap, max_block_size)
#    assert (X_t[stops,0]-X_t[starts,0] > overlap).all()
#    assert 0 in starts
#    assert X_t.shape[0] - 1 in stops
#
#
#    
#    overlap = 32
#    max_block_size = 200
#    starts, stops = define_subsets(X_t, overlap, max_block_size)
#    assert (X_t[stops,0]-X_t[starts,0] > overlap).all()
#    assert 0 in starts
#    assert X_t.shape[0] - 1 in stops
#
#    
#    
#    
#    overlap = 32
#    max_block_size = 100
#    starts, stops = define_subsets(X_t, overlap, max_block_size)
#    assert (X_t[stops,0]-X_t[starts,0] > overlap).all()
#    assert 0 in starts
#    assert X_t.shape[0] - 1 in stops

    
def test_make_example_datapack():
    datapack = make_example_datapack(10,4, 100, name='datapack_test_utils.hdf5')
    with datapack:
        assert datapack.phase[0].shape==(10, 62, 4, 100)
    os.unlink('datapack_test_utils.hdf5')

#def test_weights():
#    Y = 5*np.random.normal(size=[10,1000])
#    weights = calculate_weights(Y,N=100,phase_wrap=True)
#    assert not np.any(np.isnan(weights))
#    assert np.all(weights>0)
#    weights = calculate_weights(Y,N=100,phase_wrap=False)
#    assert not np.any(np.isnan(weights))
#    assert np.all(weights>0)


from bayes_tec.utils.data_utils import make_coord_array
import numpy as np

def test_make_coord_array():
    a,b,c,d = np.random.normal(size=[4,10])
    a,b,c,d = a[:,None],b[:,None],c[:,None],d[:,None]
    X1 = make_coord_array(a,b,c,d,flat=True)
    X2 = make_coord_array(a,b,c,d,flat=False)

    idx = np.ravel_multi_index([1,2,3,4],[10,10,10,10])

    assert np.all(X2[1,2,3,4,:] == np.array([a[1],b[2],c[3],d[4]]).flatten())
    assert np.all(X1[idx,:] == X2[1,2,3,4,:])
    
    a = np.random.normal(size=[10,2])
    b = np.random.normal(size=[10,1])
    
    X1 = make_coord_array(a,b,flat=True)
    X2 = make_coord_array(a,b,flat=False)

    idx = np.ravel_multi_index([1,2],[10,10])

    assert np.all(X2[1,2,:2] == a[1])
    assert np.all(X2[1,2,2:3] == b[2])
    assert np.all(X1[idx,:] == X2[1,2,:])
    
    ###
    # no freq test
    a = np.random.normal(size=[10,2])
    b = np.random.normal(size=[10,1])
    c = np.random.normal(size=[10,1])
    X1 = make_coord_array(a,b,c,flat=True)[..., [0,1, 3]]
    X2 = make_coord_array(a,b,c,flat=False)[..., [0,1, 3]]

    idx = np.ravel_multi_index([1,2,3],[10,10,10])

    assert np.all(X1[idx,:] == X2[1,:,3,:])
