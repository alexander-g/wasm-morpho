import sys
sys.path.insert(0, './build/')
import traininglib_cpp_ext

import numpy as np

import time

def test_dfs():
    x0 = np.zeros([1000,1000], dtype=bool)
    x0[100:200,100:150] = 1
    x0[200:300,200:220] = 1
    x0[300:400, 300] = 1 # single line

    out0 = traininglib_cpp_ext.dfs(x0, (100,100))
    assert out0['visited'].shape == (100*50, 2)
    assert out0['visited'].min(0).tolist() == [100,100]
    assert out0['visited'].max(0).tolist() == [199,149]
    assert out0['predecessors'].shape == (len(out0['visited']),)
    assert out0['predecessors'].max() <= len(out0['visited'])
    assert out0['predecessors'][0] == -1
    assert out0['predecessors'][1] == 0

    # begining of the single line
    out1 = traininglib_cpp_ext.dfs(x0, (300,300))
    assert out1['leaves'].tolist() == [0,99]

    # in the middle of the single line
    out2 = traininglib_cpp_ext.dfs(x0, (333,300))
    assert len(out2['leaves']) == 3



def test_concom():
    x0 = np.zeros([1000,1000], dtype=bool)
    x0[100:200,100:150] = 1
    x0[200:300,200:220] = 1
    x0[300:400, 300] = 1 # single line
    x0[500,500] = 1 # single point

    t0 = time.time()
    out0 = traininglib_cpp_ext.connected_components(x0)
    t1 = time.time()
    print(t1-t0)

    assert out0.max() == 4
    assert len( np.unique(out0) ) == 5
    assert len( np.unique( out0[200:300, 200:220] ) ) == 1

