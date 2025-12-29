import os
import sys
sys.path.insert(0, './build/')
import morpho_pyext

import numpy as np
import PIL.Image


def test_dfs():
    x0 = np.zeros([1000,1000], dtype=bool)
    x0[100:200,100:150] = 1
    x0[200:300,200:220] = 1
    x0[300:400, 300] = 1 # single line

    out0 = morpho_pyext.dfs(x0, (100,100))
    assert out0['visited'].shape == (100*50, 2)
    assert out0['visited'].min(0).tolist() == [100,100]
    assert out0['visited'].max(0).tolist() == [199,149]
    assert out0['predecessors'].shape == (len(out0['visited']),)
    assert out0['predecessors'].max() <= len(out0['visited'])
    assert out0['predecessors'][0] == -1
    assert out0['predecessors'][1] == 0

    # begining of the single line
    out1 = morpho_pyext.dfs(x0, (300,300))
    assert out1['leaves'].tolist() == [0,99]

    # in the middle of the single line
    out2 = morpho_pyext.dfs(x0, (333,300))
    assert len(out2['leaves']) == 3



def test_concom():
    x0 = np.zeros([1000,1000], dtype=bool)
    x0[100:200,100:150] = 1
    x0[200:300,200:220] = 1
    x0[300:400, 300] = 1 # single line
    x0[500,500] = 1 # single point

    out0 = morpho_pyext.connected_components(x0)

    assert out0.max() == 4
    assert len( np.unique(out0) ) == 5
    assert len( np.unique( out0[200:300, 200:220] ) ) == 1


def test_concom_skel():
    # bug
    skeletonfile0 = os.path.join( os.path.dirname(__file__), 'assets', 'skel0.png' )
    x1 = np.array( PIL.Image.open( skeletonfile0 ).convert('L') ) > 0
    out1 = morpho_pyext.connected_components(x1)
    assert out1.max() == 4



def test_concom_streaming():
    x0 = np.zeros([1000,1000], dtype=bool)
    x0[100:200,100:150] = 1
    x0[200:300,200:220] = 1
    x0[300:400, 300] = 1 # single line
    x0[500,500] = 1 # single point

    out0 = morpho_pyext.connected_components_streaming(x0)

    assert len(out0) == 4
    lengths = sorted([len(out) for out in out0])
    assert lengths == [1, 100, 100*20, 100*50]


    # U-shaped object
    x1 = np.zeros([1000,1000], dtype=bool)
    x1[800:900, 100:900] = 1
    x1[100:900, 100:200] = 1
    x1[100:900, 800:900] = 1

    out1 = morpho_pyext.connected_components_streaming(x1)
    assert len(out1) == 1
    assert len(out1[0]) == x1.sum()


    # bug: diagonal line
    x2 = np.zeros([1000,1000], dtype=bool)
    idxs0 = np.stack([np.arange(10,20)]*2, axis=-1).reshape(-1)
    idxs1 = np.arange(10,30)[::-1]
    x2[idxs0, idxs1] = 1

    out2 = morpho_pyext.connected_components_streaming(x2)
    assert len(out2) == 1
    assert len(out2[0] == x2.sum())

