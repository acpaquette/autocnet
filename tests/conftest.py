import numpy as np
import pandas as pd
import pytest

import os
import sys

sys.path.insert(0, '..')

from autocnet.graph.network import CandidateGraph

@pytest.fixture(scope='module')
def candidategraph():
    a = '/fake_path/a.img'
    b = '/fake_path/b.img'
    c = '/fake_path/c.img'
    adj = {a:[b,c],
           b:[a,c],
           c:[a, b]}
    cg = CandidateGraph.from_adjacency(adj)

    match_data = np.array([[0.0,	188.0,	1.0, 0.0, 170.754211],
                           [0.0,	189.0,	1.0, 0.0, 217.451141],
                           [0.0,	185.0,	1.0, 1.0, 108.843925]])
    matches = pd.DataFrame(match_data, columns=['source_image', 'source_idx',
                                                'destination_image', 'destination_idx',
                                                'distance'])
    masks = pd.DataFrame([[True, True],
                          [False, True],
                          [True, False]],
                          columns=['rain', 'maker'])

    for s, d, e in cg.edges.data('data'):
        e['fundamental_matrix'] = np.random.random(size=(3,3))
        e.matches = matches
        e.masks = masks
        e['source_mbr'] = [[0,1], [0,2]]
        e['destin_mbr'] = [[0.5, 0.5], [1, 1]]

    kps = np.array([[233.358475,	105.288162,	0.035672, 4.486887,	164.181046,	0.0, 1.0],
                    [366.288116,	98.761131,	0.035900, 4.158592,	147.278580,	0.0, 1.0],
                    [170.932114,	114.173912,	0.037852, 94.446655, 0.401794,	0.0, 3.0]])

    keypoints = pd.DataFrame(kps, columns=['x', 'y', 'response', 'size', 'angle',
                                           'octave', 'layer'])

    for i, n in cg.nodes.data('data'):
        n.keypoints = keypoints
        n.descriptors = np.random.random(size=(3, 128))
        n.masks = masks

    return cg
