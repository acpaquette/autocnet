import os
import sys
import itertools

from time import gmtime, strftime

import pytest

from unittest.mock import Mock, MagicMock


import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath('..'))

from autocnet.control import control


@pytest.fixture
def controlnetwork_data():
    df = pd.DataFrame([[0, 0.0, 0.0, (0.0, 1.0), 0, 0.0, 0.0],
                             [0, 1.0, 0.0, (0.0, 1.0), 0, 0.0, 0.0],
                             [1, 0.0, 1.0, (0.0, 1.0), 1, 0.0, 0.0],
                             [1, 1.0, 1.0, (0.0, 1.0), 1, 0.0, 0.0],
                             [2, 0.0, 2.0, (0.0, 1.0), 2, 0.0, 0.0],
                             [2, 1.0, 2.0, (0.0, 1.0), 2, 0.0, 0.0],
                             [3, 0.0, 3.0, (0.0, 1.0), 3, 0.0, 0.0],
                             [3, 1.0, 3.0, (0.0, 1.0), 3, 0.0, 0.0],
                             [4, 0.0, 4.0, (0.0, 1.0), 4, 0.0, 0.0],
                             [4, 1.0, 4.0, (0.0, 1.0), 4, 0.0, 0.0],
                             [5, 0.0, 5.0, (0.0, 1.0), 5, 0.0, 0.0],
                             [5, 1.0, 5.0, (0.0, 1.0), 5, 0.0, 0.0],
                             [6, 0.0, 6.0, (0.0, 1.0), 6, 0.0, 0.0],
                             [6, 1.0, 6.0, (0.0, 1.0), 6, 0.0, 0.0],
                             [7, 0.0, 7.0, (0.0, 1.0), 7, 0.0, 0.0],
                             [7, 1.0, 7.0, (0.0, 1.0), 7, 0.0, 0.0],
                             [0, 2.0, 0.0, (0.0, 2.0), 0, 0.0, 0.0],
                             [1, 2.0, 1.0, (0.0, 2.0), 1, 0.0, 0.0],
                             [2, 2.0, 2.0, (0.0, 2.0), 2, 0.0, 0.0],
                             [3, 2.0, 3.0, (0.0, 2.0), 3, 0.0, 0.0],
                             [4, 2.0, 4.0, (0.0, 2.0), 4, 0.0, 0.0],
                             [5, 2.0, 5.0, (0.0, 2.0), 5, 0.0, 0.0],
                             [8, 0.0, 8.0, (0.0, 2.0), 6, 0.0, 0.0],
                             [8, 2.0, 8.0, (0.0, 2.0), 6, 0.0, 0.0],
                             [9, 0.0, 9.0, (0.0, 2.0), 7, 0.0, 0.0],
                             [9, 2.0, 9.0, (0.0, 2.0), 7, 0.0, 0.0],
                             [10, 1.0, 8.0, (1.0, 2.0), 6, 0.0, 0.0],
                             [10, 2.0, 6.0, (1.0, 2.0), 6, 0.0, 0.0],
                             [11, 1.0, 9.0, (1.0, 2.0), 7, 0.0, 0.0],
                             [11, 2.0, 7.0, (1.0, 2.0), 7, 0.0, 0.0]],
                            columns=['point_id', 'image_index', 'keypoint_index',
                                     'edge', 'match_idx', 'x', 'y'])

    df.index.name = 'measure_id'

    #Fix types
    df['point_id'] = df['point_id'].astype(object)
    df['match_idx'] = df['match_idx'].astype(object)

    return df

@pytest.fixture
def candidategraph():
    edges = [(0,1), (0,2), (1,2)]


    match_indices = [([0,1,2,3,4,5,6,7], [0,1,2,3,4,5,6,7]),
                     ([0,1,2,3,4,5,8,9], [0,1,2,3,4,5,8,9]),
                     ([0,1,2,3,4,5,8,9], [0,1,2,3,4,5,6,7])]

    matches = []
    for i, e in enumerate(edges):
        c = match_indices[i]
        source_image = np.repeat(e[0], 8)
        destin_image = np.repeat(e[1], 8)
        coords = np.zeros(8)
        data = np.vstack((source_image, c[0], destin_image, c[1],
                          coords, coords, coords, coords)).T
        matches_df = pd.DataFrame(data, columns=['source_image', 'source_idx', 'destination_image', 'destination_idx',
                                                 'source_x', 'source_y', 'destination_x', 'destination_y'])
        matches.append(matches_df)
    # Mock in the candidate graph
    cg = MagicMock()
    cg.get_matches = MagicMock(return_value=matches)

    return cg

@pytest.fixture()
def controlnetwork(controlnetwork_data):
    cn = control.ControlNetwork()
    cn.data = controlnetwork_data
    # Patching data this way does NOT update the internal _measure_id and _point_id attributes
    return cn

@pytest.fixture()
def bad_controlnetwork(controlnetwork_data):
    cn = control.ControlNetwork()
    cn.data = controlnetwork_data
    # Since the data is being patched in, fix the measure counter
    cn._measure_id = len(cn.data) + 1
    # Add a duplicate measure in image 0 to point 0
    cn.add_measure((0,11), (0,1), 2, [1,1], point_id=0)
    return cn

def test_fromcandidategraph(candidategraph, controlnetwork_data):#, controlnetwork):
    matches = candidategraph.get_matches()
    cn = control.ControlNetwork.from_candidategraph(matches)
    assert cn.data.equals(controlnetwork_data)

def test_add_measure():
    cn = control.ControlNetwork()

    # Add the point 0 from image 0
    key = (0,0)
    cn.add_measure(key, (0,1), 3, [1,1])
    assert key in cn.measure_to_point.keys()
    assert cn.measure_to_point[key] == 0

    # Add the point 1 from image 2
    key = (1,2)
    cn.add_measure(key, (0,1), 2, [1,1])
    assert key in cn.measure_to_point.keys()

    # Add another measure associated with point (0,0)
    #  Key is the source and this methods is called to add the destination
    key = (2,1)
    cn.add_measure(key, (0,2), 3, [1,1], point_id = 0)
    assert key in cn.measure_to_point.keys()
    assert cn.measure_to_point[key] == 0

def test_validate_points(controlnetwork):
    assert controlnetwork.validate_points().any()

def test_bad_validate_points(bad_controlnetwork):
    assert bad_controlnetwork.validate_points().iloc[0] == False
    assert bad_controlnetwork.validate_points().iloc[1:].all()

def test_potential_overlap(controlnetwork):
    pass
