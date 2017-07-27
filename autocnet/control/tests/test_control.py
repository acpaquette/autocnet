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
def candidategraph():
    edges = [(0,1), (0,2), (1,2)]

    matches = []
    e = edges[0]
    source_image = np.repeat(e[0], 8)
    destin_image = np.repeat(e[1], 8)
    data = np.vstack((source_image, [0,1,2,3,4,5,6,7], destin_image, [0,1,2,3,4,5,6,7])).T
    matches_df = pd.DataFrame(data, columns=['source_image', 'source_idx', 'destination_image', 'destination_idx'])
    matches.append(matches_df)

    e = edges[1]
    c1 = [0,1,2,3,4,5,8,9]
    c2 = [0,1,2,3,4,5,8,9]
    source_image = np.repeat(e[0], 8)
    destin_image = np.repeat(e[1], 8)
    data = np.vstack((source_image, c1, destin_image, c2)).T
    matches_df = pd.DataFrame(data, columns=['source_image', 'source_idx', 'destination_image', 'destination_idx'])
    matches.append(matches_df)

    e = edges[2]
    c1 = [0,1,2,3,4,5,8,9]
    c2 = [0,1,2,3,4,5,6,7]
    source_image = np.repeat(e[0], 8)
    destin_image = np.repeat(e[1], 8)
    data = np.vstack((source_image, c1, destin_image, c2)).T
    matches_df = pd.DataFrame(data, columns=['source_image', 'source_idx', 'destination_image', 'destination_idx'])
    matches.append(matches_df)

    # Mock in the candidate graph
    cg = MagicMock()
    cg.get_matches = MagicMock(return_value=matches)

    return cg

@pytest.fixture()
def controlnetwork(candidategraph):
    cn =  control.ControlNetwork()
    cn.add_from_matches(candidategraph.get_matches())
    return cn

class TestControlMediator():

    def test_fromcandidategraph(self, candidategraph):
        cm = control.ControlMediator.from_candidategraph(candidategraph)
        test_image_ids(cm._cn)
        test_data_exists(cm._cn)




# Test the Control Object Independent of any Candidate Graph
def test_instantiate(candidategraph):
    cn = control.ControlNetwork()
    cn.add_from_matches(candidategraph.get_matches())
    assert isinstance(cn.data, pd.DataFrame)

def test_data_exists(controlnetwork):
    pt_counts = controlnetwork.data.groupby('point_id').count()
    # Ensure 6 points with 3 measures
    assert len(pt_counts.query('image_index == 3').image_index) == 6

    # Ensure 6 points with 2 measures
    assert len(pt_counts.query('image_index == 2').image_index) == 6

    # Ensure that the measure to point lookup is correct
    assert len(controlnetwork.measure_to_point.keys()) == 30

def test_add_measure(controlnetwork):
    assert False

def test_image_ids(controlnetwork):
    iids = controlnetwork.get_image_ids()
    truth = np.array([0., 1. , 2.])
    np.testing.assert_array_equal(iids, truth)

def test_find_measure_from_points(controlnetwork):
    measures = controlnetwork.data.query('point_id == 3')
    assert len(measures) == 3
    np.testing.assert_array_equal(measures.image_index.unique(), np.array([0,1,2]))
    assert len(measures.keypoint_index.unique()) == 1

def test_find_controlpoint_from_measure(controlnetwork):
    assert False
