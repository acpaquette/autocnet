from unittest.mock import Mock, MagicMock

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from autocnet.control import control
from autocnet.graph.network import CandidateGraph
from autocnet.graph.node import Node
from plio.io.io_gdal import GeoDataset

@pytest.fixture(scope='session')
def candidategraph(geodata_a, geodata_b, geodata_c):
    cg = CandidateGraph()

    # Create a candidategraph object - we instantiate a real CandidateGraph to
    # have access of networkx functionality we do not want to test and then
    # mock all autocnet functionality to control test behavior.
    edges = [(0,1), (0,2), (1,2)]
    cg.add_edges_from(edges)

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

    # Mock in autocnet methods
    cg.get_matches = MagicMock(return_value=matches)

    # Mock in the node objects onto the candidate graph
    cg.node[0] = geodata_a
    cg.node[1] = geodata_b
    cg.node[2] = geodata_c

    return cg

#TODO: Can these be a single parameterized fixture - so much boilerplate!
@pytest.fixture(scope='session')
def geodata_a():
    a = Mock(spec=Node)
    a.geodata = Mock(spec=GeoDataset)
    a.geodata.pixel_to_latlon = MagicMock(side_effect=lambda x, y: (x, y))
    return a

@pytest.fixture(scope='session')
def geodata_b():
    b = Mock(spec=Node)
    b.geodata = Mock(spec=GeoDataset)
    b.geodata.pixel_to_latlon = MagicMock(side_effect=lambda x, y: (x, y))
    return b

@pytest.fixture(scope='session')
def geodata_c():
    c = Mock(spec=Node)
    c.geodata = Mock(spec=GeoDataset)
    c.geodata.pixel_to_latlon = MagicMock(side_effect=lambda x, y: (x, y))
    return c

@pytest.fixture(scope='session')
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

@pytest.fixture(scope='session')
def controlnetwork(controlnetwork_data):
    cn = control.ControlNetwork()
    cn.data = controlnetwork_data
    # Patching data this way does NOT update the internal _measure_id and _point_id attributes
    return cn

@pytest.fixture(scope='session')
def bad_controlnetwork(controlnetwork_data):
    cn = control.ControlNetwork()
    cn.data = controlnetwork_data
    # Since the data is being patched in, fix the measure counter
    cn._measure_id = len(cn.data) + 1
    # Add a duplicate measure in image 0 to point 0
    cn.add_measure((0,11), (0,1), 2, [1,1], point_id=0)
    return cn
