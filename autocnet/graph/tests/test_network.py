import os
import time
import sys

import pytest
import unittest

from unittest.mock import patch, PropertyMock, MagicMock

import geopandas as gpd
import numpy as np
from osgeo import ogr
from plio.io import io_gdal

from autocnet.examples import get_path

from .. import network
from .. import node

sys.path.insert(0, os.path.abspath('..'))


@pytest.fixture()
def graph():
    basepath = get_path('Apollo15')
    return network.CandidateGraph.from_adjacency(get_path('three_image_adjacency.json'),
                                                      basepath=basepath)

@pytest.fixture()
def geo_graph():
    basepath = get_path('Apollo15')
    a = 'AS15-M-0297_crop.cub'
    b = 'AS15-M-0298_crop.cub'
    c = 'AS15-M-0299_crop.cub'
    adjacency = {a:[b,c],
                 b:[a,c],
                 c:[a,b]}
    return network.CandidateGraph.from_adjacency(adjacency, basepath=basepath)

@pytest.fixture()
def disconnected_graph():
    return network.CandidateGraph.from_adjacency(get_path('adjacency.json'))


def test_get_name(graph):
    node_number = graph.graph['node_name_map']['AS15-M-0297_SML.png']
    name = graph.get_name(node_number)
    assert name == 'AS15-M-0297_SML.png'


def test_size(graph):
    assert graph.size() == graph.number_of_edges()
    for u, v, e in graph.edges_iter(data=True):
        e['edge_weight'] = 10

    assert graph.size('edge_weight') == graph.number_of_edges()*10


def test_add_image(graph):
    with pytest.raises(NotImplementedError):
        graph.add_image()


def test_island_nodes(disconnected_graph):
    assert len(disconnected_graph.island_nodes()) == 1


def test_triangular_cycles(graph):
    cycles = graph.compute_triangular_cycles()
    # Node order is variable, length is not
    assert len(cycles) == 1


def test_connected_subgraphs(graph, disconnected_graph):
    subgraph_list = disconnected_graph.connected_subgraphs()
    assert len(subgraph_list) == 2

    islands = disconnected_graph.island_nodes()
    assert islands[0] in subgraph_list[1]

    subgraph_list = graph.connected_subgraphs()
    assert len(subgraph_list) == 1


def test_filter(graph):
    graph = graph.copy()
    test_sub_graph = graph.create_node_subgraph([0, 1])

    test_sub_graph.extract_features(extractor_parameters={'nfeatures': 25})
    test_sub_graph.match(k=2)

    filtered_nodes = graph.filter_nodes(lambda node: node.descriptors is not None)
    filtered_edges = graph.filter_edges(lambda edge: edge.matches.empty is not True)

    assert filtered_nodes.number_of_nodes() == test_sub_graph.number_of_nodes()
    assert filtered_edges.number_of_edges() == test_sub_graph.number_of_edges()


def test_subset_graph(graph):
    g = graph
    edge_sub = g.create_edge_subgraph([(0, 2)])
    assert len(edge_sub.nodes()) == 2

    node_sub = g.create_node_subgraph([0, 1])
    assert len(node_sub) == 2


def test_subgraph_from_matches(graph):
    test_sub_graph = graph.create_node_subgraph([0, 1])
    test_sub_graph.extract_features(extractor_parameters={'nfeatures': 25})
    test_sub_graph.match(k=2)

    sub_graph_from_matches = graph.subgraph_from_matches()

    assert test_sub_graph.edges() == sub_graph_from_matches.edges()


def test_minimum_spanning_tree():
    test_dict = {"0": ["4", "2", "1", "3"],
                 "1": ["0", "3", "2", "6", "5"],
                 "2": ["1", "0", "3", "4", "7"],
                 "3": ["2", "0", "1", "5"],
                 "4": ["2", "0"],
                 "5": ["1", "3"],
                 "6": ["1"],
                 "7": ["2"]}

    graph = network.CandidateGraph.from_adjacency(test_dict)
    mst_graph = graph.minimum_spanning_tree()

    assert sorted(mst_graph.nodes()) == sorted(graph.nodes())
    assert len(mst_graph.edges()) == len(graph.edges())-5


def test_fromlist():
    mock_list = ['AS15-M-0295_SML.png', 'AS15-M-0296_SML.png', 'AS15-M-0297_SML.png',
                 'AS15-M-0298_SML.png', 'AS15-M-0299_SML.png', 'AS15-M-0300_SML.png']

    good_poly = ogr.CreateGeometryFromWkt('POLYGON ((30 10, 40 40, 20 40, 10 20, 30 10))')
    bad_poly = ogr.CreateGeometryFromWkt('POLYGON ((9999 10, 40 40, 20 40, 10 20, 30 10))')

    with patch('plio.io.io_gdal.GeoDataset.footprint', new_callable=PropertyMock) as patch_fp:
        patch_fp.return_value = good_poly
        n = network.CandidateGraph.from_filelist(mock_list, get_path('Apollo15'))
        assert n.number_of_nodes() == 6
        assert n.number_of_edges() == 15

        patch_fp.return_value = bad_poly
        n = network.CandidateGraph.from_filelist(mock_list, get_path('Apollo15'))
        assert n.number_of_nodes() == 6
        assert n.number_of_edges() == 0

    n = network.CandidateGraph.from_filelist(mock_list, get_path('Apollo15'))
    assert len(n.nodes()) == 6

    n = network.CandidateGraph.from_filelist(get_path('adjacency.lis'), get_path('Apollo15'))
    assert len(n.nodes()) == 6


def test_apply_func_to_edges(graph):
    graph = graph.copy()
    mst_graph = graph.minimum_spanning_tree()

    try:
        graph.apply_func_to_edges('incorrect_func')
    except AttributeError:
        pass

    mst_graph.extract_features(extractor_parameters={'nfeatures': 50})
    mst_graph.match()
    mst_graph.apply_func_to_edges("symmetry_check")

    # Test passing the func by signature
    mst_graph.apply_func_to_edges(graph[0][1].symmetry_check)

    assert not graph[0][2].masks['symmetry'].all()
    assert not graph[0][1].masks['symmetry'].all()


def test_intersection():
    # Generate the footprints for the mock nodes
    ogr_poly_list = []
    wkt0 = "MULTIPOLYGON (((2.5 7.5,7.5 7.5,7.5 12.5,2.5 12.5,2.5 7.5)))"
    ogr_poly_list.append(ogr.CreateGeometryFromWkt(wkt0))
    wkt1 = "MULTIPOLYGON (((0 10, 5 10, 5 15, 0 15, 0 10)))"
    ogr_poly_list.append(ogr.CreateGeometryFromWkt(wkt1))
    wkt2 = "MULTIPOLYGON (((5.5 5.0,10.5 5.0,10.5 10.0,5.5 10.0,5.5 5.0)))"
    ogr_poly_list.append(ogr.CreateGeometryFromWkt(wkt2))
    wkt3 = "MULTIPOLYGON (((5.5 7.5,10.5 7.5,10.5 12.5,5.5 12.5,5.5 7.5)))"
    ogr_poly_list.append(ogr.CreateGeometryFromWkt(wkt3))
    wkt4 = "MULTIPOLYGON (((8 11,13 11,13 16,8 16,8 11)))"
    ogr_poly_list.append(ogr.CreateGeometryFromWkt(wkt4))
    wkt5 = "MULTIPOLYGON (((8 14,13 14,13 19,8 19,8 14)))"
    ogr_poly_list.append(ogr.CreateGeometryFromWkt(wkt5))
    wkt6 = "MULTIPOLYGON (((11 11,16 11,16 16,11 16,11 11)))"
    ogr_poly_list.append(ogr.CreateGeometryFromWkt(wkt6))
    wkt7 = "MULTIPOLYGON (((11 14,16 14,16 19,11 19,11 14)))"
    ogr_poly_list.append(ogr.CreateGeometryFromWkt(wkt7))

    # Create the graph and all the mocked nodes
    cang = network.CandidateGraph()
    for n in range(0, 8):
        cang.add_node(n)
        new_node = MagicMock(spec=node.Node())
        geodata = MagicMock(spec=io_gdal.GeoDataset)
        new_node.geodata = geodata
        geodata.footprint = ogr_poly_list[n]
        new_node.__getitem__ = MagicMock(return_value=n)
        cang.node[n] = new_node

    # Create the edges between the nodes in the graph
    cang.add_edges_from([(0, 1), (0, 2), (0, 3), (2, 3), (3, 4), (4, 5),
                                            (5, 6), (6, 7), (7, 4), (4, 6), (5, 7)])

    # Define source and destination for each edge
    for s, d in cang.edges():
        if s > d:
            s, d = d, s
        e = cang.edge[s][d]
        e.source = cang.node[s]
        e.destination = cang.node[d]

    overlap, intersect_gdf = cang.compute_intersection(3)

    # Test the correct areas were found for the overlap and
    # the intersect_gdf
    print(overlap.geometry.area)
    assert intersect_gdf.geometry[0].area == 7.5
    assert intersect_gdf.geometry[1].area == 5
    assert intersect_gdf.geometry[2].area == 5
    assert intersect_gdf.geometry[3].area == 3.75
    assert overlap.geometry.area.values == 21.25


def test_set_maxsize(graph):
    maxsizes = network.MAXSIZE
    assert(graph.maxsize == maxsizes[0])
    graph.maxsize = 12
    assert(graph.maxsize == maxsizes[12])
    with pytest.raises(KeyError):
        graph.maxsize = 7


def test_update_data(graph):
   ctime = graph.graph['modifieddate']
   time.sleep(1)
   graph._update_date()
   ntime = graph.graph['modifieddate']
   assert ctime != ntime


def test_is_complete(graph):
    # Create a small incomplete graph with three nodes and two edges
    incomplete_graph = network.CandidateGraph()
    incomplete_graph.add_nodes_from([1, 2, 3])
    incomplete_graph.add_edges_from([(1, 2), (2, 3)])

    assert False == incomplete_graph.is_complete()
    assert True == graph.is_complete()
    
def test_apply(graph):
    def set_matches(x):
        s,d,e = x
        e.matches = ['fake', 'fake', 'fake']

    def get_matches(x):
        s,d,e = x
        return e.matches

    graph.apply(set_matches)
    results = graph.apply(get_matches)

    for matches in results:
        assert len(matches) == 3

def test_footprints(geo_graph):
    # This is just testing the interface - should get a geodataframe back
    assert isinstance(geo_graph.footprints(), gpd.GeoDataFrame)
