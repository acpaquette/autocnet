from collections import defaultdict
import itertools
import math
import os
from time import gmtime, strftime
import warnings

import networkx as nx
import geopandas as gpd
import pandas as pd
import shapely.affinity
import shapely.geometry
import shapely.wkt as swkt
import shapely.ops

from plio.io import io_hdf, io_json
from plio.utils import utils as io_utils
from plio.io.io_gdal import GeoDataset
from autocnet.cg.cg import geom_mask
from autocnet.cg.cg import compute_voronoi
from autocnet.graph import markov_cluster
from autocnet.graph.edge import Edge
from autocnet.graph.node import Node
from autocnet.io import network as io_network
from autocnet.vis.graph_view import plot_graph, cluster_plot

# The total number of pixels squared that can fit into the keys number of GB of RAM for SIFT.
MAXSIZE = {0:None,
           2:6250,
           4:8840,
           8:12500,
           12:15310}

class CandidateGraph(nx.Graph):
    """
    A NetworkX derived directed graph to store candidate overlap images.

    Attributes
    ----------

    node_counter : int
                   The number of nodes in the graph.
    node_name_map : dict
                    The mapping of image labels (i.e. file base names) to their
                    corresponding node indices

    clusters : dict
               of clusters with key as the cluster id and value as a
               list of node indices

    cn : object
         A control network object instantiated by calling generate_cnet.
    ----------
    """
    edge_attr_dict_factory = Edge

    def __init__(self, *args, basepath=None, **kwargs):
        super(CandidateGraph, self).__init__(*args, **kwargs)
        self.graph['node_counter'] = 0
        node_labels = {}
        self.graph['node_name_map'] = {}

        for node_name in self.nodes():
            image_name = os.path.basename(node_name)
            image_path = node_name
            # Replace the default attr dict with a Node object
            self.node[node_name] = Node(image_name, image_path, self.graph['node_counter'])

            # fill the dictionary used for relabelling nodes with relative path keys
            node_labels[node_name] = self.graph['node_counter']
            # fill the dictionary used for mapping base name to node index
            self.graph['node_name_map'][self.node[node_name]['image_name']] = self.graph['node_counter']
            self.graph['node_counter'] += 1

        nx.relabel_nodes(self, node_labels, copy=False)

        for s, d in self.edges():
            if s > d:
                s, d = d, s
            e = self.edge[s][d]
            e.source = self.node[s]
            e.destination = self.node[d]

        self.graph['creationdate'] = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        self.graph['modifieddate'] = strftime("%Y-%m-%d %H:%M:%S", gmtime())

    def __eq__(self, other):
        eq = True
        # Check the nodes
        for n in self.nodes_iter():
            if not self.node[n] == other.node[n]:
                eq = False
        for s, d in self.edges_iter():
            if not self.edge[s][d] == other.edge[s][d]:
                eq = False
        return eq

    @property
    def maxsize(self):
        if not hasattr(self, '_maxsize'):
            self._maxsize = MAXSIZE[0]
        return self._maxsize

    @maxsize.setter
    def maxsize(self, value):
        if not value in MAXSIZE.keys():
            raise KeyError('Value must be in {}'.format(','.join(map(str,MAXSIZE.keys()))))
        else:
            self._maxsize = MAXSIZE[value]


    @classmethod
    def from_filelist(cls, filelist, basepath=None):
        """
        Instantiate the class using a filelist as a python list.
        An adjacency structure is calculated using the lat/lon information in the
        input images. Currently only images with this information are supported.

        Parameters
        ----------
        filelist : list
                   A list containing the files (with full paths) to construct an adjacency graph from

        Returns
        -------
        : object
          A Network graph object
        """
        if isinstance(filelist, str):
            filelist = io_utils.file_to_list(filelist)
        # TODO: Reject unsupported file formats + work with more file formats
        if basepath:
            datasets = [GeoDataset(os.path.join(basepath, f)) for f in filelist]
        else:
            datasets = [GeoDataset(f) for f in filelist]

        # This is brute force for now, could swap to an RTree at some point.
        adjacency_dict = {}
        valid_datasets = []

        for i in datasets:
            adjacency_dict[i.file_name] = []

            fp = i.footprint
            if fp and fp.IsValid():
                valid_datasets.append(i)
            else:
                warnings.warn('Missing or invalid geospatial data for {}'.format(i.base_name))

        # Grab the footprints and test for intersection
        for i, j in itertools.permutations(valid_datasets, 2):
            i_fp = i.footprint
            j_fp = j.footprint

            try:
                if i_fp.Intersects(j_fp):
                    adjacency_dict[i.file_name].append(j.file_name)
                    adjacency_dict[j.file_name].append(i.file_name)
            except:
                warnings.warn('Failed to calculated intersection between {} and {}'.format(i, j))

        return cls(adjacency_dict)

    @classmethod
    def from_adjacency(cls, input_adjacency, basepath=None):
        """
        Instantiate the class using an adjacency dict or file. The input must contain relative or
        absolute paths to image files.

        Parameters
        ----------
        input_adjacency : dict or str
                          An adjacency dictionary or the name of a file containing an adjacency dictionary.

        Returns
        -------
         : object
           A Network graph object

        Examples
        --------
        >>> from autocnet.examples import get_path
        >>> inputfile = get_path('adjacency.json')
        >>> candidate_graph = network.CandidateGraph.from_adjacency(inputfile)
        """
        if not isinstance(input_adjacency, dict):
            input_adjacency = io_json.read_json(input_adjacency)
        if basepath is not None:
            for k, v in input_adjacency.items():
                input_adjacency[k] = [os.path.join(basepath, i) for i in v]
                input_adjacency[os.path.join(basepath, k)] = input_adjacency.pop(k)
        return cls(input_adjacency)

    def _update_date(self):
        """
        Update the last modified date attribute.
        """
        self.graph['modifieddate'] = strftime("%Y-%m-%d %H:%M:%S", gmtime())

    def get_name(self, node_index):
        """
        Get the image name for the given node.

        Parameters
        ----------
        node_index : int
                     The index of the node.

        Returns
        -------
         : str
           The name of the image attached to the given node.


        """
        return self.node[node_index]['image_name']

    def get_matches(self, clean_keys=[]):
        """
        For each edge get all valid matches, masked by the clean_keys.

        Parameters
        ----------
        clean_keys: list
                    of masks to use

        Returns
        -------
        matches : list
                  of matches dataframes
        """
        matches = []
        for s, d, e in self.edges_iter(data=True):
            match, _ = e.clean(clean_keys=clean_keys)
            match = match[['source_image', 'source_idx',
                           'destination_image', 'destination_idx']]
            skps = e.get_keypoints('source', index=match.source_idx)
            skps.columns = ['source_x', 'source_y']
            dkps = e.get_keypoints('destination', index=match.destination_idx)
            dkps.columns = ['destination_x', 'destination_y']
            match = match.join(skps, on='source_idx')
            match = match.join(dkps, on='destination_idx')
            matches.append(match)
        return matches

    def get_matches(self, clean_keys=[]):
        matches = []
        for s, d, e in self.edges_iter(data=True):
            match, _ = e.clean(clean_keys=clean_keys)
            match = match[['source_image', 'source_idx',
                           'destination_image', 'destination_idx']]
            skps = e.get_keypoints('source', index=match.source_idx)
            skps.columns = ['source_x', 'source_y']
            dkps = e.get_keypoints('destination', index=match.destination_idx)
            dkps.columns = ['destination_x', 'destination_y']
            match = match.join(skps, on='source_idx')
            match = match.join(dkps, on='destination_idx')
            matches.append(match)
        return matches

    def add_image(self, *args, **kwargs):
        """
        Adds an image node to the graph.

        Parameters
        ----------

        """

        raise NotImplementedError

    def extract_features(self, band=1, *args, **kwargs):  # pragma: no cover
        """
        Extracts features from each image in the graph and uses the result to assign the
        node attributes for 'handle', 'image', 'keypoints', and 'descriptors'.
        """
        for i, node in self.nodes_iter(data=True):
            array = node.geodata.read_array(band=band)
            node.extract_features(array, *args, **kwargs),

    def extract_features_with_downsampling(self, downsample_amount=None, *args, **kwargs): # pragma: no cover
        """
        Extract interest points from a downsampled array.  The array is downsampled
        by the downsample_amount keyword using the Lanconz downsample amount.  If the
        downsample keyword is not supplied, compute a downsampling constant as the
        total array size divided by the network maxsize attribute.

        Parameters
        ----------

        downsample_amount : int
                            The amount of downsampling to apply to the image
        """
        for i, node in self.nodes_iter(data=True):
            if downsample_amount == None:
                total_size = node.geodata.raster_size[0] * node.geodata.raster_size[1]
                downsample_amount = math.ceil(total_size / self.maxsize**2)
            node.extract_features_with_downsampling(downsample_amount, *args, **kwargs)

    def extract_features_with_tiling(self, tilesize=1000, overlap=500, *args, **kwargs): #pragma: no cover
        for i, node in self.nodes_iter(data=True):
            print('Processing {}'.format(node['image_name']))
            node.extract_features_with_tiling(tilesize=tilesize, overlap=overlap, *args, **kwargs)

    def extract_subsets(self, *args, **kwargs):
        """
        Extracts features from each image in those regions estimated to be
        overlapping.

        *args and **kwargs are passed to the feature extractor.  For example,
        passing method='sift' will cause the extractor to use the sift method.
        """
        for source, destination, e in self.edges_iter(data=True):
            e.extract_subset(*args, **kwargs)


    def save_features(self, out_path, nodes=[], **kwargs):
        """

        Save the features (keypoints and descriptors) for the
        specified nodes.

        Parameters
        ----------
        out_path : str
                   Location of the output file.  If the file exists,
                   features are appended.  Otherwise, the file is created.

        nodes : list
                of nodes to save features for.  If empty, save for all nodes
        """

        for i, n in self.nodes_iter(data=True):
            if nodes and not i in nodes:
                continue
            n.save_features(out_path, **kwargs)

    def load_features(self, in_path, nodes=[], nfeatures=None, **kwargs):
        """
        Load features (keypoints and descriptors) for the
        specified nodes.

        Parameters
        ----------
        in_path : str
                  Location of the input file.

        nodes : list
                of nodes to load features for.  If empty, load features
                for all nodes
        """
        for i, n in self.nodes_iter(data=True):
            if nodes and not i in nodes:
                continue
            else:
                n.load_features(in_path, **kwargs)

    def match(self, *args, **kwargs):
        """
        For all connected edges in the graph, apply feature matching

        See Also
        ----------
        autocnet.graph.edge.Edge.match
        """
        self.apply_func_to_edges('match', *args, **kwargs)

    def decompose_and_match(self, *args, **kwargs):
        """
        For all edges in the graph, apply coupled decomposition followed by
        feature matching.

        See Also
        --------
        autocnet.graph.edge.Edge.decompose_and_match
        """
        self.apply_func_to_edges('decompose_and_match', *args, **kwargs)

    def estimate_mbrs(self, *args, **kwargs):
        """
        For each edge, estimate the overlap and compute a minimum bounding
        rectangle (mbr) in pixel space.

        See Also
        --------
        autocnet.graoh.edge.Edge.compute_mbr
        """
        self.apply_func_to_edges('estimate_mbr', *args, **kwargs)

    def compute_clusters(self, func=markov_cluster.mcl, *args, **kwargs):
        """
        Apply some graph clustering algorithm to compute a subset of the global
        graph.

        Parameters
        ----------
        func : object
               The clustering function to be applied.  Defaults to
               Markov Clustering Algorithm

        args : list
               of arguments to be passed through to the func

        kwargs : dict
                 of keyword arguments to be passed through to the func
        """
        _, self.clusters = func(self, *args, **kwargs)

    def compute_triangular_cycles(self):
        """
        Find all cycles of length 3.  This is similar
         to cycle_basis (networkX), but returns all cycles.
         As opposed to all basis cycles.

        Returns
        -------
        cycles : list
                 A list of cycles in the form [(a,b,c), (c,d,e)],
                 where letters indicate node identifiers

        Examples
        --------
        >>> g = CandidateGraph()
        >>> g.add_edges_from([(0,1), (0,2), (1,2), (0,3), (1,3), (2,3)])
        >>> g.compute_triangular_cycles()
        [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
        """
        cycles = []
        for s, d in self.edges_iter():
            for n in self.nodes():
                if(s,n) in self.edges() and (d,n) in self.edges():
                    cycles.append((s,d,n))
        return cycles

    def minimum_spanning_tree(self):
        """
        Calculates the minimum spanning tree of the graph

        Returns
        -------

         : DataFrame
           boolean mask for edges in the minimum spanning tree
        """

        mst = nx.minimum_spanning_tree(self)
        return self.create_edge_subgraph(mst.edges())

    def apply_func_to_edges(self, function, *args, **kwargs):
        """
        Iterates over edges using an optional mask and and applies the given function.
        If func is not an attribute of Edge, raises AttributeError

        Parameters
        ----------
        function : obj
                   function to be called on every edge

        graph_mask_keys : list
                          of keys in graph_masks
        """
        if not isinstance(function, str):
            function = function.__name__

        for s, d, edge in self.edges_iter(data=True):
            try:
                func = getattr(edge, function)
            except:
                raise AttributeError(function, ' is not an attribute of Edge')
            else:
                func(*args, **kwargs)

    def symmetry_checks(self):
        '''
        Apply a symmetry check to all edges in the graph
        '''
        self.apply_func_to_edges('symmetry_check')

    def ratio_checks(self, *args, **kwargs):
        '''
        Apply a ratio check to all edges in the graph

        See Also
        --------
        autocnet.matcher.cpu_outlier_detector.DistanceRatio.compute
        '''
        self.apply_func_to_edges('ratio_check', *args, **kwargs)

    def compute_overlaps(self, *args, **kwargs):
        '''
        Computes overlap MBRs for all edges
        '''
        self.apply_func_to_edges('compute_overlap', *args, **kwargs)

    def overlap_checks(self, *args, **kwargs):
        '''
        Apply overlap check to all edges in the graph
        '''
        self.apply_func_to_edges('overlap_check', *args, **kwargs)

    def compute_homographies(self, *args, **kwargs):
        '''
        Compute homographies for all edges using identical parameters

        See Also
        --------
        autocnet.graph.edge.Edge.compute_homography
        autocnet.matcher.cpu_outlier_detector.compute_homography
        '''
        self.apply_func_to_edges('compute_homography', *args, **kwargs)

    def compute_fundamental_matrices(self, *args, **kwargs):
        '''
        Compute fundmental matrices for all edges using identical parameters

        See Also
        --------
        autocnet.matcher.cpu_outlier_detector.compute_fundamental_matrix
        '''
        self.apply_func_to_edges('compute_fundamental_matrix', *args, **kwargs)

    def subpixel_register(self, *args, **kwargs):
        '''
        Compute subpixel offsets for all edges using identical parameters

        See Also
        --------
        autocnet.graph.edge.Edge.subpixel_register
        '''
        self.apply_func_to_edges('subpixel_register', *args, **kwargs)

    def suppress(self, *args, **kwargs):
        '''
        Apply a metric of point suppression to the graph

        See Also
        --------
        autocnet.matcher.cpu_outlier_detector.SpatialSuppression
        '''
        self.apply_func_to_edges('suppress', *args, **kwargs)

    def overlap(self):
        '''
        Compute the percentage and area coverage of two images

        See Also
        --------
        autocnet.cg.cg.two_image_overlap
        '''
        self.apply_func_to_edges('overlap')

    def to_filelist(self):
        """
        Generate a file list for the entire graph.

        Returns
        -------

        filelist : list
                   A list where each entry is a string containing the full path to an image in the graph.
        """
        filelist = []
        for i, node in self.nodes_iter(data=True):
            filelist.append(node['image_path'])
        return filelist

    def generate_control_network(self, clean_keys=['fundamental']):
        """
        Generate a correspondence graph from the current candidate graph object.
        The control network is a single graph object, composed of n-sub graphs,
        where each sub-graph is the aggregation of all assocaited correspondences.

        Parameters
        ----------
        clean_keys : list
                     of strings used to mask the matches on each edge of the
                     Candidate Graph object

        """
        return generate_control_network(self)

    def island_nodes(self):
        """
        Finds single nodes that are completely disconnected from the rest of the graph

        Returns
        -------
        : list
          A list of disconnected nodes, nodes of degree zero, island nodes, etc.
        """
        return nx.isolates(self)

    def connected_subgraphs(self):
        """
        Finds and returns a list of each connected subgraph of nodes. Each subgraph is a set.

        Returns
        -------

         : list
           A list of connected sub-graphs of nodes, with the largest sub-graph first. Each subgraph is a set.
        """
        return sorted(nx.connected_components(self), key=len, reverse=True)

    def save(self, filename):
        """
        Save the graph object to disk.
        Parameters
        ----------
        filename : str
                   The relative or absolute PATH where the network is saved
        """
        io_network.save(self, filename)

    def plot(self, ax=None, **kwargs):  # pragma: no cover
        """
        Plot the graph object

        Parameters
        ----------
        ax : object
             A MatPlotLib axes object.

        Returns
        -------
         : object
           A MatPlotLib axes object
        """
        return plot_graph(self, ax=ax, **kwargs)

    def plot_cluster(self, ax=None, **kwargs):  # pragma: no cover
        """
        Plot the graph based on the clusters generated by
        the markov clustering algorithm

        Parameters
        ----------
        ax : object
             A MatPlotLib axes object.

        Returns
        -------
        ax : object
             A MatPlotLib axes object.

        """
        return cluster_plot(self, ax, **kwargs)

    def create_edge_subgraph(self, edges):
        """
        Create a subgraph using a list of edges.
        This is pulled directly from the networkx dev branch.

        Parameters
        ----------
        edges : list
                A list of edges in the form [(a,b), (c,d)] to retain
                in the subgraph

        Returns
        -------
        H : object
            A networkx subgraph object
        """
        H = self.__class__()
        adj = self.adj
        # Filter out edges that don't correspond to nodes in the graph.
        edges = ((u, v) for u, v in edges if u in adj and v in adj[u])
        for u, v in edges:
            # Copy the node attributes if they haven't been copied
            # already.
            if u not in H.node:
                H.node[u] = self.node[u]
            if v not in H.node:
                H.node[v] = self.node[v]
            # Create an entry in the adjacency dictionary for the
            # nodes u and v if they don't exist yet.
            if u not in H.adj:
                H.adj[u] = H.adjlist_dict_factory()
            if v not in H.adj:
                H.adj[v] = H.adjlist_dict_factory()
            # Copy the edge attributes.
            H.edge[u][v] = self.edge[u][v]
            # H.edge[v][u] = self.edge[v][u]
        H.graph = self.graph
        return H

    def size(self, weight=None):
        """
        This replaces the built-in size method to properly
        support Python 3 rounding.

        Parameters
        ----------
        weight : string or None, optional (default=None)
           The edge attribute that holds the numerical value used
           as a weight.  If None, then each edge has weight 1.

        Returns
        -------
        nedges : int
            The number of edges or sum of edge weights in the graph.

        """
        if weight:
            return sum(e[weight] for s, d, e in self.edges_iter(data=True))
        else:
            return len(self.edges())

    def create_node_subgraph(self, nodes):
        """
        Given a list of nodes, create a sub-graph and
        copy both the node and edge attributes to the subgraph.
        Changes to node/edge attributes are propagated back to the
        parent graph, while changes to the graph structure, i.e.,
        the topology, are not.

        Parameters
        ----------
        nodes : iterable
                An iterable (list, set, ndarray) of nodes to subset
                the graph

        Returns
        -------
        H : object
            A networkX graph object

        """
        bunch = set(self.nbunch_iter(nodes))
        # create new graph and copy subgraph into it
        H = self.__class__()

        # copy node and attribute dictionaries
        for n in bunch:
            H.node[n] = self.node[n]
        # namespace shortcuts for speed
        H_adj = H.adj
        self_adj = self.adj
        for i in H.node:
            adj_nodes = set(self.adj[i].keys()).intersection(bunch)
            H.adj[i] = {}
            for j, edge in self.adj[i].items():
                if j in adj_nodes:
                    H.adj[i][j] = edge

        H.graph = self.graph
        return H

    def subgraph_from_matches(self):
        """
        Returns a sub-graph where all edges have matches.
        (i.e. images with no matches are removed)

        Returns
        -------
        : Object
          A networkX graph object
        """

        # get all edges that have matches
        matches = [(u, v) for u, v, edge in self.edges_iter(data=True)
                   if not edge.matches.empty]

        return self.create_edge_subgraph(matches)

    def filter_nodes(self, func, *args, **kwargs):
        """
        Filters graph and returns a sub-graph from matches. Mimics
        python's filter() function

        Parameters
        ----------
        func : function which returns bool used to filter out nodes

        Returns
        -------
        : Object
          A networkX graph object

        """
        nodes = [n for n, d in self.nodes_iter(data=True) if func(d, *args, **kwargs)]
        return self.create_node_subgraph(nodes)

    def filter_edges(self, func, *args, **kwargs):
        """
        Filters graph and returns a sub-graph from matches. Mimics
        python's filter() function

        Parameters
        ----------
        func : function which returns bool used to filter out edges

        Returns
        -------
        : Object
          A networkX graph object
        """
        edges = [(u, v) for u, v, edge in self.edges_iter(data=True) if func(edge, *args, **kwargs)]
        return self.create_edge_subgraph(edges)

    def compute_cliques(self, node_id=None):  # pragma: no cover
        """
        Computes all maximum complete subgraphs for the given graph.
        If a node_id is given, method will return only the complete subgraphs that
        contain that node

        Parameters
        ----------
        node_id : int
                       Integer value for a given node

        Returns
        -------
        : list
          A list of lists of node ids that make up maximum complete subgraphs of the given graph
        """
        if node_id is not None:
            return list(nx.cliques_containing_node(self, nodes=node_id))
        else:
            return list(nx.find_cliques(self))

    def compute_weight(self, clean_keys, **kwargs): # pragma: no cover
        """
        Computes a voronoi weight for each edge in a given graph.
        Can function as is, but is slightly optimized for complete subgraphs.

        Parameters
        ----------
        kwargs : dict
                      keyword arguments that get passed to compute_voronoi

        clean_keys : list
                     Strings used to apply masks to omit correspondences
        """

        if not self.is_complete():
            warnings.warn('The given graph is not complete and may yield garbage.')

        for s, d, edge in self.edges_iter(data=True):
            source_node = edge.source
            overlap, _ = self.compute_intersection(source_node, clean_keys = clean_keys)

            matches, _ = edge.clean(clean_keys)
            kps = edge.get_keypoints(edge.source, index=matches['source_idx'])[['x', 'y']]
            reproj_geom = source_node.reproject_geom(overlap.geometry.values[0].__geo_interface__['coordinates'][0])
            initial_mask = geom_mask(kps, reproj_geom)

            if (len(kps[initial_mask]) <= 0):
                continue

            kps['geometry'] = kps.apply(lambda x: shapely.geometry.Point(x['x'], x['y']), axis=1)
            kps_mask = kps['geometry'][initial_mask].apply(lambda x: reproj_geom.contains(x))
            voronoi_df = compute_voronoi(kps[initial_mask][kps_mask], reproj_geom, **kwargs)

            edge['weights']['voronoi'] = voronoi_df

    def compute_fully_connected_components(self):
        """
        For a given graph, compute all of the fully connected subgraphs with
        3+ components.

        Returns
        -------
        fc : list
             of lists of node identifiers

        Examples
        --------
        >>> G = CandidateGraph()
        >>> G.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'C'), ('B', 'D'), ('A', 'E'), ('A', 'F'), ('E', 'F') ])
        >>> fc = G.fully_connected()
        >>> fc['A']
        [['C', 'B', 'A'], ['A', 'F', 'E']]
        """
        fully_connected = [i for i in nx.enumerate_all_cliques(self) if len(i) > 2]
        fc = defaultdict(list)
        for i in fully_connected:
            for j in i:
                fc[j].append(tuple(i))
        return fc


    def compute_intersection(self, source, clean_keys=[]):
        """
        Computes the intercetion of all images in a graph
        based around a given source node

        Parameters
        ----------
        source: object or int
                     Either a networkx Node object or an integer

        clean_keys : list
                           Strings used to apply masks to omit correspondences

        Returns
        -------
        intersect_gdf : dataframe
                               A geopandas dataframe of intersections for all images
                               that overlap with the source node. Also includes the common
                               overlap for all images in the source node.
        """
        if type(source) is int:
            source = self.node[source]
        # May want to use a try except block here, but what error to raise?
        source_poly = swkt.loads(source.geodata.footprint.GetGeometryRef(0).ExportToWkt())

        source_gdf = gpd.GeoDataFrame({'geometry': [source_poly], 'source_node': [source['node_id']]})

        proj_gdf = gpd.GeoDataFrame(columns=['geometry', 'proj_node'])
        proj_poly_list = []
        proj_node_list = []
        # Begin iterating through the edges in the graph that include the source node
        for s, d, edge in self.edges_iter(data=True):
            if s == source['node_id']:
                proj_poly = swkt.loads(edge.destination.geodata.footprint.GetGeometryRef(0).ExportToWkt())
                proj_poly_list.append(proj_poly)
                proj_node_list.append(d)

            elif d == source['node_id']:
                proj_poly = swkt.loads(edge.source.geodata.footprint.GetGeometryRef(0).ExportToWkt())
                proj_poly_list.append(proj_poly)
                proj_node_list.append(s)

        proj_gdf = gpd.GeoDataFrame({"geometry": proj_poly_list, "proj_node": proj_node_list})
        # Overlay all geometry and find the one geometry element that overlaps all of the images
        intersect_gdf = gpd.overlay(source_gdf, proj_gdf, how='intersection')
        if len(intersect_gdf) == 0:
            raise ValueError('Node ' + str(source['node_id']) +  ' does not overlap with any other images in the candidate graph.')
        overlaps_mask = intersect_gdf.geometry.apply(lambda x:proj_gdf.geometry.contains(shapely.affinity.scale(x, .9, .9)).all())
        overlaps_all = intersect_gdf[overlaps_mask]

        # If there is no intersection polygon that overlaps all of the images, union all of the intersection
        # polygons into one large polygon that does overlap all of the images
        if len(overlaps_all) <= 0:
            new_poly = shapely.ops.unary_union(intersect_gdf.geometry)
            overlaps_all = gpd.GeoDataFrame({'source_node': source['node_id'], 'proj_node': source['node_id'],
                                             'geometry': [new_poly]})

        return overlaps_all, intersect_gdf

    def is_complete(self):
        """
        Checks if the graph is a complete graph
        """
        neighbors_dict = nx.degree(self)
        for value in neighbors_dict.values():
            if value == len(self.neighbors(self.nodes()[0])):
                continue
            else:
                return False

        return True

    def footprints(self):
        geoms = [n.footprint for i, n in self.nodes_iter(data=True)]
        return gpd.GeoDataFrame(geometry=geoms)
