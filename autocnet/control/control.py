import networkx as nx
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

from plio.io.io_controlnetwork import to_isis, write_filelist
from autocnet.matcher.deepen_funcs import dist_to_epipolar


def identify_potential_overlaps(cg, cn, overlap=True):
    """
    Identify those points that could have additional measures

    Parameters
    ----------
    overlap : bool
              If True, apply an additional point in polygon check, where
              the polygon is the footprint intersection between images and
              the point is a keypoint projected into lat/lon space.  Note
              that the projection can be inaccurate if the method used
              estimates the transformation.

    Returns
    -------
    candidate_cliques : DataFrame
                        with the index as the point id (in the data attribute)
                        and the value as an iterable of image ids to search
                        for a new point.
    """


    fc = cg.compute_fully_connected_components()

    candidate_cliques = []
    geoms = []
    idx = []
    for i, p in cn.data.groupby('point_id'):
        # Which images are covered already.  This finds any connected cycles that
        #  a node is in (this can be more than one - an hourglass network for example)
        # Extract the fully connected subgraph for each covered image in order to
        #  identify which subgraph the measure is in
        covered = p['image_index']
        candidate_cycles = [fc[c] for c in covered]
        cycle = [i for i in candidate_cycles if candidate_cycles.count(i) > 1]
        cycle_to_punch = cycle[0][0]

        # Using the cycles to punch, which images could also be covered?
        uncovered = tuple(set(cycle_to_punch).difference(set(covered)))

        # All candidates are covered, skip this point
        if not uncovered:
            continue

        # Determine whether a 'real' lat/lon are to be used and reproject
        if overlap:
            row = p.iloc[0]
            lat, lon = cg.node[row.image_index].geodata.pixel_to_latlon(row.x, row.y)
        else:
            lat, lon = 0,0

        # Build the data for the geodataframe - can the index be cleaner?
        geoms.append(Point(lon, lat))
        candidate_cliques.append([uncovered, cycle_to_punch])
        idx.append(i)


    candidate_cliques = gpd.GeoDataFrame(candidate_cliques, index=idx,
                                     columns=['candidates', 'subgraph'], geometry=geoms)

    def overlaps(group):
        """
        Take a group, find the subgraph, compute the intersection of footprints
        and apply a group point in polygon check. This is an optimization where
        n-points are intersected with the poly at once (as opposed to the
        single iteration approach.)
        """
        cycle_to_punch = group.subgraph.iloc[0]
        subgraph = cg.create_node_subgraph(cycle_to_punch)
        union, _ = subgraph.compute_intersection(cycle_to_punch[0])#.query('overlaps_all == True')
        intersection = group.intersects(union.unary_union)
        return intersection

    # If the overlap check is going to be used, apply it.
    if overlap:
        candidate_cliques['overlap'] = False
        for i, g in candidate_cliques.groupby('candidates'):
            intersection = overlaps(g)
            candidate_cliques.loc[intersection.index, 'overlap'] = intersection
        return candidate_cliques.query('overlap == True')['candidates']
    else:
         return candidate_cliques.candidates


def deepen_correspondences(cg, overlap, func=dist_to_epipolar, dist = 275, *args, **kwargs):
    """
    Given some cg, find and deepen all points that have the potential to exist
    in more images than the currently do.

    Parameters
    ----------
    cg : object
         Networkx graph object

    overlap : bool
              Boolean to limit the candidates of the method to only points
              that lie within the overlap between two images

    func : function
           Static function that returns keypoint indices

    dist : int
           Threshold to use when comparing descriptor distances

    Returns
    ----------
    all_measures : list
                   A list of keypoint index, (source, destination), and new correspondences pairs
    """
    # TODO: Shouldn't return these values. Should append them to the
    # controlnetwork data dataframe. Or be adjusted as needed.

    # Get the candidates that lie within the overlap as an innitial mask
    candidates = identify_potential_overlaps(cg, cg.controlnetwork, overlap)
    all_measures = []
    # Loop through the candidates
    for i, j in enumerate(candidates):
        # Get the control point, "source", and keypoint for the candidate
        control_point = cg.controlnetwork.data.query('point_id == ' + str(candidates.index[i]))
        source = control_point.loc[control_point.index[0]].image_index
        kp_idx = control_point.loc[control_point.index[0]].keypoint_index
        # Loop over all potential images that should overlap
        for destination in j:
            # Create an edge based off of source and destination
            edge = cg.edge[source][destination]
            if source < destination:
                keypoints = edge.destination.get_keypoint_coordinates(homogeneous = True).values
            else:
                keypoints = edge.source.get_keypoint_coordinates(homogeneous = True).values
            new_correspondence = find_match(kp_idx, keypoints, edge, source, destination, dist, func, *args, **kwargs)
            measure = [kp_idx, (source, destination), new_correspondence]
            all_measures.append(measure)
    return all_measures


def find_match(keypoint_idx, keypoints, edge, source, destination, dist, func, *args, **kwargs):
    """
    Parameters
    ----------
    keypoint_idx : int
                   Index value for a given keypoint

    keypoints : list
                A list of x, y coordinate pairs from either source or
                destination

    edge : object
           networkx edge object

    source : int
             Node id of a source image

    destination : int
                  Node id of a destination image

    dist : int
           Threshold to use when comparing descriptor distances

    func : function
           Static function that takes in at least keypoint(x, y pair),
           keypoints(list of x, y pairs), edge(edge object), source(int),
           destination(int)

    Returns
    ----------
    all_measures : list
                   A list of keypoint index, (source, destination), and new correspondences pairs
    """
    # TODO: Explore possible ways to find points that aren't already defined
    # matches. In other words. Find some way to not return none if nothing
    # from matches is found.

    # TODO: Find some way to clean this up. It's basically duplicated code.

    matches, mask = edge.clean(clean_keys = [])
    # Find potential good matches that were thrown out due to a check
    if source < destination:
        keypoint = edge.source.get_keypoint_coordinates(index = keypoint_idx, homogeneous=True).values
        # Use some function to limit the number of correspondences
        correspondences = func(keypoint, keypoints, edge, source, destination, *args, **kwargs)
        # Check if those correspondences exist as a match within some distance threshold
        kp_id_matches = matches.loc[matches['source_idx'] == keypoint_idx]
        edge_id_matches = kp_id_matches[kp_id_matches['destination_idx'].isin(correspondences)]
        dist_matches = edge_id_matches[edge_id_matches['distance'] < dist]
        # If there is some reasonable correspondence return the found match index
        if any(dist_matches):
            return matches.loc[dist_matches.loc[:, ['distance']].idxmin()].destination_idx.values
        else:
            subpixel_keypoints = edge.destination.get_keypoint_coordinates(index = correspondences).values
    else:
        # Same as above but all calculations are source, destination dependent
        keypoint = edge.destination.get_keypoint_coordinates(index = keypoint_idx, homogeneous=True).values
        correspondences = func(keypoint, keypoints, edge, source, destination, *args, **kwargs)
        kp_id_matches = matches.loc[matches['destination_idx'] == keypoint_idx]
        edge_id_matches = kp_id_matches[kp_id_matches['source_idx'].isin(correspondences)]
        dist_matches = edge_id_matches[edge_id_matches['distance'] < dist]
        if any(dist_matches):
            return matches.loc[dist_matches.loc[:, ['distance']].idxmin()].source_idx.values
        else:
            subpixel_keypoints = edge.source.get_keypoint_coordinates(index = correspondences).values
    return None


class ControlNetwork(object):
    measures_keys = ['point_id', 'image_index', 'keypoint_index', 'edge', 'match_idx', 'x', 'y']

    def __init__(self):
        self._point_id = 0
        self._measure_id = 0
        self.measure_to_point = {}
        self.data = pd.DataFrame(columns=self.measures_keys)

    @classmethod
    def from_candidategraph(cls, matches):
        cls = ControlNetwork()
        for match in matches:
            for idx, row in match.iterrows():
                edge = (row.source_image, row.destination_image)
                source_key = (row.source_image, row.source_idx)
                source_fields = row[['source_x', 'source_y']]
                destin_key = (row.destination_image, row.destination_idx)
                destin_fields = row[['destination_x', 'destination_y']]
                if cls.measure_to_point.get(source_key, None) is not None:
                    tempid = cls.measure_to_point[source_key]
                    cls.add_measure(destin_key, edge, row.name, destin_fields, point_id=tempid)
                elif cls.measure_to_point.get(destin_key, None) is not None:
                    tempid = cls.measure_to_point[destin_key]
                    cls.add_measure(source_key, edge, row.name,  source_fields, point_id=tempid)
                else:
                    cls.add_measure(source_key, edge, row.name,  source_fields)
                    cls.add_measure(destin_key, edge,row.name,  destin_fields)
                    cls._point_id += 1

        cls.data.index.name = 'measure_id'
        return cls

    def add_measure(self, key, edge, match_idx, fields, point_id=None):
        """
        Create a new measure that is coincident to a given point.  This method does not
        create the point if is missing.  When a measure is added to the graph, an associated
        row is added to the measures dataframe.

        Parameters
        ----------
        key : hashable
                  Some hashable id.  In the case of an autocnet graph object the
                  id should be in the form (image_id, match_id)

        point_id : hashable
                   The point to link the node to.  This is most likely an integer, but
                   any hashable should work.
        """
        if key in self.measure_to_point.keys():
            return
        if point_id == None:
            point_id = self._point_id
        self.measure_to_point[key] = point_id
        # The node_id is a composite key (image_id, correspondence_id), so just grab the image
        image_id = key[0]
        match_id = key[1]
        self.data.loc[self._measure_id] = [point_id, image_id, match_id, edge, match_idx, *fields]
        self._measure_id += 1

    def validate_points(self):
        """
        Ensure that all control points currently in the nework are valid.

        Criteria for validity:

          * Singularity: A control point can have one and only one measure from any image

        Returns
        -------
         : pd.Series

        """

        def func(g):
            print(g)
            # One and only one measure constraint
            if not g.image_index.duplicated().any():
                return True
            else: return False

        return self.data.groupby('point_id').apply(func)

    def to_isis(self, outname, serials, olist, *args, **kwargs): #pragma: no cover
        """
        Write the control network out to the ISIS3 control network format.
        """

        if self.validate_points().any() == True:
            warnings.warn('Control Network is not ISIS3 compliant.  Please run the validate_points method on the control network.')
            return

        to_isis(outname + '.net', self.data, serials, *args, **kwargs)
        write_filelist(olist, outname + '.lis')

    def to_bal(self):
        """
        Write the control network out to the Bundle Adjustment in the Large
        (BAL) file format.  For more information see:
        http://grail.cs.washington.edu/projects/bal/
        """
        pass
