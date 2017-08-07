import networkx as nx
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

from plio.io.io_controlnetwork import to_isis, write_filelist


def identify_potential_overlaps(cg, cn, overlap=True):
    """
    Identify those points that could have additional measures

    Parameters
    ----------
    overlap : boolean
              If True, apply aprint(g)n additional point in polygon check, where
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

def deepen_correspondences(cg, cn):
    pass

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

    def to_isis(self, outname, serials, olist, *args, **kwargs):
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
