import numpy as np
from scipy.spatial import KDTree


def nearest_neighbor(keypoint, keypoints, edge, source, destination, num_neighbors = 20): 
    """
    Given some keypoint in either source or destination, reproject said keypoint
    into the other image and find the n nearest neighbors using the list of
    keypoints provided. (Keypoint and keypoints should not be from the same image.)

    Parameters
    ----------
    keypoint : tuple
               x, y coordinates for a point in either source or destination

    keypoints : list
                A list of x, y coordinate pairs from either source or destination

    edge : object
           An autocnet edge object

    source : int
             Node id of a source image

    destination : int
                  Node id of a source image

    num_neighbors : int
                    Number of neighbors to find

    Returns
    ----------
    : list
      A list of the n nearest neighbors to the given reprojected point
    """
    if source < destination:
        reproj_keypoint = source_to_dest(keypoint, edge.source, edge.destination, homogeneous=True)
    else:
        reproj_keypoint = dest_to_source(keypoint, edge.source, edge.destination, homogeneous=True)
    kd_tree = KDTree(data = keypoints)
    return kd_tree.query(reproj_keypoint, num_neighbors, eps = 2, p = 1)[1]

def dist_to_epipolar(keypoint, keypoints, edge, source, destination, geometric_threshold=.2):
    """
    Given some keypoint in either source or destination, create an epipolar line
    to use as a search space in the other image. This returns whatever points
    found that are within the geometric threshold specified. (Keypoint and
    keypoints should not be from the same image.)

    Parameters
    ----------
    keypoint : tuple
               x, y coordinates for a point in either source or destination

    keypoints : list
                A list of x, y coordinate pairs from either source or
                destination

    edge : object
           An autocnet edge object

    source : int
             Node id of a source image

    destination : int
                  Node id of a source image

    geometric_threshold : float
                          Defines the size of the search buffer to use along the
                          epipolar line.

    Returns
    ----------
    : list
      A list of points that lie within the geometric threshold of the epipolar
      line.
    """
    if source < destination:
        f_matrix = edge['fundamental_matrix'].T
    else:
        f_matrix = edge['fundamental_matrix']
    epipolar_line = compute_epipolar_line(keypoint, f_matrix)
    bc_distance = np.abs(epipolar_line.dot(keypoints.T))
    return np.where(bc_distance < geometric_threshold)[0]
