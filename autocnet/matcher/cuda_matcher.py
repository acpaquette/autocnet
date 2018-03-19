import warnings

try:
    import cudasift as cs
except:
    cs = None

import numpy as np
import pandas as pd

def match(edge, aidx=None, bidx=None, **kwargs):

    """
    Apply a composite CUDA matcher and ratio check.  If this method is used,
    no additional ratio check is necessary and no symmetry check is required.
    The ratio check is embedded on the cuda side and returned as an
    ambiguity value.  In testing symmetry is not required as it is expensive
    without significant gain in accuracy when using this implementation.
    """

    source_kps = edge.source.get_keypoints(index=aidx)
    source_des = edge.source.descriptors[aidx]
    source_map = {k:v for k, v in enumerate(source_kps.index)}

    destin_kps = edge.destination.get_keypoints(index=bidx)
    destin_des = edge.destination.descriptors[bidx]
    destin_map = {k:v for k, v in enumerate(destin_kps.index)}

    s_siftdata = cs.PySiftData.from_data_frame(source_kps, source_des)
    d_siftdata = cs.PySiftData.from_data_frame(destin_kps, destin_des)


    cs.PyMatchSiftData(s_siftdata, d_siftdata)
    matches, _ = s_siftdata.to_data_frame()
    # Matches are reindexed 0-n, but need to be remapped to the source_kps,
    #  destin_kps indices.  This is the mismatch)
    source = np.empty(len(matches))
    source[:] = edge.source['node_id']
    destination = np.empty(len(matches))
    destination[:] = edge.destination['node_id']

    df = pd.concat([pd.Series(source), pd.Series(matches.index),
            pd.Series(destination), matches.match,
            matches.score, matches.ambiguity], axis=1)
    df.columns = ['source_image', 'source_idx', 'destination_image',
            'destination_idx', 'score', 'ambiguity']


    df.source_idx = df.source_idx.map(source_map)
    df.destination_idx = df.destination_idx.map(destin_map)
    # Set the matches and set the 'ratio' (ambiguity) mask
    edge.matches = df
