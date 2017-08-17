import warnings

try:
    import cudasift as cs
except:
    cs = None
    
import numpy as np
import pandas as pd

def match(edge, ratio=0.8, **kwargs):

    """
    Apply a composite CUDA matcher and ratio check.  If this method is used,
    no additional ratio check is necessary and no symmetry check is required.
    The ratio check is embedded on the cuda side and returned as an
    ambiguity value.  In testing symmetry is not required as it is expensive
    without significant gain in accuracy when using this implementation.
    """

    source_kps = edge.source.get_keypoints()
    source_des = edge.source.descriptors

    destin_kps = edge.destination.get_keypoints()
    destin_des = edge.destination.descriptors

    s_siftdata = cs.PySiftData.from_data_frame(source_kps, source_des)
    d_siftdata = cs.PySiftData.from_data_frame(destin_kps, destin_des)

    cs.PyMatchSiftData(s_siftdata, d_siftdata)
    matches, _ = s_siftdata.to_data_frame()
    source = np.empty(len(matches))
    source[:] = edge.source['node_id']
    destination = np.empty(len(matches))
    destination[:] = edge.destination['node_id']

    df = pd.concat([pd.Series(source), pd.Series(matches.index),
            pd.Series(destination), matches.match,
            matches.score, matches.ambiguity], axis=1)
    df.columns = ['source_image', 'source_idx', 'destination_image',
            'destination_idx', 'score', 'ambiguity']

    # Set the matches and set the 'ratio' (ambiguity) mask
    edge.matches = df
    edge.masks = pd.DataFrame()
    edge.masks['ratio'] = df['ambiguity'] <= ratio
