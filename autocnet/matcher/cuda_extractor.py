import warnings

try:
    import cudasift as cs
except:
    cs = None
    
def extract_features(array, nfeatures=None, **kwargs):
    """
    A custom docstring.
    """
    if not nfeatures:
        nfeatures = int(max(array.shape) / 1.75)
    else:
        warnings.warn('NFeatures specified with the CudaSift implementation.  Please ensure the distribution of keypoints is what you expect.')

    siftdata = cs.PySiftData(nfeatures)
    cs.ExtractKeypoints(array, siftdata, **kwargs)
    keypoints, descriptors = siftdata.to_data_frame()
    keypoints = keypoints[['x', 'y', 'scale', 'sharpness', 'edgeness', 'orientation', 'score', 'ambiguity']]
    # Set the columns that have unfilled values to zero to avoid confusion
    keypoints['score'] = 0.0
    keypoints['ambiguity'] = 0.0

    return keypoints, descriptors
