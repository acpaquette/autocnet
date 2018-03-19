def distance_ratio(edge, matches, ratio=0.8):
    """
    Compute and return a mask for a matches dataframe
    using Lowe's ratio test.  If keypoints have a single
    Lowe (2004) [Lowe2004]_

    Parameters
    ----------
    ratio : float
            the ratio between the first and second-best match distances
            for each keypoint to use as a bound for marking the first keypoint
            as "good". Default: 0.8

    single : bool
             If True, points with only a single entry are included (True)
             in the result mask, else False.

    Returns
    -------
    mask : pd.dataframe
           A Pandas DataFrame mask for the matches with those failing the
           ratio test set to False.
    """

    return matches['ambiguity'] <= ratio
