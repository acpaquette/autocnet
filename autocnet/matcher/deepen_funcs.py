import numpy as np

def dist_to_epipolar(keypoints, epipolar_line, geometric_threshold):
    bc_distance = np.abs(epipolar_line.dot(keypoints.T))
    return np.where(bc_distance < geometric_threshold)[0]
    # if len(candidate) >= 2:
    #     return candidate[0]
    # else:
    #     return candidate
