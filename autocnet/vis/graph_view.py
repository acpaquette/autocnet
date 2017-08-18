import numpy as np
import networkx as nx

from matplotlib import pyplot as plt
import matplotlib

from scipy.misc import imresize

def downsample(array, amount):
    return imresize(array,
                      (int(array.shape[0] / amount),
                      int(array.shape[1] / amount)),
                      interp='lanczos')

def plot_graph(graph, ax=None, cmap='Spectral', labels=False, font_size=12, clusters=None, **kwargs):
    """

    Parameters
    ----------
    graph : object
            A networkX or derived graph object

    ax : objext
         A MatPlotLib axes object

    cmap : str
           A MatPlotLib color map string. Default 'Spectral'

    Returns
    -------
    ax : object
         A MatPlotLib axes object. Either the argument passed in
         or a new object
    """
    if ax is None:
        ax = plt.gca()

    cmap = matplotlib.cm.get_cmap(cmap)

    # Setup edge color based on the health metric
    colors = []
    for s, d, e in graph.edges_iter(data=True):
        if hasattr(e, 'health'):
            colors.append(cmap(e.health)[0])
        else:
            colors.append(cmap(0)[0])

    pos = nx.spring_layout(graph)
    nx.draw_networkx_nodes(graph, pos, ax=ax)
    nx.draw_networkx_edges(graph, pos, ax=ax)
    if labels:
        labels = dict((i, d['image_name']) for i, d in graph.nodes_iter(data=True))
        nx.draw_networkx_labels(graph, pos, labels, font_size=font_size, ax=ax)
    ax.axis('off')
    return ax


def plot_node(node, ax=None, clean_keys=[], index_mask=None, downsampling=1, **kwargs):
    """
    Plot the array and keypoints for a given node.

    Parameters
    ----------
    node : object
           A Node object from which data is extracted

    ax : object
         A MatPlotLIb axes object

    clean_keys : list
                 of strings of masking array names to apply

    kwargs : dict
             of MatPlotLib plotting options

    Returns
    -------
    ax : object
         A MatPlotLib axes object.  Either the argument passed in
         or a new object
    """

    if ax is None:
        ax = plt.gca()

    band = 1
    if 'band' in kwargs.keys():
        band = kwargs['band']
        kwargs.pop('band', None)

    array = node.get_array(band)

    if isinstance(downsampling, bool):
        downsampling = node['downsample_amount']

    array = downsample(array, downsampling)

    ax.set_title(node['image_name'])
    ax.margins(tight=True)
    ax.axis('off')

    if 'cmap' in kwargs:
        cmap = kwargs['cmap']
    else:
        cmap = 'Greys'

    ax.imshow(array, cmap=cmap)

    keypoints = node.get_keypoints(index=index_mask)
    # Node has no clean method
    # if clean_keys:
    #     matches, mask = node.clean(clean_keys)
    #     keypoints = keypoints[mask]

    marker = '.'
    if 'marker' in kwargs.keys():
        marker = kwargs['marker']
        kwargs.pop('marker', None)
    color = 'r'
    if 'color' in kwargs.keys():
        color = kwargs['color']
        kwargs.pop('color', None)
    ax.scatter(keypoints['x'], keypoints['y'], marker=marker, color=color, **kwargs)

    return ax


def plot_edge_decomposition(edge, ax=None, clean_keys=[], image_space=100,
                            scatter_kwargs={}, line_kwargs={}, image_kwargs={}):

    if ax is None:
        ax = plt.gca()

    # Plot setup
    ax.set_title('Matching: {} to {}'.format(edge.source['image_name'],
                                             edge.destination['image_name']))
    ax.margins(tight=True)
    ax.axis('off')

    # Image plotting
    source_array = edge.source.get_array()
    destination_array = edge.destination.get_array()

    s_shape = source_array.shape
    d_shape = destination_array.shape

    y = max(s_shape[0], d_shape[0])
    x = s_shape[1] + d_shape[1] + image_space
    composite = np.zeros((y, x))
    composite_decomp = np.zeros((y, x), dtype=np.int16)

    composite[0: s_shape[0], :s_shape[1]] = source_array
    composite[0: d_shape[0], s_shape[1] + image_space:] = destination_array

    composite_decomp[0: s_shape[0], :s_shape[1]] = edge.smembership
    composite_decomp[0: d_shape[0], s_shape[1] + image_space:] = edge.dmembership

    if 'cmap' in image_kwargs:
        cmap = image_kwargs['cmap']
    else:
        cmap = 'Greys'

    matches, mask = edge.clean(clean_keys)

    source_keypoints = edge.source.get_keypoints(index=matches['source_idx'])
    destination_keypoints = edge.destination.get_keypoints(index=matches['destination_idx'])

    # Plot the source
    source_idx = matches['source_idx'].values
    s_kps = source_keypoints.loc[source_idx]
    ax.scatter(s_kps['x'], s_kps['y'], **scatter_kwargs, cmap='gray')

    # Plot the destination
    destination_idx = matches['destination_idx'].values
    d_kps = destination_keypoints.loc[destination_idx]
    x_offset = s_shape[1] + image_space
    newx = d_kps['x'] + x_offset
    ax.scatter(newx, d_kps['y'], **scatter_kwargs)

    ax.imshow(composite, cmap=cmap)
    ax.imshow(composite_decomp, cmap='spectral', alpha=0.35)
    # Draw the connecting lines
    color = 'y'
    if 'color' in line_kwargs.keys():
        color = line_kwargs['color']
        line_kwargs.pop('color', None)

    s_kps = s_kps[['x', 'y']].values
    d_kps = d_kps[['x', 'y']].values
    d_kps[:, 0] += x_offset

    for l in zip(s_kps, d_kps):
        ax.plot((l[0][0], l[1][0]), (l[0][1], l[1][1]), color=color, **line_kwargs)

    return ax

def plot_edge(edge, ax=None, clean_keys=[], image_space=100, downsampling=1,
              scatter_kwargs={}, line_kwargs={}, image_kwargs={}):
    """
    Plot the correspondences for a given edge

    Parameters
    ----------
    edge : object
           A graph edge object

    ax : object
         A MatPlotLIb axes object

    clean_keys : list
                 of strings of masking array names to apply

    image_space : int
                  The number of pixels to insert between the images

    downsample : bool

    scatter_kwargs : dict
                     of MatPlotLib arguments to be applied to the scatter plots

    line_kwargs : dict
                  of MatPlotLib arguments to be applied to the lines connecting matches

    image_kwargs : dict
                   of MatPlotLib arguments to be applied to the image rendering

    Returns
    -------
    ax : object
         A MatPlotLib axes object.  Either the argument passed in
         or a new object
    """

    if ax is None:
        ax = plt.gca()

    # Plot setup
    ax.set_title('Matching: {} to {}'.format(edge.source['image_name'],
                                             edge.destination['image_name']))
    ax.margins(tight=True)
    ax.axis('off')

    # Image plotting
    if isinstance(downsampling, bool):
        downsample_source = edge.source['downsample_amount']
    else:
        downsample_source = downsampling
    source_array = edge.source.get_array()
    source_array = downsample(source_array, downsample_source)

    if isinstance(downsampling, bool):
        downsample_destin = edge.destination['downsample_amount']
    else:
        downsample_destin = downsampling
    destination_array = edge.destination.get_array()
    destination_array = downsample(destination_array, downsample_destin)

    s_shape = source_array.shape
    d_shape = destination_array.shape

    y = max(s_shape[0], d_shape[0])
    x = s_shape[1] + d_shape[1] + image_space
    composite = np.zeros((y, x))

    composite[0: s_shape[0], :s_shape[1]] = source_array
    composite[0: d_shape[0], s_shape[1] + image_space:] = destination_array

    if 'cmap' in image_kwargs:
        image_cmap = image_kwargs['cmap']
    else:
        image_cmap = 'Greys'

    matches, mask = edge.clean(clean_keys)

    source_keypoints = edge.source.get_keypoints(index=matches['source_idx'])
    destination_keypoints = edge.destination.get_keypoints(index=matches['destination_idx'])

    # Plot the source
    source_idx = matches['source_idx'].values
    s_kps = source_keypoints.loc[source_idx]
    ax.scatter(s_kps['x'], s_kps['y'], **scatter_kwargs)

    # Plot the destination
    destination_idx = matches['destination_idx'].values
    d_kps = destination_keypoints.loc[destination_idx]
    x_offset = s_shape[1] + image_space
    newx = d_kps['x'] + x_offset
    ax.scatter(newx, d_kps['y'], **scatter_kwargs)

    ax.imshow(composite, cmap=image_cmap)

    # Draw the connecting lines
    color = 'y'
    if 'color' in line_kwargs.keys():
        color = line_kwargs['color']
        line_kwargs.pop('color', None)

    s_kps = s_kps[['x', 'y']].values
    d_kps = d_kps[['x', 'y']].values
    d_kps[:, 0] += x_offset

    for l in zip(s_kps, d_kps):
        ax.plot((l[0][0], l[1][0]), (l[0][1], l[1][1]), color=color, **line_kwargs)

    return ax


def cluster_plot(graph, ax=None, cmap='Spectral'):  # pragma: no cover
    """
    Parameters
    ----------
    graph : object
            A networkX or derived graph object

    ax : object
         A MatPlotLib axes object

    cmap : str
           A MatPlotLib color map string. Default 'Spectral'

    Returns
    -------
    ax : object
         A MatPlotLib axes object that was either passed in
         or a new axes object
    """
    if ax is None:
        ax = plt.gca()

    if not hasattr(graph, 'clusters'):
        raise AttributeError('Clusters have not been computed.')

    cmap = matplotlib.cm.get_cmap(cmap)

    colors = []

    for i, n in graph.nodes_iter(data=True):
        for j in enumerate(graph.clusters):
            if i in graph.clusters.get(j[1]):
                colors.append(cmap(j[1])[0])
                continue

    nx.draw(graph, ax=ax, node_color=colors)
    return ax

def plot_eline(edge, source, destin, source_kp_idx, destin_kp_idx, ax = None):
    """
    Plots an epipolar line with the source and destination images, along with
    the associated indicies of the keypoints.

    -->NOTE<--: source and destin have to be in the order that the epipolar
    line is computed. If you want an epipolar line from the 1, 0 edge on the 0
    image source must be 1 and destination must be 0. The same is true for the
    source_kp_idx and destin_kp_idx, the source should be from 1 and the
    destination should be from 0
    Parameters
    ----------

    edge : object
           networkx edge object

    source : int
             The node id of the source image

    destination : int
                  The node if of the destination image

    source_kp_idx : int
                    Source keypoint index

    destin_kp_idx : int
                    Destination keypoint index

    ax : object
         A MatPlotLIb axes object

    Returns
    ----------
    ax : object
         A MatPlotLIb axes object. Either the argument passed in
         or a new object
    """
    # TODO: Add better arguments. Bring in line with the rest of vis
    # ALSO: Potentially change the indices to keypoints, may help make things
    # a little easier.
    if ax is None:
        ax = plt.gca()
    # Code "Borrowed" from plot_edge
    # with some hard coded values
    downsample_source = 1
    source_array = edge.source.get_array()

    downsample_destin = 1
    destination_array = edge.destination.get_array()

    s_shape = source_array.shape
    d_shape = destination_array.shape

    y = max(s_shape[0], d_shape[0])
    x = s_shape[1] + d_shape[1] + 100
    composite = np.zeros((y, x))

    composite[0: s_shape[0], :s_shape[1]] = source_array
    composite[0: d_shape[0], s_shape[1] + 100:] = destination_array

    r = lambda: random.uniform(0.0, 1.0)
    color = [r(), r(), r()]

    if not hasattr(source_kp_idx, '__iter__'):
        s_i = np.asarray([source_kp_idx])
    else:
        s_i = source_kp_idx

    if not hasattr(destin_kp_idx, '__iter__'):
        d_i = np.asarray([destin_kp_idx])
    else:
        d_i = destin_kp_idx

    # Compute the epipolar line deepending on the source and destination given
    if source < destin:
        keypoint = edge.source.get_keypoint_coordinates(index = s_i, homogeneous = True).values
        reproj_keypoint = source_to_dest(keypoint[0], edge.source, edge.destination)
        f_matrix = edge['fundamental_matrix'].T
    else:
        keypoint = edge.destination.get_keypoint_coordinates(index = d_i, homogeneous = True).values
        reproj_keypoint = dest_to_source(keypoint[0], edge.source, edge.destination)
        f_matrix = edge['fundamental_matrix']

    e_line = compute_epipolar_line(keypoint[0], f_matrix=f_matrix)
    m=(-(e_line[0]/e_line[1]))
    b=(-(e_line[2]/e_line[1]))

    ax.imshow(composite, cmap="Greys")

    ax.plot(edge.source.get_keypoint_coordinates(index = s_i).x,
         edge.source.get_keypoint_coordinates(index = s_i).y,
         markersize = 7, marker = '.', linewidth = 0, color = 'r')

    x_offset = s_shape[1] + 100

    destin_coords = edge.destination.get_keypoint_coordinates(index = d_i)
    destin_xcoords = destin_coords.x + x_offset
    destin_ycoords = destin_coords.y

    e_line_x = np.asarray([0, d_shape[0]])
    e_line_y = np.asarray([get_y(0, m, b), get_y(500, m, b)])

    if source < destin:
        e_line_x = e_line_x + x_offset
        reproj_keypoint[0] = reproj_keypoint[0] + x_offset

    ax.plot(destin_xcoords, destin_ycoords,
         markersize = 7, marker = '.', linewidth = 0, color = 'r')

    ax.plot(e_line_x, e_line_y,
         color = 'y', linewidth = 3, alpha = .3)
    ax.plot(reproj_keypoint[0], reproj_keypoint[1],
         markersize = 7, marker = '.', linewidth = 0, color = 'b')

    return ax
