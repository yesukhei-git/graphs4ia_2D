# This module contains functions to create data sets from simulations
import numpy as np
import tensorflow as tf
import scipy.sparse as sp
from multiprocessing import Pool
from sklearn.neighbors import radius_neighbors_graph
from .utils import sparse_to_tuple, rand_rotation_matrix

def _process_graph(args):
    """
    Function that preprocesses the dataset for fast training
    """
    gid, group_ids, Xsp, X, Y, n_features, n_labels, graph_radius = args
    g = np.where(group_ids == gid)[0]
    xsp = Xsp[g]
    x = X[g]
    y = Y[g]
    # Compute adjacency matrix for each entry
    graph = radius_neighbors_graph(xsp, graph_radius, mode='distance',
                                   include_self=False)
    return (xsp, x, y, graph)

def graph_input_fn(catalog,
                   vector_features=(), scalar_features=(),
                   vector_labels=(), scalar_labels=(),
                   pos_key=['halos.x', 'halos.y', 'halos.z'],
                   group_key='groups.groupId', batch_size=128,
                   noise_size=32,
                   graph_radius=1000., shuffle=False, repeat=False,
                   prefetch=100, poolsize=12, balance_key='groups.mass_scaled',
                   rotate=False):
    """
    Python generator function that will create batches of graphs from
    input catalog.
    """
    features = vector_features + scalar_features
    labels = vector_labels + scalar_labels

    # It takes a minute but we precompute all the graphs and data
    # Identify the individual groups and pre-extract the relevant data
    group_ids = catalog[group_key]
    gids, idx = np.unique(group_ids, return_index=True)

    # Extracts columns of interest into memory first
    Xsp = np.array(catalog[pos_key]).view(np.float64).reshape((-1, 3)).astype(np.float32)
    X = np.array(catalog[features]).view(np.float64).reshape((-1, len(features))).astype(np.float32)
    Y = np.array(catalog[labels]).view(np.float64).reshape((-1, len(labels))).astype(np.float32)

    n_batches = len(gids) // batch_size
    last_batch = len(gids) % batch_size

    n_features = len(vector_features) // 3
    n_labels = len(vector_labels) // 3

    print("Precomputing dataset")
    with Pool(poolsize) as p:
        cache = p.map(_process_graph, [(gid, group_ids, Xsp, X, Y, n_features, n_labels, graph_radius) for gid in gids])
    print("Done")

    if balance_key is not None:
        # Balance probablities of graphs based on group mass
        p, bin = np.histogram(catalog[balance_key][idx], 16)
        mbin = np.digitize(catalog[balance_key][idx], bin[:-1]) - 1
        cat_probs = (1./p)[mbin]
        cat_probs /= cat_probs.sum()

    def graph_generator():

        while True:
            # Apply permutation
            if shuffle:
                if balance_key is not None:
                    batch_gids = np.random.choice(len(gids), len(gids), p=cat_probs)
                else:
                    batch_gids = np.random.permutation(len(gids))
            else:
                batch_gids = range(len(gids))

            for b in range(n_batches+1):
                if b == n_batches:
                    bs = last_batch
                else:
                    bs = batch_size

                # Extract the groupId for the elements of the batch
                inds = batch_gids[batch_size*b:batch_size*b + bs]

                res = [cache[i] for i in inds]

                graphs = [r[-1] for r in res]
                xsp = np.concatenate([r[0] for r in res])
                x = np.concatenate([r[1] for r in res])
                y = np.concatenate([r[2] for r in res])
                n = np.random.randn(len(x), noise_size).astype(np.float32)

                # Apply rotation of vector quantities if requested
                if rotate:
                    M = rand_rotation_matrix()
                    xsp = xsp.dot(M.T)
                    for i in range(n_features):
                        x[:, i*3:i*3+3] = x[:, i*3:i*3+3].dot(M.T)
                    for i in range(n_labels):
                        y[:, i*3:i*3+3] = y[:, i*3:i*3+3].dot(M.T)

                # Block adjacency matrix for the batch
                W = sp.block_diag(graphs)

                # Building pooling matrix for the batch
                data = np.concatenate([np.ones(graphs[i].shape[0])/graphs[i].shape[0] for i in range(len(graphs))])
                row = np.concatenate([a*np.ones(graphs[i].shape[0]) for a,i in enumerate(range(len(graphs)))]).astype('int')
                col = np.arange(W.shape[0]).astype('int32')

                # Preparing sparse matrices in TF format
                pooling_matrix = sparse_to_tuple(sp.coo_matrix((data, (row,col)), shape=(bs, W.shape[0])))
                W = sparse_to_tuple(W)

                yield (W[0], W[1], W[2],
                       pooling_matrix[0], pooling_matrix[1], pooling_matrix[2],
                       xsp, x, n), y

            if not repeat:
                break

    graph_generator.output_types = ((tf.int32, tf.float32, tf.int64,
                                     tf.int32, tf.float32, tf.int64,
                                     tf.float32, tf.float32, tf.float32),
                                    tf.float32)
    graph_generator.output_shapes = (((None, 2), (None,), (2,),
                                      (None, 2), (None,), (2,),
                                      (None, 3), (None, len(features)),
                                      (None, noise_size)),
                                     (None, len(labels)))

    dataset = tf.data.Dataset.from_generator(graph_generator,
                               output_types = graph_generator.output_types,
                               output_shapes = graph_generator.output_shapes)
    dataset = dataset.prefetch(64)
    return dataset
