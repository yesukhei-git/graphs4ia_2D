# This module contains functions to create data sets from simulations
import numpy as np
import tensorflow as tf
import scipy.sparse as sp
from sklearn.neighbors import radius_neighbors_graph
from .utils import sparse_to_tuple, rand_rotation_matrix


def graph_input_fn(catalog,
                   vector_features=(), scalar_features=(),
                   vector_labels=(), scalar_labels=(),
                   pos_key=['halos.x', 'halos.y', 'halos.z'],
                   group_key='groups.groupId', batch_size=128,
                   noise_size=32,
                   graph_radius=1000., shuffle=False, repeat=False,
                   prefecth=100,
                   rotate=False):
    """
    Python generator function that will create batches of graphs from
    input catalog.
    """
    features = vector_features + scalar_features
    labels = vector_labels + scalar_labels

    def graph_generator():
        # Identify the individual groups and pre-extract the relevant data
        group_ids = catalog[group_key]
        gids = np.unique(group_ids)

        # Extracts columns of interest into memory first
        Xsp = np.array(catalog[pos_key]).view(np.float64).reshape((-1, 3)).astype(np.float32)
        X = np.array(catalog[features]).view(np.float64).reshape((-1, len(features))).astype(np.float32)
        Y = np.array(catalog[labels]).view(np.float64).reshape((-1, len(labels))).astype(np.float32)

        n_batches = len(gids) // batch_size
        last_batch = len(gids) % batch_size

        while True:
            # Apply permutation
            if shuffle:
                gids = np.random.permutation(gids)

            for b in range(n_batches+1):
                if b == n_batches:
                    bs = last_batch
                else:
                    bs = batch_size

                # Extract the groupId for the elements of the batch
                inds = gids[batch_size*b:batch_size*b + bs]

                # Extract and concatenate each array
                groups = [np.where(group_ids == i)[0] for i in inds]
                xsp = np.concatenate([Xsp[g] for g in groups])
                x = np.concatenate([X[g] for g in groups])
                y = np.concatenate([Y[g] for g in groups])

                n = np.random.randn(len(x), noise_size).astype(np.float32)

                # Apply rotation of vector quantities if requested
                if rotate:
                    M = rand_rotation_matrix()
                    xsp = xsp.dot(M.T)

                    # Apply rotation to labels and features
                    n_features = len(vector_features) // 3
                    n_labels = len(vector_labels) // 3
                    for i in range(n_features):
                        x[:, i*3:i*3+3] = x[:, i*3:i*3+3].dot(M.T)
                    for i in range(n_labels):
                        y[:, i*3:i*3+3] = y[:, i*3:i*3+3].dot(M.T)

                # Compute adjacency matrix for each entry
                graphs = [radius_neighbors_graph(Xsp[g], graph_radius,
                                                 mode='distance',
                                                 include_self=False) for g in groups]

                # Block adjacency matrix for the batch
                W = sp.block_diag(graphs)

                # Building pooling matrix for the batch
                data = np.concatenate([np.ones(graphs[i].shape[0])/graphs[i].shape[0] for i in range(len(groups))])
                row = np.concatenate([a*np.ones(graphs[i].shape[0]) for a,i in enumerate(range(len(groups)))]).astype('int')
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
    dataset = dataset.prefetch(prefecth)
    return dataset
