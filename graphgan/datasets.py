# This module contains functions to create data sets from simulations
import numpy as np
import tensorflow as tf

def mb2_graph_input_fn(catalog,
                       scalar_features=(),
                       vector_features=(),
                       scalar_labels=(),
                       vector_labels=(),
                       group_key='groups.groupId',
                       pos_keys=('halos.x', 'halos.y', 'halos.z'),
                       rotate=True,
                       repeat=True,
                       shuffle=True,
                       cache=None,
                       batch_size=128,
                       num_parallel_calls=None):
    """
    Input function that creates graphs halos
    """
    labels = vector_labels + scalar_labels
    features = vector_features + scalar_features

    def _graph_preprocessing(groupId):
        """
        Extacts features and labels from dataset for a given group
        """
        mask = catalog[group_key] == groupId
        x = catalog[mask][features]
        y = catalog[mask][labels]


    # Identify the individual groups and pre-extract the relevant data

    gids = np.unique(catalog[group_key])

    # Loop over the groups and extract graphs and data for each group
    dataset = tf.data.Dataset.range(len(gids))
    dataset = dataset.map(_graph_preprocessing,
                          num_parallel_calls=num_parallel_calls)
    if repeat:
        dataset = dataset.repeat()
    if shuffle:
        dataset = dataset.shuffle(len(gids))
    dataset = dataset.batch(batch_size)

    # Apply pre-processing function on each batch
    dataset = dataset.map(mapping_function)
