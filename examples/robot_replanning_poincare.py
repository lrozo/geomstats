"""Learning embedding of graph using Poincare Ball Model."""

import logging
import numpy as np
import matplotlib.cm as cm

import matplotlib.pyplot as plt

import pymanopt
from pymanopt.manifolds import PoincareBall
from pymanopt.solvers import ConjugateGradient, TrustRegions

import geomstats.backend as gs
import geomstats.visualization as visualization
from geomstats.datasets.utils import load_robot_planning_graph
from geomstats.geometry.poincare_ball import PoincareBall


def grid_embedding_init(grid_rows, grid_cols):
    grid_limits = [-0.5, 0.5, -0.5, 0.5]
    x_coord = np.linspace(grid_limits[0], grid_limits[1], grid_cols)
    y_coord = np.linspace(grid_limits[3], grid_limits[2], grid_rows)
    grid_embeddings = np.zeros((grid_rows * grid_cols, 2))
    node = 0
    for node_y in range(grid_cols):
        for node_x in range(grid_rows):
            grid_embeddings[node, :] = np.array([x_coord[node_x], y_coord[node_y]])
            node += 1
    return grid_embeddings


def log_sigmoid(vector):
    """Logsigmoid function.

    Apply log sigmoid function

    Parameters
    ----------
    vector : array-like, shape=[n_samples, dim]

    Returns
    -------
    result : array-like, shape=[n_samples, dim]
    """
    return gs.log((1 / (1 + gs.exp(-vector))))


def grad_log_sigmoid(vector):
    """Gradient of log sigmoid function.

    Parameters
    ----------
    vector : array-like, shape=[n_samples, dim]

    Returns
    -------
    gradient : array-like, shape=[n_samples, dim]
    """
    return 1 / (1 + gs.exp(vector))


def grad_squared_distance(point_a, point_b):
    """Gradient of squared hyperbolic distance.

    Gradient of the squared distance based on the
    Ball representation according to point_a

    Parameters
    ----------
    point_a : array-like, shape=[n_samples, dim]
        First point in hyperbolic space.
    point_b : array-like, shape=[n_samples, dim]
        Second point in hyperbolic space.

    Returns
    -------
    dist : array-like, shape=[n_samples, 1]
        Geodesic squared distance between the two points.
    """
    hyperbolic_metric = PoincareBall(2).metric
    log_map = hyperbolic_metric.log(point_b, point_a)

    return -2 * log_map


def loss(example_embedding, context_embedding, negative_embedding,
         manifold):
    """Compute loss and grad.

    Compute loss and grad given embedding of the current example,
    embedding of the context and negative sampling embedding.
    """
    n_edges, dim =\
        negative_embedding.shape[0], example_embedding.shape[-1]
    example_embedding = gs.expand_dims(example_embedding, 0)
    context_embedding = gs.expand_dims(context_embedding, 0)

    positive_distance =\
        manifold.metric.squared_dist(
            example_embedding, context_embedding)
    positive_loss =\
        log_sigmoid(-positive_distance)

    reshaped_example_embedding =\
        gs.repeat(example_embedding, n_edges, axis=0)

    negative_distance =\
        manifold.metric.squared_dist(
            reshaped_example_embedding, negative_embedding)
    negative_loss = log_sigmoid(negative_distance)

    total_loss = -(positive_loss + negative_loss.sum())

    positive_log_sigmoid_grad =\
        -grad_log_sigmoid(-positive_distance)

    positive_distance_grad =\
        grad_squared_distance(example_embedding, context_embedding)

    positive_grad =\
        gs.repeat(positive_log_sigmoid_grad, dim, axis=-1)\
        * positive_distance_grad

    negative_distance_grad =\
        grad_squared_distance(reshaped_example_embedding, negative_embedding)

    negative_distance = gs.to_ndarray(negative_distance,
                                      to_ndim=2, axis=-1)
    negative_log_sigmoid_grad =\
        grad_log_sigmoid(negative_distance)

    negative_grad = negative_log_sigmoid_grad\
        * negative_distance_grad

    example_grad = -(positive_grad + negative_grad.sum(axis=0))

    return total_loss, example_grad


def main():
    """Learning Poincaré graph embedding.

    Learns Poincaré Ball embedding by using Riemannian
    gradient descent algorithm.
    """
    gs.random.seed(1234)
    dim = 2
    max_epochs = 250
    lr = .02
    n_negative = 2
    context_size = 1
    initialization = 'Grid'
    n_cols, n_rows = 7, 7
    planning_graph = load_robot_planning_graph(n_rows, n_cols, connection_type=1)

    nb_vertices_by_edges =\
        [len(e_2) for _, e_2 in planning_graph.edges.items()]
    logging.info('Number of edges: %s', len(planning_graph.edges))
    logging.info(
        'Mean vertices by edges: %s',
        (sum(nb_vertices_by_edges, 0) / len(planning_graph.edges)))

    negative_table_parameter = 5
    negative_sampling_table = []

    for i, nb_v in enumerate(nb_vertices_by_edges):
        negative_sampling_table +=\
            ([i] * int((nb_v**(3. / 4.))) * negative_table_parameter)

    negative_sampling_table = gs.array(negative_sampling_table)
    random_walks = planning_graph.random_walk(walk_length=7)
    if initialization == 'Random':
        embeddings = gs.random.normal(size=(planning_graph.n_nodes, dim))
        embeddings = embeddings * 0.2
    elif initialization == 'Grid':
        embeddings = grid_embedding_init(n_rows, n_cols)

    hyperbolic_manifold = PoincareBall(2)

    colors = cm.rainbow(np.linspace(0, 1, len(planning_graph.labels)))
    circle = visualization.PoincareDisk(point_type='ball')
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111)
    # circle.add_points(gs.array([[0, 0]]))
    circle.set_ax(ax)
    circle.draw(ax=ax)
    for i_embedding, embedding in enumerate(embeddings):
        plt.scatter(embedding[0], embedding[1], alpha=1.0, color=colors[planning_graph.labels[i_embedding][0] - 1],
                    label=i_embedding)
        plt.annotate(i_embedding + 1, (embedding[0], embedding[1]))
    # plt.legend()
    plt.show()

    for epoch in range(max_epochs):
        total_loss = []
        for path in random_walks:

            for example_index, one_path in enumerate(path):
                context_index = path[max(0, example_index - context_size):
                                     min(example_index + context_size,
                                     len(path))]
                negative_index = []
                for i in range(context_size):
                    auxiliar_neg_table = negative_sampling_table[negative_sampling_table != context_index[i]]
                    for j in planning_graph.edges[context_index[i]]:
                        auxiliar_neg_table = auxiliar_neg_table[auxiliar_neg_table != j]
                    aux_negative_index = gs.random.randint(auxiliar_neg_table.shape[0], size=(1, n_negative))
                    negative_index.append(auxiliar_neg_table[aux_negative_index])
                negative_index = np.array(negative_index).squeeze()
                # negative_index =\
                #     gs.random.randint(negative_sampling_table.shape[0],
                #                       size=(len(context_index),
                #                       n_negative))
                # negative_index = negative_sampling_table[negative_index]

                example_embedding = embeddings[one_path]

                for one_context_i, one_negative_i in zip(context_index,
                                                         negative_index):
                    context_embedding = embeddings[one_context_i]
                    negative_embedding = embeddings[one_negative_i]
                    l, g_ex = loss(
                        example_embedding,
                        context_embedding,
                        negative_embedding,
                        hyperbolic_manifold)
                    total_loss.append(l)

                    example_to_update = embeddings[one_path]
                    embeddings[one_path] = hyperbolic_manifold.metric.exp(
                        -lr * g_ex, example_to_update)

        logging.info(
            'iteration %d loss_value %f',
            epoch, sum(total_loss, 0) / len(total_loss))

    circle = visualization.PoincareDisk(point_type='ball')
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111)
    circle.add_points(gs.array([[0, 0]]))
    circle.set_ax(ax)
    circle.draw(ax=ax)
    for i_embedding, embedding in enumerate(embeddings):
        plt.scatter(embedding[0], embedding[1], alpha=1.0, color=colors[planning_graph.labels[i_embedding][0]-1],
                    label=i_embedding)
        plt.annotate(i_embedding+1, (embedding[0], embedding[1]))
    # plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
