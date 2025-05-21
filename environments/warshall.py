import numpy as np


def warshall(configuration_set):
    n_agent = len(configuration_set)
    graph_matrix = np.ones((n_agent, n_agent))
    configure = list(list(x) for x in configuration_set)  # 转为list
    configure = np.array(configure)
    for i in range(configure.shape[0]):
        for j in range(configure.shape[0]):
            if np.linalg.norm(configure[i] - configure[j]) <= 1:
                graph_matrix[i, j] = 0
    resultMatrix = graph_matrix
    for k in range(configure.shape[0]):
        for i in range(configure.shape[0]):
            for j in range(configure.shape[0]):
                resultMatrix[i][j] = min(
                    resultMatrix[i][j], resultMatrix[i][k] + resultMatrix[k][j])
    connect_flag = resultMatrix.sum(0).sum(0)
    return connect_flag
