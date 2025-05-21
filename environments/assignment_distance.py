import numpy as np
from scipy.optimize import linear_sum_assignment
import torch


def assignment_distance(achieved_goal, desired_goal):
    assert len(achieved_goal) == len(desired_goal)
    assert type(achieved_goal[0]) is np.ndarray
    assert type(desired_goal[0]) is np.ndarray
    n_agent = len(achieved_goal)
    weight_matrix = np.eye(n_agent)
    for i in range(n_agent):
        for j in range(n_agent):
            weight_matrix[i, j] = np.linalg.norm(achieved_goal[i] - desired_goal[j])
    row_ind, col_ind = linear_sum_assignment(weight_matrix)
    return weight_matrix[row_ind, col_ind].sum()  # self.eng.distance(weight_matrix)  # 两个构型之间的距离


def assignment_distance_batch(achieved_goal, desired_goal):
    # 确保输入是 PyTorch 张量
    assert isinstance(achieved_goal, torch.Tensor)
    assert isinstance(desired_goal, torch.Tensor)
    assert achieved_goal.shape == desired_goal.shape
    batch_size, n_agent, _ = achieved_goal.shape
    # 初始化结果张量
    distances = torch.zeros(batch_size, device=achieved_goal.device)

    for i in range(batch_size):
        weight_matrix = torch.zeros(n_agent, n_agent, device=achieved_goal.device)
        for j in range(n_agent):
            for k in range(n_agent):
                weight_matrix[j, k] = torch.norm(achieved_goal[i, j] - desired_goal[i, k])

        # 使用线性求和分配算法
        row_ind, col_ind = linear_sum_assignment(weight_matrix.cpu().numpy())

        # 计算总距离
        distances[i] = weight_matrix[row_ind, col_ind].sum()

    return distances.detach().numpy()


if __name__ == "__main__":
    # 示例数据
    A = torch.tensor([[[0, 0, 0], [1, 1, 0], [0, 2, 4], [1, 3, 0]],[[0, 0, 0], [1, 1, 0], [0, 2, 4], [1, 3, 0]],[[0, 0, 0], [1, 1, 0], [0, 2, 4], [1, 3, 0]]], dtype=torch.float32)
    B = torch.tensor([[[0, 0, 0], [0, -1, 0], [0, 1, 0], [0, 2, 0]],[[0, 0, 0], [0, -1, 0], [0, 1, 0], [0, 2, 0]],[[0, 0, 0], [0, -1, 0], [0, 1, 0], [0, 2, 0]]], dtype=torch.float32)

    # 将数据移动到 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A = A.to(device)
    B = B.to(device)

    # 计算距离
    print(assignment_distance_batch(A, B))
