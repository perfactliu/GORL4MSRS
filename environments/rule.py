import numpy as np
import torch


def rule(action):
    """
    action rule for single agent without extra constrain.

    Parameters:
        action - the desired action

    Returns:
        the obstacle of this action
        the neighbor who provide pivot
        the neighbor new connection built with after this action
        the displacement vector of this action
    """
    if action == 0:
        return {tuple(np.array([0, 0, 0]))}, {}, {}, np.array([0, 0, 0])
    # 动作的特征位置
    action = action - 1
    N = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1], [0, -1, 0], [-1, 0, 0]])  # 特征矩阵
    pivot = action % 6  # 旋转支持面
    direction = torch.div(action, 6, rounding_mode="floor") % 4  # 旋转方向
    angle = torch.div(action, 24, rounding_mode="floor") % 2  # 0为90度，1为180度
    vPivot = N[pivot, :]
    v1 = N[5 - pivot, :]
    N_ = np.delete(N, [pivot, 5 - pivot], 0)
    v2 = v1 + N_[direction, :]
    v3 = 2 * N_[direction, :]
    v4 = vPivot + 2 * N_[direction, :]
    v5 = N_[direction, :]
    v6 = vPivot + N_[direction, :]
    newpivot = set()
    if angle == 0:
        obstacle = {tuple(v1), tuple(v2), tuple(v5)}
        newpivot.add(tuple(v6))
        velocity = N_[direction, :]
    else:
        obstacle = {tuple(v1), tuple(v2), tuple(v5), tuple(v3), tuple(v4), tuple(v6)}
        newpivot.add(tuple(vPivot))
        velocity = N[pivot, :] + N_[direction, :]
    return obstacle, tuple(vPivot), newpivot, velocity
    # 旋转过程障碍，本次旋转所用支持面，旋转后全部支持面，旋转后的位置，全部相对于卫星的原位置