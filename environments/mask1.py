"""
mask illegal action of single agent
"""
from environments.rule import rule
import numpy as np
import torch
# from environment import SatelliteEnv


def mask1(positions, agent):
    obs = torch.zeros(3, 5, 5)
    for i in range(3):
        for j in range(5):
            for k in range(5):
                p = [j - 2, k - 2]
                p.insert(i, 0)
                v = tuple(agent.position + np.array(p))
                if v in positions:
                    obs[i][j][k] = 1
    # print(obs)
    # 根据obs计算合法动作集合
    b = (obs == 1)
    # print(b)
    neighbor = set()
    for i in range(3):
        for j in range(5):
            for k in range(5):
                if b[i][j][k]:
                    p = [j - 2, k - 2]
                    p.insert(i, 0)
                    p = tuple(p)
                    neighbor.add(p)
    # print(neighbor)
    # print(len(neighbor))
    legal_action = [0] * 48
    for a in range(1, 49):
        obstacle, pivot, new_connection, velocity = rule(a)
        # print(neighbor,obstacle,pivot,new_connection)
        # print(pivot in neighbor,new_connection & neighbor != set(),obstacle & neighbor == set())
        if pivot in neighbor and new_connection & neighbor != set() and obstacle & neighbor == set():
            legal_action[a-1] = 1
            # print(velocity+agent.position)
    return np.array(legal_action)


if __name__ == '__main__':
    env = SatelliteEnv(1, 5)
    env.reset()
    print(mask1(env.configuration_set(), env.agents[3]))
