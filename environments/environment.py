import numpy as np
import torch
import yaml
from environments.rule import rule
from environments.warshall import warshall
from environments.assignment_distance import assignment_distance, assignment_distance_batch
from environments.mask1 import mask1
from environments.mask2 import mask2
import random


class Module:
    def __init__(self):
        self._position = None  # 星本体系下位置
        self.action = None  # 位移
        self.color = None  # 动画演示中的颜色

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, position):
        assert type(position) is np.ndarray
        assert position.ndim == 1
        assert position.shape == (3,)
        self._position = position

    def step(self):
        self._position = self._position + self.action  # 位置矢量加位置矢量


class Target:
    def __init__(self):
        self._position = None
        self.color = None  # 动画演示中的颜色

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, position):
        assert type(position) is np.ndarray
        assert position.ndim == 1
        assert position.shape == (3,)
        self._position = position


class SatelliteEnv:
    def __init__(self, seed, change_goal_cycle):
        with open('environments/satellite_config.yaml', 'r') as file:
            satellite_config = yaml.safe_load(file)
        self.start_positions = satellite_config['start_positions']
        self.target_positions = satellite_config['target_positions']
        self.satellite_number = satellite_config['satellite_number']
        self.agents = [Module() for _ in range(self.satellite_number)]
        self.targets = [Target() for _ in range(self.satellite_number)]
        self.success = False
        self.cycle = 0  # 重构轮数
        self.success_cycle = 0  # 成功轮数
        self.seed = seed
        self.change_goal_cycle = change_goal_cycle  # 每cycle轮变换一次goal
        for i, agent in enumerate(self.agents):
            agent.position = np.array(self.start_positions[i])
        for i, target in enumerate(self.targets):
            target.position = np.array(self.target_positions[i])

    def _create_new_goal(self):
        if self.seed is not None:
            random.seed(self.seed)

        cubes = [[0, 0, 0]]  # 初始立方体
        directions = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]  # 六个方向

        while len(cubes) < self.satellite_number:
            base_cube = random.choice(cubes)  # 随机选择一个已有立方体
            direction = random.choice(directions)  # 随机选择一个方向
            new_cube = [base_cube[0] + direction[0], base_cube[1] + direction[1], base_cube[2] + direction[2]]

            if new_cube not in cubes:  # 确保不重合
                cubes.append(new_cube)

        self.target_positions = cubes
        for i, target in enumerate(self.targets):
            target.position = np.array(cubes[i])
        print(f'new goal:{cubes}')

    def reset(self):
        self.cycle += 1
        for i, agent in enumerate(self.agents):
            agent.position = np.array(self.start_positions[i])
            agent.action = None
        self.success = False
        if self.cycle % self.change_goal_cycle == 0:
            print("Changing goal step.")
            self._create_new_goal()
            while assignment_distance(np.array(self.start_positions), np.array(self.target_positions)) < 1:
                self._create_new_goal()
            print("Goal has changed.")
        return self._get_state(), self.get_mask()

    def test_reset(self):
        self.cycle += 1
        for i, agent in enumerate(self.agents):
            agent.position = np.array(self.start_positions[i])
            agent.action = None
        self.success = False
        print("Changing goal step.")
        self._create_new_goal()
        while assignment_distance(np.array(self.start_positions), np.array(self.target_positions)) < 1:
            self._create_new_goal()
        print("Goal has changed.")

        return self._get_state(), self.get_mask()

    def step(self, action_n):
        last_position = []
        for i, agent in enumerate(self.agents):
            last_position.append(agent.position)
            self._set_action(action_n[i], agent)

        for agent in self.agents:
            agent.step()
            agent.position = agent.position - self.agents[0].action  # 将坐标原点移回0号模块
        return self._get_state(), self._get_done(), self.get_mask()

    def _set_action(self, action, agent):
        _, _, _, velocity = rule(action)
        agent.action = velocity

    def _get_state(self):
        position_n = []
        desired_goal = []
        for agent in self.agents:
            position_n.append(agent.position)
        for target in self.targets:
            desired_goal.append(target.position)
        achieved_goal = position_n
        return position_n, achieved_goal, desired_goal

    def configuration_voxel(self):
        # refresh voxel description of configuration
        points_list = []  # 点云
        for agent in self.agents:
            points_list.append(agent.position)
        points = np.array(points_list)
        points = torch.tensor(points).int()
        min = torch.min(points, 0)[0].float()
        max = torch.max(points, 0)[0].float()
        window_len = (max - min + torch.tensor([1.0, 1.0, 1.0])).int().tolist()
        configuration_voxel = torch.zeros(window_len)
        for agent in self.agents:
            position = (torch.from_numpy(agent.position).float() - min).int()
            configuration_voxel[position[0]][position[1]][position[2]] = 1
        return configuration_voxel

    def get_global_reward(self, last_position, achieved_goal_next, desired_goal):
        # if the configuration is not connected, end the episode
        configuration = self.configuration_set()
        connect_flag = warshall(configuration)
        if connect_flag != 0:
            return -10
        distance_before = assignment_distance(last_position, desired_goal)
        distance_after = assignment_distance(achieved_goal_next, desired_goal)
        if distance_after < 1:
            return 10
        else:
            return distance_before/40 - distance_after/40

    def _get_done(self):
        # if the configuration is not connected, end the episode
        configuration = self.configuration_set()
        connect_flag = warshall(configuration)
        if connect_flag != 0:
            # print('cause of done:connection failure.')
            return True

        achieved_goal, desired_goal = [], []
        for agent, target in zip(self.agents, self.targets):
            achieved_goal.append(agent.position)
            desired_goal.append(target.position)
        distance = assignment_distance(achieved_goal, desired_goal)
        if distance < 1:
            # print('cause of done:reconfigure successfully.')
            self.success = True
            # self.success_cycle += 1
            return True
        return False

    def configuration_set(self):
        # refresh set description of configuration
        configuration_set = set()
        for agent in self.agents:
            configuration_set.add(tuple(agent.position))
        return configuration_set

    def compute_reward(self, last_position, achieved_goal_next, desired_changed_goal, moveless_indices):
        distance_before = assignment_distance_batch(last_position, desired_changed_goal)
        distance_after = assignment_distance_batch(achieved_goal_next, desired_changed_goal)
        done_indices = np.where(distance_after < 1)
        distance_before[done_indices] = distance_after[done_indices]+400
        distance_before[moveless_indices] = distance_after[moveless_indices]-40
        return distance_before/40 - distance_after/40

    def get_mask(self):
        final_mask = []
        for agent in self.agents:
            mask_rule = mask1(self.configuration_set(), agent)
            mask_warshall = mask2(self.configuration_set(), agent)
            final_mask.append(mask_rule*mask_warshall)
        final_mask = np.concatenate(final_mask)
        final_mask = np.append(final_mask, 1)  # 将最后一维的全局不动动作置为合法
        return final_mask

    def cal_success_rate(self):
        return format(self.success_cycle/self.cycle*100, '.2f')


# environment test
if __name__ == '__main__':
    env = SatelliteEnv(1,5)
    for cycle in range(5):
        (position_n, achieved_goal, desired_goal), mask = env.reset()
        for step in range(10):
            print(env.get_mask())
            actions = [0,0,0,48]
            (next_position, achieved_goal, desired_goal), reward, done = env.step(actions)
            print(f"next_position:{next_position},\n achieved_goal:{achieved_goal},\n desired_goal:{desired_goal},\n reward:{reward}, done:{done}")
            print(warshall(desired_goal))
            if done:
                break
    print(env.get_mask())

    # actions = [0, 0, 0, 0]
    # print(env.configuration_set())
    # state_re = env.reset()
    # print('next state:', next_voxel, '\nreward', reward, '\ndone:', done)
    # print(state_re)
