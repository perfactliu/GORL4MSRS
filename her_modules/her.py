import numpy as np
import torch


class her_sampler:
    def __init__(self, replay_k, reward_func=None):
        self.replay_k = replay_k
        self.future_p = 1 - (1. / (1 + replay_k))  # replay_k越大，future_p越接近1，her采样越多
        self.reward_func = reward_func

    def sample_her_transitions(self, episode_batch, batch_size_in_transitions):
        # for key in episode_batch.keys():
        #     print(key)
        #     for i in range(episode_batch[key].shape[0]):
        #         print(len(episode_batch[key][i]))
        T = episode_batch['actions'].shape[1]  # 时间步长
        rollout_batch_size = episode_batch['actions'].shape[0]  # buffer size
        batch_size = batch_size_in_transitions  # batch size
        # select which rollouts and which timesteps to be used
        episode_idxs = np.random.randint(0, rollout_batch_size, size=batch_size)  # 轨迹index
        t_samples = np.random.randint(0, T, size=batch_size)  # 具体某一个时间步
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy() for key in episode_batch.keys()}
        # 此时返回的transitions不再是轨迹,而是具体时间步
        # for key in episode_batch.keys():
        #     print(key)
        #     print(transitions[key])
        # her idx
        her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + future_offset)[her_indexes]
        # replace go with achieved goal
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        for i, her_index in enumerate(her_indexes[0]):
            if np.array_equal(transitions['ag_next'][her_index], future_ag[i]):
                transitions['done'][her_index] = True
            else:
                transitions['done'][her_index] = False
        transitions['g'][her_indexes] = future_ag
        # to get the params to re-compute reward
        moveless_indices = np.where(np.all(transitions['actions'] == [0]*transitions['actions'].shape[1], axis=1))[0]
        transitions['r'] = np.expand_dims(self.reward_func(torch.from_numpy(transitions['obs']),
                                                           torch.from_numpy(transitions['ag_next']),
                                                           torch.from_numpy(transitions['g']),
                                                           moveless_indices), 1)
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}

        return transitions



