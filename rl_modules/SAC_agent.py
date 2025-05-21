import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from rl_modules.sac_models import Critic, Actor
from rl_modules.replay_buffer import replay_buffer
from her_modules.her import her_sampler
from tqdm import tqdm
from environments.plot import plot_and_save_3d_cubes

"""
SAC with HER 

"""


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class Agent(nn.Module):
    def __init__(self, argus, env, env_params):
        super(Agent, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.args = argus
        self.update_count = 0
        self.env = env
        self.env_params = env_params
        self.target_update_freq = self.args.target_update_freq
        self.training_step = 0
        self.update_actor = True
        
        self.actor_network = Actor(self.args.lr_actor, env_params, self.args.layer_norm).to(self.device)
        self.actor_target_network = Actor(self.args.lr_actor, env_params, self.args.layer_norm).to(self.device)
        hard_update(self.actor_target_network, self.actor_network)
        
        self.critic_network = Critic(self.args.lr_critic, env_params, self.args.layer_norm).to(self.device)
        self.critic_target_network = Critic(self.args.lr_critic, env_params, self.args.layer_norm).to(self.device)
        hard_update(self.critic_target_network, self.critic_network)

        self.log_alpha = torch.tensor(self.args.log_alpha, requires_grad=True)  # alpha 的对数，确保alpha始终为正
        self.target_entropy = np.log(env_params['action'])*0.1  # 根据经验取动作维度负值 -48
        # self.target_entropy = 0.01
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.args.alpha_lr)  # alpha优化器

        # her sampler
        self.her_module = her_sampler(self.args.replay_k, self.env.compute_reward)
        # create the replay buffer
        self.buffer = replay_buffer(self.env_params, self.args.buffer_size, self.her_module.sample_her_transitions)
        self.writer = SummaryWriter(log_dir=self.args.log_dir)

    def train_cycle(self):
        """
        train the network
        """
        # start to collect samples
        for epoch in range(self.args.n_epochs):
            mb_obs, mb_ag, mb_g, mb_actions, mb_done, mb_mask = [], [], [], [], [], []
            for _ in range(self.args.n_batch):
                ep_obs, ep_ag, ep_g, ep_actions, ep_done, ep_mask = [], [], [], [], [], []
                # reset the environment
                (obs, ag, g), mask = self.env.reset()
                # start to collect samples
                success_flag = 1
                for t in range(self.env_params['max_timestep']):
                    with torch.no_grad():
                        # concatenate the stuffs
                        inputs = np.concatenate([obs, g]).flatten()
                        input_tensor = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0).to(self.device)
                        pi = self.actor_network(input_tensor, mask)
                        action = self._select_actions_train(pi)
                    # feed the actions into the environment
                    (obs_new, ag_new, _), done, mask_new = self.env.step(action)
                    # append rollouts
                    ep_obs.append(obs)
                    ep_ag.append(ag)
                    ep_g.append(g)
                    ep_actions.append(action)
                    ep_done.append([done])
                    ep_mask.append(mask)
                    # re-assign the observation
                    obs = obs_new
                    ag = ag_new
                    mask = mask_new
                    if done and self.env.success and success_flag:
                        print('cause of done:reconfigure successfully.')
                        self.env.success_cycle += 1
                        success_flag = 0
                    if t == self.env_params['max_timestep']-1 and success_flag:
                        print('cause of done:reconfigure time out.')
                    # 在动作空间中加入了全局不动的一维，让模型学习到已达成目标后采取不动的策略
                    # if done and self.env.success:
                    #     rest_steps = self.env_params['max_timestep']-t-1
                    #     for _ in range(rest_steps):
                    #         ep_obs.append(obs)
                    #         ep_ag.append(ag)
                    #         ep_g.append(g)
                    #         ep_actions.append([0]*self.env.satellite_number)
                    #     break
                self.writer.add_scalar('train/success rate', self.env.cal_success_rate(), self.env.cycle)
                ep_obs.append(obs)
                ep_ag.append(ag)
                mb_obs.append(ep_obs)
                mb_ag.append(ep_ag)
                mb_g.append(ep_g)
                mb_actions.append(ep_actions)
                mb_done.append(ep_done)
                mb_mask.append(ep_mask)
            # convert them into arrays
            # print(mb_obs)
            mb_obs = np.array(mb_obs)
            mb_ag = np.array(mb_ag)
            mb_g = np.array(mb_g)
            mb_actions = np.array(mb_actions)
            mb_done = np.array(mb_done)
            mb_mask = np.array(mb_mask)

            # store the episodes
            self.buffer.store_episode([mb_obs, mb_ag, mb_g, mb_actions, mb_done, mb_mask])
            critic_q1_loss, critic_q2_loss, actor_loss, alpha_loss, entropy, alpha = 0, 0, 0, 0, 0, 0
            for _ in tqdm(range(self.args.network_learn_freq)):
                # train the network
                self.training_step += 1
                critic_q1_loss, critic_q2_loss, actor_loss, alpha_loss, entropy, alpha = self._update_network()
                self.writer.add_scalar('loss/critic q1 loss', critic_q1_loss, self.training_step)
                self.writer.add_scalar('loss/critic q2 loss', critic_q2_loss, self.training_step)
                self.writer.add_scalar('loss/actor loss', actor_loss, self.training_step)
                self.writer.add_scalar('loss/alpha loss', alpha_loss, self.training_step)
                self.writer.add_scalar('stat/entropy', entropy, self.training_step)
                self.writer.add_scalar('stat/alpha', alpha, self.training_step)
            print(f"epoch[{epoch+1}/{self.args.n_epochs}]:")
            print(f"critic q1 loss: {critic_q1_loss}")
            print(f"critic q2 loss: {critic_q2_loss}")
            print(f"actor loss: {actor_loss}")
            print(f"alpha loss: {alpha_loss}")
            print(f"entropy: {entropy}")
            print(f"alpha: {alpha}")

    def test_cycle(self):
        self.actor_network.eval()
        for episode in range(self.args.test_episodes):
            (obs, ag, g), mask = self.env.test_reset()
            reward = 0
            if self.args.plot:
                plot_and_save_3d_cubes(g, f"test_plot/episode{episode+1}/goal.png")
                plot_and_save_3d_cubes(obs, f"test_plot/episode{episode + 1}/step0.png")

            for t in range(self.env_params['max_timestep']):
                # concatenate the stuffs
                inputs = np.concatenate([obs, g]).flatten()
                input_tensor = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0).to(self.device)
                pi = self.actor_network(input_tensor, mask)
                action = self._select_actions_train(pi)

                # feed the actions into the environment
                (obs_new, ag_new, _), done, mask_new = self.env.step(action)
                reward += self.env.get_global_reward(obs, ag_new, g)
                # re-assign the observation
                if self.args.plot:
                    plot_and_save_3d_cubes(obs_new, f"test_plot/episode{episode + 1}/step{t+1}.png")
                obs = obs_new
                mask = mask_new
                if done and self.env.success:
                    print('cause of done:reconfigure successfully.')
                    self.env.success_cycle += 1
                    break
                if t == self.env_params['max_timestep'] - 1:
                    print('cause of done:reconfigure time out.')
        print(f'test/success rate:{self.env.cal_success_rate()}')

    def _select_actions_train(self, pi):
        action = torch.multinomial(pi, num_samples=1)
        if action.item() == 48*self.env.satellite_number:
            final_action = [0] * self.env.satellite_number
            return final_action
        max_id = action.item()//48
        max_num = action.item() % 48 + 1
        final_action = [0]*self.env.satellite_number
        final_action[max_id] = max_num
        return final_action

    def _select_actions_test(self, pi):
        action = torch.argmax(pi)
        if action.item() == 48*self.env.satellite_number:
            final_action = [0] * self.env.satellite_number
            return final_action
        max_id = action.item()//48
        max_num = action.item() % 48 + 1
        final_action = [0]*self.env.satellite_number
        final_action[max_id] = max_num
        return final_action

    def _update_network(self):
        # sample the episodes
        transitions = self.buffer.sample(self.args.batch_size)
        # pre-process the observation and goal
        o, o_next, g, a, mask = transitions['obs'], transitions['obs_next'], transitions['g'], transitions['actions'], transitions['mask']
        transitions['g_next'] = transitions['g']
        g_next = transitions['g_next']
        transitions['actions'] = self._preproc_action(a)
        batch_size = o.shape[0]
        o = o.reshape(batch_size, -1)
        g = g.reshape(batch_size, -1)
        o_next = o_next.reshape(batch_size, -1)
        g_next = g_next.reshape(batch_size, -1)
        # start to do the update
        inputs = np.concatenate([o, g], axis=1)
        inputs_next = np.concatenate([o_next, g_next], axis=1)
        # transfer them into the tensor
        inputs_tensor = torch.tensor(inputs, dtype=torch.float32).to(self.device)
        inputs_next_tensor = torch.tensor(inputs_next, dtype=torch.float32).to(self.device)
        actions_tensor = torch.tensor(transitions['actions'], dtype=torch.float32).to(self.device)
        action_indices = actions_tensor.argmax(dim=1).unsqueeze(-1)    # 将独热编码转换为索引
        r_tensor = torch.tensor(transitions['r'], dtype=torch.float32).to(self.device).squeeze(-1)
        done_tensor = torch.tensor(transitions['done'], dtype=torch.float32).to(self.device).squeeze(-1)

        # calculate loss and backward
        self.actor_target_network.eval()
        self.critic_target_network.eval()

        # 计算目标 Q 值
        with torch.no_grad():
            # next_action_probs = self.actor_target_network(inputs_next_tensor, mask)  # 策略网络生成下一步的动作概率
            next_action_probs = self.actor_target_network(inputs_next_tensor, use_mask = False)  # 策略网络生成下一步的动作概率
            target_q1, target_q2 = self.critic_target_network(inputs_next_tensor)  # 目标 Q 网络估计下一步所有动作的 Q 值
            target_q_min = torch.min(target_q1, target_q2)
            v_next = torch.sum(next_action_probs * (target_q_min - self.log_alpha.exp() * torch.log(next_action_probs + 1e-10)),dim=1)
            target_q = r_tensor + self.args.gamma * v_next * (1 - done_tensor)  # 计算 TD 目标值
        # 计算均方误差损失
        self.critic_network.train()
        current_q1, current_q2 = self.critic_network(inputs_tensor)
        # print(current_q1.shape, action_indices.shape)
        current_q1 = current_q1.gather(1, action_indices).squeeze(-1)
        current_q2 = current_q2.gather(1, action_indices).squeeze(-1)
        self.critic_q1_loss = F.mse_loss(current_q1, target_q) 
        self.critic_q2_loss = F.mse_loss(current_q2, target_q)

        # 更新critic参数
        self.critic_network.update_Q1(self.critic_q1_loss)
        self.critic_network.update_Q2(self.critic_q2_loss)

        if self.update_actor:
            # 更新alpha
            # 采样动作，并计算 Q 值
            self.actor_network.eval()
            # action_probs = self.actor_network(inputs_tensor, mask)
            action_probs = self.actor_network(inputs_tensor, use_mask = False)
            self.entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-10), dim=1).mean().detach()
            # 计算 alpha 的损失
            self.alpha_loss = self.log_alpha.exp() * (self.entropy - self.target_entropy).detach()
            # 反向传播并更新 alpha
            self.alpha_optimizer.zero_grad()
            self.alpha_loss.backward()
            self.alpha_optimizer.step()

            # 更新actor
            self.actor_network.train()
            # 用新的 alpha 计算 actor_loss 并更新 Actor
            self.critic_network.eval()
            # # 计算当前 Q 值
            current_q1, current_q2 = self.critic_network(inputs_tensor)  # 形状: (batch_size, action_dim)
            current_q_min = torch.min(current_q1, current_q2)
            self.actor_loss = torch.sum(action_probs * (self.log_alpha.exp() * torch.log(action_probs + 1e-10) - current_q_min), dim=1).mean()
            self.actor_network.optimizer.zero_grad()
            self.actor_loss.backward()
            self.actor_network.optimizer.step()

            self.update_count += 1
            if self.update_count % self.target_update_freq == 0:
                soft_update(self.critic_target_network, self.critic_network, self.args.tau)
                soft_update(self.actor_target_network, self.actor_network, self.args.tau)
        self.update_actor = not self.update_actor
        
        return (self.critic_q1_loss.mean().cpu().detach().numpy(), self.critic_q2_loss.mean().cpu().detach().numpy(), self.actor_loss.cpu().detach().numpy(),
                self.alpha_loss.cpu().detach().numpy(), self.entropy.cpu().detach().numpy(), self.log_alpha.exp().cpu().detach().numpy())

    def _preproc_action(self, a):
        extend_batch_action = []
        for i in range(len(a)):
            extend_action = [0] * 48 * self.env.satellite_number
            flag = 0
            for index, value in enumerate(a[i]):
                if value != 0:
                    extend_action[index*48+int(value)-1] = 1
                    extend_action.append(0)
                    flag = 1
                    break
            if flag == 0:
                extend_action.append(1)
            extend_batch_action.append(extend_action)
        return extend_batch_action

    def saveCheckpoints(self, model_name):
        self.critic_network.saveCheckpoint(model_name)
        self.actor_network.saveCheckpoint(model_name)
        
    def loadCheckpoints(self, model_name):
        self.critic_network.loadCheckpoint(model_name)
        self.actor_network.loadCheckpoint(model_name)
