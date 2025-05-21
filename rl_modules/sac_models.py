import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
"""
the input x in both networks should be [o, g], where o is the observation and g is the goal.

"""


class Critic(nn.Module):
    def __init__(self, lr, env_params, layer_norm):
        super(Critic, self).__init__()

        self.layerNorm = layer_norm
        # Q1
        self.full1 = nn.Linear((env_params['obs'] + env_params['goal'])*env_params['sat_num'], 256)
        nn.init.kaiming_uniform_(self.full1.weight, a=0.01, mode='fan_in', nonlinearity='relu')

        self.layer_norm1 = nn.LayerNorm(256)

        self.full2 = nn.Linear(256, 256)
        nn.init.kaiming_uniform_(self.full2.weight, a=0.01, mode='fan_in', nonlinearity='relu')

        self.layer_norm2 = nn.LayerNorm(256)

        self.full3 = nn.Linear(256, 256)
        nn.init.kaiming_uniform_(self.full3.weight, a=0.01, mode='fan_in', nonlinearity='relu')

        self.layer_norm3 = nn.LayerNorm(256)

        self.final1 = nn.Linear(256, env_params['action'])

        # Q2
        self.full4 = nn.Linear((env_params['obs'] + env_params['goal'])*env_params['sat_num'], 256)
        nn.init.kaiming_uniform_(self.full4.weight, a=0.01, mode='fan_in', nonlinearity='relu')

        self.layer_norm4 = nn.LayerNorm(256)

        self.full5 = nn.Linear(256, 256)
        nn.init.kaiming_uniform_(self.full5.weight, a=0.01, mode='fan_in', nonlinearity='relu')

        self.layer_norm5 = nn.LayerNorm(256)

        self.full6 = nn.Linear(256, 256)
        nn.init.kaiming_uniform_(self.full6.weight, a=0.01, mode='fan_in', nonlinearity='relu')

        self.layer_norm6 = nn.LayerNorm(256)

        self.final2 = nn.Linear(256, env_params['action'])

        # Separate optimizers for Q1 and Q2
        self.optimizer_Q1 = optim.Adam(
            list(self.full1.parameters()) +
            list(self.full2.parameters()) +
            list(self.full3.parameters()) +
            list(self.final1.parameters()), lr=lr
        )
        self.optimizer_Q2 = optim.Adam(
            list(self.full4.parameters()) +
            list(self.full5.parameters()) +
            list(self.full6.parameters()) +
            list(self.final2.parameters()), lr=lr
        )

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        if self.layerNorm:
            Q1 = F.relu(self.layer_norm1(self.full1(x)))
            Q1 = F.relu(self.layer_norm2(self.full2(Q1)))
            Q1 = F.relu(self.layer_norm3(self.full3(Q1)))
            Q1 = self.final1(Q1)

            Q2 = F.relu(self.layer_norm4(self.full4(x)))
            Q2 = F.relu(self.layer_norm5(self.full5(Q2)))
            Q2 = F.relu(self.layer_norm6(self.full6(Q2)))
            Q2 = self.final2(Q2)

        else:
            Q1 = F.relu(self.full1(x))
            Q1 = F.relu(self.full2(Q1))
            Q1 = F.relu(self.full3(Q1))
            Q1 = self.final1(Q1)

            Q2 = F.relu(self.full4(x))
            Q2 = F.relu(self.full5(Q2))
            Q2 = F.relu(self.full6(Q2))
            Q2 = self.final2(Q2)

        return Q1, Q2

    def update_Q1(self, loss):
        self.optimizer_Q1.zero_grad()
        loss.backward()
        self.optimizer_Q1.step()

    def update_Q2(self, loss):
        self.optimizer_Q2.zero_grad()
        loss.backward()
        self.optimizer_Q2.step()

    def saveCheckpoint(self, model_name):
        torch.save(self.state_dict(), model_name + '_critic.pt')

    def loadCheckpoint(self, model_name):
        self.load_state_dict(torch.load(model_name + '_critic.pt', map_location=self.device))
        

class Actor(nn.Module):
    def __init__(self, lr, env_params, layer_norm):
        super(Actor, self).__init__()

        self.layerNorm = layer_norm

        self.full1 = nn.Linear((env_params['obs'] + env_params['goal'])*env_params['sat_num'], 256)
        nn.init.kaiming_uniform_(self.full1.weight, a=0.01, mode='fan_in', nonlinearity='relu')

        self.layer_norm1 = nn.LayerNorm(256)

        self.full2 = nn.Linear(256, 256)
        nn.init.kaiming_uniform_(self.full2.weight, a=0.01, mode='fan_in', nonlinearity='relu')

        self.layer_norm2 = nn.LayerNorm(256)

        self.full3 = nn.Linear(256, 256)
        nn.init.kaiming_uniform_(self.full3.weight, a=0.01, mode='fan_in', nonlinearity='relu')

        self.layer_norm3 = nn.LayerNorm(256)

        self.final = nn.Linear(256, env_params['action'])

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x, mask=None, use_mask=None):
        if self.layerNorm:

            x = F.relu(self.layer_norm1(self.full1(x)))
            x = F.relu(self.layer_norm2(self.full2(x)))
            x = F.relu(self.layer_norm3(self.full3(x)))

        else:

            x = F.relu(self.full1(x))
            x = F.relu(self.full2(x))
            x = F.relu(self.full3(x))
        x = self.final(x)

        if use_mask is None:
            use_mask = mask is not None  # 如果提供了mask，默认启用mask

        if use_mask:
            if mask is None:
                raise ValueError("启用mask时必须提供mask参数")
            return self.masked_softmax(x, mask)
        else:
            return F.softmax(x, dim=-1)

    def masked_softmax(self, x, mask):
        mask_tensor = torch.from_numpy(mask).to(x.device, dtype=torch.float32)
        x = x + (1 - mask_tensor) * -1e11  # 让非法动作的概率接近 0
        return F.softmax(x, dim=-1)

    def saveCheckpoint(self, model_name):
        torch.save(self.state_dict(), model_name + '_actor.pt')

    def loadCheckpoint(self, model_name):
        self.load_state_dict(torch.load(model_name + '_actor.pt', map_location=self.device))
