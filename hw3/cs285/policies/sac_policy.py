from cs285.policies.MLP_policy import MLPPolicy
from cs285.infrastructure.sac_utils import SquashedNormal
import torch
import numpy as np
from cs285.infrastructure import sac_utils
from cs285.infrastructure import pytorch_util as ptu
from torch import distributions
from torch import nn
from torch import optim
import itertools
from torch.distributions import Normal
import torch.nn.functional as F

class MLPPolicySAC(MLPPolicy):
    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=3e-4,
                 training=True,
                 log_std_bounds=[-20,2],
                 action_range=[-1,1],
                 init_temperature=1.0,
                 **kwargs
                 ):
        super(MLPPolicySAC, self).__init__(ac_dim, ob_dim, n_layers, size, discrete, learning_rate, training, **kwargs)
        self.log_std_bounds = log_std_bounds
        self.action_range = action_range
        self.init_temperature = init_temperature
        self.learning_rate = learning_rate

        self.log_alpha = torch.tensor(np.log(self.init_temperature)).to(ptu.device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.learning_rate)

        self.target_entropy = -ac_dim

    @property
    def alpha(self):
        entropy=self.init_temperature
        # TODO: Formulate entropy term
        return entropy

    def get_action(self, obs: np.ndarray, sample=True) -> np.ndarray:

        # TODO: return sample from distribution if sampling
        # if not sampling return the mean of the distribution
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]

            # TODO return the action that the policy prescribes
            # print(type(observation))
        res,_ = self.forward(ptu.from_numpy(observation))
        if sample:
            return res.sample()
        else:
            return res.mean

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor):
        # TODO: Implement pass through network, computing logprobs and apply correction for Tanh squashing
        if self.discrete:
            logits = self.logits_na(observation)
            self.init_temperature=-logits
            action_distribution = distributions.Categorical(logits=logits)
            return action_distribution,logits
        else:
            mu=self.mean_net(observation)
            std=F.softplus(self.logstd(observation))
            dist=Normal(mu,std)
            normal_sample = dist.rsample()  # 在标准化正态分布上采样
            log_prob = dist.log_prob(normal_sample)  # 计算该值的标准正太分布上的概率
            action = torch.tanh(normal_sample)  # 对数值进行tanh
            # 计算tanh_normal分布的对数概率密度
            logits = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)  # 为了提升目标对应的概率值
            self.init_temperature=-logits
            action = action * self.action_range[1]  # 对action求取范围
        # HINT: 
        # You will need to clip log values
        # You will need SquashedNormal from sac_utils file 
        return action,logits

    def update(self, obs, critic):
        # TODO Update actor network and entropy regularizer
        # return losses and alpha value
        new_actions, log_prob = self.forward(obs)
        entropy = -log_prob
        q_value_1,q_value_2 = critic(obs, new_actions)
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - torch.min(q_value_1,q_value_2))
        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()

        # 更新alpha值
        alpha_loss = torch.mean(
            (entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()
        return actor_loss, alpha_loss, self.alpha