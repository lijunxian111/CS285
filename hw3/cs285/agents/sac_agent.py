from collections import OrderedDict

from cs285.critics.bootstrapped_continuous_critic import \
    BootstrappedContinuousCritic
from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.infrastructure.utils import *
from cs285.infrastructure.sac_utils import *
from cs285.policies.MLP_policy import MLPPolicyAC
from .base_agent import BaseAgent
import torch
import gym
from cs285.policies.sac_policy import MLPPolicySAC
from cs285.critics.sac_critic import SACCritic
import cs285.infrastructure.pytorch_util as ptu

class SACAgent(BaseAgent):
    def __init__(self, env: gym.Env, agent_params):
        super(SACAgent, self).__init__()

        self.env = env
        self.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent_params = agent_params
        self.gamma = self.agent_params['gamma']
        self.critic_tau = 0.005
        self.learning_rate = self.agent_params['learning_rate']
        self.loss=nn.MSELoss()

        self.actor = MLPPolicySAC(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            self.agent_params['discrete'],
            self.agent_params['learning_rate'],
            action_range=self.action_range,
            init_temperature=self.agent_params['init_temperature']
        )
        self.actor_update_frequency = self.agent_params['actor_update_frequency']
        self.critic_target_update_frequency = self.agent_params['critic_target_update_frequency']

        self.critic = SACCritic(self.agent_params)
        self.critic_target = copy.deepcopy(self.critic).to(ptu.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.training_step = 0
        self.replay_buffer = ReplayBuffer(max_size=100000)

    def update_critic(self, ob_no, ac_na, next_ob_no, re_n, terminal_n):
        # TODO:
        #self.critic.optimizer.zero_grad()
        target_pro=self.critic_target.forward(next_ob_no,ac_na).squeeze()
        next_val=torch.min(target_pro[0],target_pro[1])+self.actor.log_alpha.exp()*self.actor.target_entropy
        target=re_n+self.gamma*next_val*(1.0-terminal_n)
        current_Q1,current_Q2=self.critic.forward(ob_no,ac_na)
        self.critic.optimizer.zero_grad()
        critic_loss=self.loss(current_Q1,target)
        critic_loss.backward()
        self.critic.optimizer.step()
        self.critic.optimizer.zero_grad()
        critic_loss = self.loss(current_Q2, target)
        critic_loss.backward()
        self.critic.optimizer.step()
        # 1. Compute the target Q value. 
        # HINT: You need to use the entropy term (alpha)
        # 2. Get current Q estimates and calculate critic loss
        # 3. Optimize the critic  
        return critic_loss

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        # TODO 
        # 1. Implement the following pseudocode:
        # for agent_params['num_critic_updates_per_agent_update'] steps,
        #     update the critic
        for _ in range(self.agent_params['num_critic_updates_per_agent_update']):
              loss_cri=self.update_critic(ob_no, ac_na, re_n, next_ob_no, terminal_n)

        # 2. Softly update the target every critic_target_update_frequency (HINT: look at sac_utils)
        for _ in range(self.critic_target_update_frequency):
             soft_update_params(self.critic,self.critic_target,self.critic_tau)
        # 3. Implement following pseudocode:
        # If you need to update actor
        for _ in range(self.agent_params['num_actor_updates_per_agent_update']):
             loss_actor,loss_alpha,tem=self.actor.update(ob_no,self.critic)
        #     update the actor

        # 4. gather losses for logging
        loss = OrderedDict()
        loss['Critic_Loss'] = loss_cri
        loss['Actor_Loss'] = loss_actor
        loss['Alpha_Loss'] = loss_alpha
        loss['Temperature'] = tem

        return loss

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_random_data(batch_size)
