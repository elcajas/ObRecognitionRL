import torch
from mineclip.mineagent.batch import Batch

class PPOBuffer:
    """
    A buffer for storing experience collected during environment interaction in PPO.

    This buffer stores observations, actions, rewards, log probabilities, values, and 
    other necessary information used for advantage estimation and PPO updates.
    """
    def __init__(self, env, cfg, device) -> None:
        self.cfg = cfg
        self.env = env
        capacity = cfg.agent.num_steps
        num_envs = cfg.agent.num_envs

        self.gamma = self.cfg.agent.gamma
        self.lambbda = self.cfg.agent.gae_lambda
        
        # set feature dimension depending on image model (mineclip: 512, gdino: (256, 900))
        # If train grounding dino, rgb pixel dimension is used (3,160,256)

        feat_dim = [3, 160, 256] if cfg.agent.train_image_model else [256, 900]
        obss = {
            "rgb_feat": torch.zeros((capacity, num_envs, *(feat_dim))),      
            "compass": torch.zeros((capacity, num_envs, 4)),
            "gps": torch.zeros((capacity, num_envs, 3)),
        }
        
        self.obss = Batch(**obss)
        self.actions = torch.zeros((capacity, num_envs) + env.single_action_space.shape).to(device)
        self.logprobs = torch.zeros((capacity, num_envs)).to(device)
        self.rewards = torch.zeros((capacity, num_envs)).to(device)
        self.dones = torch.zeros((capacity, num_envs)).to(device)
        self.values = torch.zeros((capacity, num_envs)).to(device)

        self.advantages = torch.zeros_like(self.rewards).to(device)
        self.returns = torch.zeros_like(self.rewards).to(device)

        self.ep_counter = torch.zeros((num_envs,)).to(device)
        self.pointer = 0
    
    def store(self, obs, action, logprob, reward, done, value, frame) -> None:
        
        self.obss[self.pointer] = obs
        self.actions[self.pointer] = action
        self.logprobs[self.pointer] = logprob
        self.rewards[self.pointer] = reward
        self.dones[self.pointer] = done
        self.values[self.pointer] = value
        self.pointer += 1

    def calc_adv_and_return(self, last_value, last_done):
        """
        Computes the Generalized Advantage Estimation (GAE) and returns for PPO.
        """

        rews = torch.zeros_like(self.rewards)
        rews[:-2] = self.rewards[2:]
        rews[-2:] = torch.zeros(2, self.rewards.size(1))

        lastgaelam = 0
        for t in reversed(range(self.pointer)):
            if t == self.pointer - 1:
                nextnonterminal = 1.0 - last_done
                nextvalues = last_value
            else:
                nextnonterminal = 1.0 - self.dones[t + 1]
                nextvalues = self.values[t + 1]
            delta = rews[t] + self.gamma * nextvalues * nextnonterminal - self.values[t]
            self.advantages[t] = lastgaelam = delta + self.gamma * self.lambbda * nextnonterminal * lastgaelam
        self.returns = self.advantages + self.values

        self.ep_counter += self.dones.sum(dim=0)
        self.pointer = 0

    def get_batch(self):
        b_obss = {}
        for key, value in self.obss.items():
            b_obss[key] = value.reshape((-1,) + value.shape[2:])

        b_obss = Batch(**b_obss)
        b_actions = self.actions.reshape((-1,) + self.env.single_action_space.shape)
        b_logprobs = self.logprobs.reshape(-1)
        b_advantages = self.advantages.reshape(-1)
        b_returns = self.returns.reshape(-1)
        b_values = self.values.reshape(-1)

        return b_obss, b_actions, b_logprobs, b_advantages, b_returns, b_values