import numpy as np
import time
import os
import logging

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from mineclip.mineagent.batch import Batch
from agents.buffers import PPOBuffer
from agents.modules import PolicyModel, VisionModel

class PPOagent:
    """
    Proximal Policy Optimization (PPO) agent for reinforcement learning (similar to openAI's PPO2).

    This class handles training, policy updates, and model management for a PPO-based RL agent.
    """
    def __init__(self, env, cfg, device) -> None:

        num_steps = cfg.agent.num_steps
        batch_size = int(num_steps * cfg.agent.num_envs)
        num_minibatch = cfg.agent.num_minibatches
        assert batch_size % num_minibatch == 0, f"Number of samples: {batch_size} is not divisible by num_minibatches: {num_minibatch}"
        minibatch_size = int(batch_size // num_minibatch)
        num_updates  = cfg.agent.total_timesteps // batch_size

        self.batch_size = batch_size
        self.minibatch_size = minibatch_size

        self.epochs = cfg.agent.learning_epochs
        self.clip_coef = cfg.agent.clip_coef
        self.vf_coef = cfg.agent.vf_coef
        self.ent_coef = cfg.agent.ent_coef
        self.max_grad_norm = cfg.agent.max_grad_norm

        self.results_dir = cfg.agent.results_dir
        self.train_vision = cfg.agent.train_image_model

        self.bf = PPOBuffer(env, cfg, device)
        self.policy_model = PolicyModel(env, cfg, device).to(device)
        self.vision_model = VisionModel(env, cfg, device).to(device)

        self.optimizer = Adam([
            {'params': self.policy_model.parameters(), 'lr': cfg.agent.policy_learning_rate},
            {'params': self.vision_model.parameters(), 'lr': cfg.agent.vision_learning_rate},
        ])
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_updates, eta_min=cfg.agent.min_lr)
        self.env = env
        self.cfg = cfg
        self.device = device
        self.start_time = time.time()

        if cfg.agent.load_ppo_model:
            self.load_model(cfg.agent.ppo_checkpoint_path, cfg.agent.load_image_model, cfg.agent.image_checkpoint_path)
    
    def select_action(self, obs) -> torch.Tensor:
        with torch.no_grad():
            if self.train_vision:
                img_feat = self.vision_model.get_features(obs.rgb_feat)
                obs = Batch(rgb_feat=img_feat, compass=obs.compass, gps=obs.gps)
            return self.policy_model.get_action_and_value(obs)
    
    def store_experience(self, *args) -> None:
        self.bf.store(*args)
        
    def process_obs(self, obs):
        pitch = torch.deg2rad(torch.from_numpy(obs['pitch']))
        yaw = torch.deg2rad(torch.from_numpy(obs['yaw']))
        new_obs = {
            "compass": torch.cat((torch.sin(pitch), torch.cos(pitch), torch.sin(yaw), torch.cos(yaw)), dim=1),
            "gps": torch.tensor(obs['pos']),
        }

        if not self.train_vision:
            raw_rgb = obs['rgb'].copy()
            with torch.no_grad():
                rgb_feat = self.vision_model.get_features(torch.tensor(raw_rgb))
            new_obs["rgb_feat"] = rgb_feat
        
        else:
            new_obs["rgb_feat"] = torch.tensor(obs['rgb'])
        
        return Batch(**new_obs), obs['rgb'].transpose((0,2,3,1))
        
    def learn(self, last_obs, last_done, writer: SummaryWriter, global_step):

        with torch.no_grad():
            if self.train_vision:
                last_img_feat = self.vision_model.get_features(last_obs.rgb_feat)
                last_obs = Batch(rgb_feat=last_img_feat, compass=last_obs.compass, gps=last_obs.gps)

            last_value = self.policy_model.get_action_and_value(last_obs, get_action=False).reshape(1, -1)
            self.bf.calc_adv_and_return(last_value, last_done)
        
        b_obss, b_actions, b_logprobs, b_advantages, b_returns, b_values =  self.bf.get_batch()

        b_inds = np.arange(self.batch_size)
        clipfracs = []

        for epoch in range(self.epochs):
            np.random.shuffle(b_inds)

            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = b_inds[start:end]
                if end == self.batch_size:
                    logging.info(f"Update [{epoch+1}/{self.epochs}] for minibatch: [{end}/{self.batch_size}]")

                obss = b_obss[mb_inds]
                if self.train_vision:
                    img_feat = self.vision_model.get_features(obss.rgb_feat)
                    obss = Batch(rgb_feat=img_feat, compass=obss.compass, gps=obss.gps)

                _, newlogprob, entropy, newvalue = self.policy_model.get_action_and_value(obss, get_action=True, action=b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]
                
                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                mb_returns = b_returns[mb_inds]
                v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()
                
                # Entropy loss
                entropy_loss = entropy.mean()

                # Final Loss
                loss = pg_loss - self.ent_coef * entropy_loss +  self.vf_coef * v_loss
                
                # Bacward pass
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.max_grad_norm)
                self.optimizer.step()

        self.scheduler.step()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate", self.optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        # print("SPS:", int(global_step / (time.time() - self.start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - self.start_time)), global_step)

    def save_model(self, update):
        try:
            import loralib as lora
            # Create the directory if it doesn't exist
            dirpath = f"{self.results_dir}/checkpoints"
            os.makedirs(dirpath, exist_ok=True)
            ppo_filepath = os.path.join(dirpath, f"policy_{update}.pth")
            torch.save(self.policy_model.state_dict(), ppo_filepath)
            logging.info(f"Saving ppo model weights for update {update} in {ppo_filepath}.")

            if self.train_vision:
                im_filepath = os.path.join(dirpath, f"vision_{update}.pth")
                torch.save(lora.lora_state_dict(self.vision_model), im_filepath)
                logging.info(f"Saving image model weights for update {update} in {im_filepath}.")

        except Exception as e:
            print("Error occurred while saving model weights:", e)
            raise

    def load_model(self, policy_path, load_vision, vision_path):
        try:
            print(f"Loading ppo model weights from {policy_path}.")
            checkpoint = torch.load(policy_path, map_location='cpu')
            self.policy_model.load_state_dict(checkpoint)
            # start_update = checkpoint['update']
            logging.info(f"Loading ppo model weights from {policy_path}.")

            if load_vision:
                print(f"Loading image model weights from {vision_path}.")
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.scheduler.load_state_dict(checkpoint['scheduler'])
                self.vision_model.load_state_dict(torch.load(vision_path), strict=False)
                logging.info(f"Loading image model weights from {vision_path}.")

        except Exception as e:
            print("Error occurred while loading model weights:", e)
            raise
