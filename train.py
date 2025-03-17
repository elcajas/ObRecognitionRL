import os, sys, socket
from datetime import datetime
from tqdm import tqdm
import argparse
import pathlib, yaml, logging
from omegaconf import OmegaConf
import wandb

import torch
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

from envs.utils import make_env
from agents import PPOagent

def main(cfg):
    
    # Set the device for training (GPU if available, otherwise CPU)
    torch.cuda.set_device(cfg.agent.cuda_number)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create multiple environments for parallel training
    num_envs = cfg.agent.num_envs
    envs = gym.vector.AsyncVectorEnv([make_env(cfg.agent.task, cfg.agent.seed + i) for i in range(num_envs)])

    # Initialize the PPO agent
    agent = PPOagent(envs, cfg, device)

    # Compute batch sizes and the number of updates required
    num_steps = cfg.agent.num_steps
    batch_size = int(num_steps * num_envs)
    num_updates  = cfg.agent.total_timesteps // batch_size
    
    global_step = 0
    initial_update = 0

    # Initialize environment and obtain first observation for training
    obs, _ = envs.reset()
    obs, frame = agent.process_obs(obs)
    next_done = torch.zeros(num_envs)

    # Main training loop
    for update in range(initial_update, initial_update + num_updates):

        # Collect experiences for `num_steps` before updating the policy (roll-out phase)
        for step in tqdm(range(num_steps), desc=f'Update {update+1}/{initial_update + num_updates} ', unit='step', file=sys.stdout):
            global_step += 1 * num_envs

            action, logprob, _, val = agent.select_action(obs)
            next_obs, reward, done, _, info = envs.step(action.cpu().numpy())
            agent.store_experience(obs, action, logprob, torch.tensor(reward), next_done, val.squeeze(), frame)

            obs, frame = agent.process_obs(next_obs)
            next_done = torch.Tensor(done).to(device)

            # Log episode statistics when an episode ends
            if "final_info" in info:
                for ind, agent_info in enumerate(info["final_info"]):
                    if agent_info is not None:
                        ep_rew = agent_info["episode"]["r"]
                        ep_len = agent_info["episode"]["l"]

                        logging.info(f"global step: {global_step}, agent_id={ind}, reward={ep_rew[-1]}, length={ep_len[-1]}")
                        writer.add_scalar("charts/episodic_return", ep_rew, global_step)
                        writer.add_scalar("charts/episodic_length", ep_len, global_step)

        # Perform PPO policy update after collecting experiences (policy update phase)
        agent.learn(last_obs=obs, last_done=next_done, writer=writer, global_step=global_step)

        # Save model periodically
        save_interval = max(1, num_updates // 40)  # Ensure interval is at least 1
        if (update + 1) % save_interval == 0 or num_updates < 40:
            agent.save_model(update + 1)

    envs.close()

if __name__ == "__main__":

    # Parse command-line arguments to get the configuration file path
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    args = parser.parse_args()

    # Load the YAML configuration file into an OmegaConf dictionary
    dir_path = pathlib.Path(__file__).parent.resolve()
    with open(dir_path.joinpath(args.config), "r") as f:
        cfg = yaml.safe_load(f)
    cfg = OmegaConf.create(cfg)

    # Create a unique results directory name based on task and timestamp
    dname = f"{cfg.agent.task.replace(' ', '_')}_{datetime.now().strftime('%m_%d-%H:%M')}"
    suf_add = 'phase1' if not cfg.agent.train_image_model else 'phase2'
    
    # Initialize Weights & Biases (WandB) for experiment tracking if enabled
    if cfg.agent.wandb_init:
        wandb.init(
            project=f"{socket.gethostname()}_{suf_add}",
            entity=None,
            sync_tensorboard=True,
            config=dict(cfg.agent),
            name=dname,
        )

    # Set up results directory and save configuration
    results_dir = f"results/{suf_add}/{dname}"
    cfg.agent.results_dir = results_dir
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    OmegaConf.save(cfg, results_dir + '/config.yaml')

    # Initialize TensorBoard logging
    writer = SummaryWriter(results_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in cfg.agent.items()])),
    )

    # Redirect standard error output to a log file
    sys.stderr = open(results_dir+'/err.e', 'w')

    # Configure logging to write detailed output to a log file
    log_file = f"{results_dir}/output.log"
    logging.basicConfig(
        filename=log_file,
        format="[%(asctime)s] [%(levelname)8s] --- %(message)s (%(filename)s:%(lineno)s)", datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        filemode='w'
    )

    main(cfg)
    writer.close()