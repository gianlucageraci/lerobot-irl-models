"""Evaluation script for Pi0 policy on real robot."""

import logging
import os
import random
import sys
from pathlib import Path

# Set protobuf implementation to pure Python to avoid compatibility issues
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import hydra
import numpy as np
import torch
import wandb
from lerobot.policies.factory import make_policy
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.policies.utils import FeatureType, PolicyFeature
from omegaconf import DictConfig, OmegaConf

from real_robot_env.real_robot_sim import RealRobot

log = logging.getLogger(__name__)

OmegaConf.register_new_resolver("add", lambda *numbers: sum(numbers))
OmegaConf.register_new_resolver("mul", lambda *numbers: np.prod(numbers))
torch.cuda.empty_cache()


def set_seed_everywhere(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


@hydra.main(
    config_path="../configs", config_name="eval_config.yaml", version_base="1.3"
)
def main(cfg: DictConfig) -> None:
    set_seed_everywhere(cfg.seed)

    # Initialize wandb
    wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        group=cfg.get("group", "eval"),
        mode=cfg.wandb.get("mode", "disabled"),
        config=wandb.config,
    )

    log.info(f"Loading pretrained model from directory {cfg.checkpoint_path}")
    agent = PI0Policy.from_pretrained(cfg.checkpoint_path)

    # Move agent to device
    agent = agent.to(cfg.device)
    agent.eval()

    # Initialize environment
    log.info("Initializing RealRobot environment...")
    env_sim = RealRobot(device=cfg.device)

    log.info("Starting evaluation on real robot...")
    env_sim.test_agent(agent)

    log.info("Evaluation completed")
    wandb.finish()


if __name__ == "__main__":
    main()
