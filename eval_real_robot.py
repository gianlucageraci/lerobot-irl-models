import logging
import random
import os
import hydra
import numpy as np
import multiprocessing as mp

from tqdm import tqdm
import wandb
from omegaconf import DictConfig, OmegaConf
import torch

from agents.utils.sim_path import sim_framework_path

log = logging.getLogger(__name__)

OmegaConf.register_new_resolver(
    "add", lambda *numbers: sum(numbers)
)
OmegaConf.register_new_resolver(
    "mul", lambda *numbers: np.prod(numbers)
)
torch.cuda.empty_cache()


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


@hydra.main(
    config_path="configs", config_name="real_robot_config.yaml", version_base="1.3"
)
def main(cfg: DictConfig) -> None:
    set_seed_everywhere(cfg.seed)

    # init wandb logger and config from hydra path
    wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        group=cfg.group,
        mode="disabled",
        config=wandb.config
    )

    # load vqvae before training the agent: add path to the config file
    # train the agent
    agent = hydra.utils.instantiate(cfg.agents)

    agent.load_pretrained_model("/home/multimodallearning/model_wei/audio_earaser_task/17-21-03", sv_name='last_model')
    agent.load_model_scaler("/home/multimodallearning/model_wei/audio_earaser_task/17-21-03")

    env_sim = hydra.utils.instantiate(cfg.simulation)
    env_sim.test_agent(agent)

    log.info("Training done")
    log.info("state_dict saved in {}".format(agent.working_dir))
    wandb.finish()


if __name__ == "__main__":
    main()
