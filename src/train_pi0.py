"""Training script for Pi0 policy using LeRobot."""

import importlib
import logging
import os
import random
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

# Force PyAV as video backend to avoid torchcodec FFmpeg issues
os.environ["LEROBOT_VIDEO_BACKEND"] = "pyav"

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from lerobot.configs.default import DatasetConfig, WandBConfig
from lerobot.configs.train import TrainPipelineConfig

log = logging.getLogger(__name__)


def train(cfg: DictConfig) -> None:
    log.info("Starting training for Pi0...")
    log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Patch LeRobot to force PyAV backend
    try:
        from lerobot.datasets import video_utils

        original_decode = video_utils.decode_video_frames

        def patched_decode_video_frames(
            video_path, timestamps, tolerance_s, backend=None
        ):
            # Force PyAV backend regardless of what was requested
            return video_utils.decode_video_frames_pyav(
                video_path, timestamps, tolerance_s
            )

        video_utils.decode_video_frames = patched_decode_video_frames
        log.info("Successfully patched video backend to use PyAV")
    except Exception as e:
        log.warning(f"Could not patch video backend: {e}")

    # Import Pi0 policy configuration from LeRobot
    from lerobot.policies.pi0.configuration_pi0 import PI0Config

    # Create PI0Config instance with settings from hydra config
    policy_cfg = PI0Config()

    # Override with custom settings from your config if they exist
    if hasattr(cfg.policy, "pretrained_path"):
        policy_cfg.pretrained_path = cfg.policy.pretrained_path
    if hasattr(cfg.policy, "compile_model"):
        policy_cfg.compile_model = cfg.policy.compile_model
    if hasattr(cfg.policy, "gradient_checkpointing"):
        policy_cfg.gradient_checkpointing = cfg.policy.gradient_checkpointing
    if hasattr(cfg.policy, "dtype"):
        policy_cfg.dtype = cfg.policy.dtype
    if hasattr(cfg.policy, "device"):
        policy_cfg.device = cfg.policy.device

    # Handle hub configuration
    if hasattr(cfg, "repo_id"):
        policy_cfg.repo_id = cfg.repo_id
    if hasattr(cfg, "push_to_hub"):
        policy_cfg.push_to_hub = cfg.push_to_hub

    dataset_path = Path(cfg.dataset.root)

    log.info(f"Loading local dataset from: {dataset_path}")

    meta_dir = dataset_path / "meta"
    info_file = meta_dir / "info.json"

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found at: {dataset_path}\n")

    if not info_file.exists():
        raise FileNotFoundError(f"Dataset metadata not found at: {info_file}\n")

    dataset_cfg = DatasetConfig(
        repo_id=cfg.dataset.repo_id,
        root=str(dataset_path),
        video_backend="pyav",
    )

    wandb_cfg = WandBConfig(
        enable=cfg.wandb.enable,
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        mode=cfg.wandb.mode,
    )

    train_cfg = TrainPipelineConfig(
        policy=policy_cfg,
        dataset=dataset_cfg,
        batch_size=cfg.training.batch_size,
        steps=cfg.training.steps,
        save_freq=cfg.training.save_freq,
        log_freq=cfg.training.log_freq,
        wandb=wandb_cfg,
    )

    # Handle pretrained model loading if specified
    if hasattr(cfg, "pretrained_policy_path") and cfg.pretrained_policy_path:
        log.info(f"Note: Pretrained weights from: {cfg.pretrained_policy_path}")
        log.info(
            "Pi0 pretrained models should be loaded via policy.pretrained_path config parameter"
        )

    # Use LeRobot's standard training script
    lerobot_train_module = importlib.import_module("lerobot.scripts.lerobot_train")
    lerobot_train_module.init_logging()
    lerobot_train_module.train(train_cfg)

    log.info("Training completed!")


@hydra.main(
    config_path="../configs", config_name="train_pi0_config", version_base="1.3"
)
def main(cfg: DictConfig) -> None:
    """Main entry point for training with Hydra configuration."""
    # Set seed if specified
    if hasattr(cfg, "seed"):
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)

    # Override data_dir if provided as command line argument
    if "data_dir" in cfg:
        cfg.dataset.root = cfg.data_dir

    # Start training
    train(cfg)


if __name__ == "__main__":
    main()
