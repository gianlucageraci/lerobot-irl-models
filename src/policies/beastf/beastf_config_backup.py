from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig


@PreTrainedConfig.register_subclass("beast_vla")
class BeastVLAConfig(SmolVLAConfig):
    # -------------------------------------------------------------------------
    # Input / Output Definition
    # -------------------------------------------------------------------------
    input_features: Dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            "observation.images.right_cam": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 256, 256)
            ),
            "observation.images.wrist_cam": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 256, 256)
            ),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
        }
    )
    
    output_features: Dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
        }
    )

    normalization_mapping: Dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "observation.images.right_cam": NormalizationMode.MEAN_STD,
            "observation.images.wrist_cam": NormalizationMode.MEAN_STD,
            "observation.state": NormalizationMode.MEAN_STD,
            "action": NormalizationMode.MEAN_STD,
        }
    )

    # -------------------------------------------------------------------------
    # Modality Keys (Used by Model to look up batch keys)
    # -------------------------------------------------------------------------
    # Helper to identify which key in the batch contains the primary image, text, etc.
    obs_modalities: Dict[str, str] = field(default_factory=dict) # Placeholder if needed
    goal_modalities: Dict[str, str] = field(default_factory=dict) # Placeholder if needed
    target_modality: str = "action"
    
    # Specific list of keys for the model to process
    img_modalities: List[str] = field(
        default_factory=lambda: ["observation.images.right_cam"]
    )
    lang_modalities: List[str] = field(
        default_factory=lambda: ["text"]
    )
    
    # -------------------------------------------------------------------------
    # VLM (Florence-2) Configuration
    # -------------------------------------------------------------------------
    vlm_path: str = "microsoft/Florence-2-base"
    
    # Freezing strategies to save memory/compute
    freeze_florence: bool = False          # Freeze entire VLM (usually False for training)
    freeze_vision_tower: bool = True       # Freeze just the image encoder
    freeze_embeddings_only: bool = True    # Freeze input embeddings
    
    # Prompt engineering for Florence
    vlm_prompt_style: str = "default"      # "default", "feature_focused", "state_oriented"
    token_dropout: float = 0.1             # Dropout on VLM tokens

    # -------------------------------------------------------------------------
    # Beast Tokenizer Configuration (B-Splines)
    # -------------------------------------------------------------------------
    # Action Space dimensions
    num_dof: int = 7                       # Must match output_features shape[0]
    
    # Spline Parameters
    num_basis: int = 5                     # Number of control points per spline
    degree_p: int = 4                      # Degree of B-spline
    
    # Discretization
    action_bins: int = 256                 # Vocab size for action tokens
    
    # Training Bounds
    update_w_bound: bool = False           # Dynamically update weight bounds during training
    # Optional explicit bounds if not updating dynamically
    action_bounds: Optional[Tuple[float, float]] = None 

    # -------------------------------------------------------------------------
    # Inference / Action Chunking
    # -------------------------------------------------------------------------
    act_window_size: int = 16              # Length of action chunk to predict
    multistep: int = 8                     # How often to re-plan (inference only)
    return_act_chunk: bool = False         # If True, returns [B, T, D], else [B, D]

    # -------------------------------------------------------------------------
    # Additional Architecture Flags
    # -------------------------------------------------------------------------
    use_second_view: bool = True
    second_view_key: str = "observation.images.wrist_cam"
    use_proprio: bool = False              # Whether to inject proprio state into VLM embeddings

    def __post_init__(self):
        """
        Post-initialization validation.
        """
        super().__post_init__()
        self.validate_features()

    def validate_features(self):
        """
        Check that configuration dimensions match feature shapes.
        """
        if "action" in self.output_features:
            act_shape = self.output_features["action"].shape
            if act_shape[0] != self.num_dof:
                raise ValueError(
                    f"Config num_dof ({self.num_dof}) does not match "
                    f"output feature shape {act_shape}."
                )

    def get_optimizer_preset(self) -> AdamWConfig:
        """
        Presets for AdamW. VLM fine-tuning usually requires lower LR.
        """
        return AdamWConfig(
            lr=2e-5,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-4,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            num_warmup_steps=1000,
            num_decay_steps=100_000,
            peak_lr=2e-5,
            decay_lr=1e-6,
        )