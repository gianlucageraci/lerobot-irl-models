from dataclasses import dataclass, field
from typing import Dict, List

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature


@PreTrainedConfig.register_subclass("flower")
@dataclass
class FlowerVLAConfig(SmolVLAConfig):
    #From SmolVLA -> check if SmolVLAConfig can be dropped
    n_obs_steps: int = 1
    chunk_size: int = 16
    n_action_steps: int = 16

    # Modalities
    obs_modalities: str = "observation"
    goal_modalities: str = "task"
    target_modality: str = "action"
    lang_modalities: List[str] = field(default_factory=lambda: ["language_instruction"])
    img_modalities: List[str] = field(default_factory=lambda: ["image_primary"])

    # Features
    input_features: Dict[str, PolicyFeature] = field(default_factory=lambda: {
        "observation.images.right_cam": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        "observation.images.wrist_cam": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        "observation.images.left_cam": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
        "task": PolicyFeature(type=FeatureType.LANGUAGE, shape=(1,)),
    })
    output_features: Dict[str, PolicyFeature] = field(default_factory=lambda: {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(8,))
    })

    # Normalization mapping (overrides SmolVLAConfig defaults)
    normalization_mapping: Dict[FeatureType, NormalizationMode] = field(default_factory=lambda: {
        FeatureType.STATE: NormalizationMode.MEAN_STD,
        FeatureType.ACTION: NormalizationMode.MEAN_STD,
    })

    # VLM configuration
    vlm_path: str = "microsoft/Florence-2-large"
    freeze_florence: bool = True
    freeze_vision_tower: bool = True
    freeze_embeddings_only: bool = True
    vlm_prompt_style: str = "default"
    token_dropout: float = 0.1
    cfg_dropout: float = 0.0
    cfg_lambda: float = 1.0

    # Action & observation config
    action_dim: int = 8
    act_window_size: int = 16
    multistep: int = 16
    num_sampling_steps: int = 4
    sampling_type: str = "uniform"
    lowdim_obs_dim: int = 16
    use_proprio: bool = True

    # Image configuration
    use_second_view: bool = True
    second_view_key: str = "image_secondary"

    # DiT architecture
    dit_dim: int = 1024
    n_heads: int = 16
    n_layers: int = 12
    attn_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    mlp_pdrop: float = 0.1

    # Attention
    use_cross_attn: bool = True
    use_causal_attention: bool = True
    use_adaln_cond: bool = False
    action_type_adaln: bool = True
    use_readout_token: bool = False

    # Positional encoding
    use_rope: bool = True
    use_nope: bool = False
    query_seq_len: int = 100
    rope_theta: float = 1000.0

    # Action output
    return_act_chunk: bool = False

    # Additional features
    use_action_scale: bool = False
    use_early_cross_fusion: bool = True

#     def get_optimizer_preset(self) -> AdamWConfig:
#         return AdamWConfig(
#             lr=2e-5,
#             betas=(0.9, 0.95),
#             eps=1e-8,
#             weight_decay=0.01,
#         )

#     def get_scheduler_preset(self):
#         return CosineDecayWithWarmupSchedulerConfig(
#             num_warmup_steps=1000,
#             num_decay_steps=400_000,
#             peak_lr=2e-5,
#             decay_lr=1e-5,
#         )

