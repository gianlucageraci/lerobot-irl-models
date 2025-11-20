import logging
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from lerobot.policies.pretrained import PreTrainedPolicy
from timm.layers.mlp import Mlp
from torchdiffeq import odeint
from transformers import AutoModelForCausalLM, AutoProcessor

from .beast_config import BeastVLAConfig
from .beast_tokenizer.beast import BeastTokenizer

from lerobot.processor.normalize_processor import (
    NormalizerProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.utils.constants import ACTION

logger = logging.getLogger(__name__)


class BeastVLAPolicy(PreTrainedPolicy):
    """
    BeastVLA Policy for LeRobot.

    Combines Florence-2 VLM with Beast Tokenizer for learning generalist manipulation policies.

    Key Features:
    - Multi-view image observations
    - Language goal conditioning
    - B-spline based action representation
    - Action chunking (predicts multiple actions)
    - Multiple action spaces (eef_delta, joint_single, bimanual_nav)
    """

    name = "beast_vla"
    config_class = BeastVLAConfig

    def __init__(
        self,
        config: BeastVLAConfig,
        dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    ):
        """
        Initialize BeastVLA Policy.

        Args:
            config: BeastVLAConfig instance with all hyperparameters
        """
        super().__init__(config)

        # If dataset_stats not provided, try to get from config
        if dataset_stats is None and hasattr(config, '_dataset_stats'):
            dataset_stats = config._dataset_stats
            logger.info("📊 Using dataset_stats from config")

        config.validate_features()
        self.config = config
        self.normalize_inputs = NormalizerProcessorStep(
            config.input_features, config.normalization_mapping, dataset_stats
        )
        self.normalize_targets = NormalizerProcessorStep(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        self.unnormalize_outputs = UnnormalizerProcessorStep(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        self.model = BeastFModel(config)

        self.model.reset()

    def forward(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass (training mode) - returns (loss, output_dict) for LeRobot training loop.

        Args:
            batch: Dictionary with observation and action data

        Returns:
            Tuple of (loss tensor, output dictionary with metrics)
        """     
        result = self.model.forward(batch)
        return result["loss"], result["loss_dict"]

    def encode_observations(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Delegate to model."""
        return self.model.encode_observations(batch)

    def rf_loss(
        self, cond: dict, actions: torch.Tensor, dataset_idx: Any = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Delegate to model."""
        return self.model.rf_loss(cond, actions, dataset_idx)

    def sample_actions(
        self, z: torch.Tensor, cond: Dict[str, torch.Tensor], inference: bool = False
    ) -> torch.Tensor:
        """Delegate to model."""
        return self.model.sample_actions(z, cond, inference)

    def compute_loss(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute loss for training (LeRobot protocol).

        Args:
            batch: Training batch

        Returns:
            - Loss tensor
            - Dictionary with loss statistics
        """
        result = self.forward(batch)
        return result["loss"], result["loss_dict"]

    def get_optim_params(self) -> dict:
        return self.parameters()

    @torch.no_grad()
    def predict_action_chunk(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Predict a chunk of actions given environment observations.

        Args:
            batch: Dictionary with observation data

        Returns:
            Action chunk [B, act_window_size, action_dim]
        """
        # Encode observations
        cond = self.model.encode_observations(batch)

        # Sample noise
        B = cond["features"].shape[0]
        noise = torch.randn(
            B,
            self.config.act_window_size,
            self.config.action_dim,
            device=self.model.device,
        )

        # Sample actions
        action_seq = self.model.sample_actions(noise, cond, inference=True)
        action_seq = self.unnormalize_outputs({ACTION: action_seq})[ACTION]
        return action_seq  # [B, T, action_dim]

    def reset(self) -> None:
        """Reset rollout state."""
        self.model.reset()

    @torch.no_grad()
    def select_action(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Select action for inference (LeRobot protocol).
        This method handles action chunking internally:
        - On first call (or every multistep): predicts new action chunk
        - Returns single action per timestep from the chunk

        Args:
            batch: Observation batch

        Returns:
            Selected action [B, action_dim]
        """
        if ACTION in batch:
            batch.pop(ACTION)
        batch = self.normalize_inputs(batch)
        # Check if we need to predict a new action chunk
        if (
            self.model.rollout_step_counter % self.config.multistep == 0
            or self.model.pred_action_seq is None
        ):
            # Predict new action chunk
            self.model.pred_action_seq = self.predict_action_chunk(batch)

        # Get current action from the chunk
        if self.config.return_act_chunk:
            # Return full chunk
            action = self.model.pred_action_seq
        else:
            # Return single action at current step
            # action = self.model.pred_action_seq
            action = self.model.pred_action_seq[:, self.model.rollout_step_counter, :]

        # Update counter
        self.model.rollout_step_counter += 1
        if self.model.rollout_step_counter >= self.config.multistep:
            self.model.rollout_step_counter = 0

        return action


class BeastFModel(nn.Module):
    """
    BeastVLA Model combining Florence-2 VLM with Flow-based DiT.
    """

    def __init__(self, config: BeastVLAConfig):
        super().__init__()
        # Core attributes
        self.device = torch.device(
            config.device
            if hasattr(config, "device") and config.device
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Initialize configuration groups.
        self._init_modalities(
            config.target_modality,
            config.obs_modalities,
            config.goal_modalities,
            config.img_modalities,
            config.lang_modalities,
        )
        self._init_dimensions(
            config.dit_dim,
            config.n_heads,
            config.lowdim_obs_dim,
            config.action_dim,
            config.act_window_size,
            config.multistep,
            config.num_sampling_steps,
        )
        self._init_flags(
            config.use_second_view,
            config.use_causal_attention,
            config.use_cross_attn,
            config.use_adaln_cond,
            config.use_readout_token,
            config.use_rope,
            config.use_nope,
            config.vlm_prompt_style,
            config.token_dropout,
            config.action_type_adaln,
            config.sampling_type,
            config.use_proprio,
            config.return_act_chunk,
            config.second_view_key,
            config.cfg_dropout,
            config.cfg_lambda,
        )

        logger.info("Configuration (modalities, dimensions, flags) initialized.")

        self._setup_vlm(
            config.vlm_path,
            config.freeze_vision_tower,
            config.freeze_florence,
            config.freeze_embeddings_only,
        )
        hidden_dim = self.vlm.config.text_config.d_model
        self.vlm_latent_dim = hidden_dim

        self._setup_action_tokenizer(config)


        # Initialize rollout state.
        self.rollout_step_counter = 0
        self.pred_action_seq = None

        # Ensure that all parameters and buffers are on the correct device.
        self.ensure_device_consistency()

    # === Initialization Helpers ===
    def _init_modalities(
        self,
        target_modality: str,
        obs_modalities: str,
        goal_modalities: str,
        img_modalities: List[str],
        lang_modalities: List[str],
    ) -> None:
        """Initializes modality-related attributes."""
        self.target_modality = target_modality
        self.obs_modalities = obs_modalities
        self.goal_modalities = goal_modalities
        self.img_modalities = img_modalities
        self.lang_modalities = lang_modalities

    def _init_dimensions(
        self,
        dit_dim: int,
        n_heads: int,
        lowdim_obs_dim: int,
        action_dim: int,
        act_window_size: int,
        multistep: int,
        num_sampling_steps: int,
    ) -> None:
        """Initializes dimension-related attributes and checks consistency."""
        if dit_dim % n_heads != 0:
            raise ValueError(
                f"dit_dim ({dit_dim}) must be divisible by n_heads ({n_heads})"
            )
        self.lowdim_obs_dim = lowdim_obs_dim
        self.action_dim = action_dim
        self.act_window_size = act_window_size
        self.multistep = multistep
        self.num_sampling_steps = num_sampling_steps
        self.dit_dim = dit_dim

    def _init_flags(
        self,
        use_second_view: bool,
        use_causal_attention: bool,
        use_cross_attn: bool,
        use_adaln_cond: bool,
        use_readout_token: bool,
        use_rope: bool,
        use_nope: bool,
        vlm_prompt_style: str,
        token_dropout: float,
        action_type_adaln: bool,
        sampling_type: str,
        use_proprio: bool,
        return_act_chunk: bool,
        second_view_key: str,
        cfg_dropout: float,
        cfg_lambda: float,
    ) -> None:
        """Initializes boolean flags and related parameters."""
        if vlm_prompt_style not in ["default", "feature_focused", "state_oriented"]:
            raise ValueError("Invalid VLM prompt style")
        if sampling_type not in [
            "ln",
            "pi_zero",
            "loglogistic",
            "uniform",
            "stratified",
        ]:
            raise ValueError(f"Invalid sampling type: {sampling_type}")
        self.use_second_view = use_second_view
        self.use_causal_attention = use_causal_attention
        self.use_cross_attn = use_cross_attn
        self.use_adaln_cond = use_adaln_cond
        self.use_readout_token = use_readout_token
        self.use_rope = use_rope
        self.use_nope = use_nope
        self.use_proprio = use_proprio
        self.return_act_chunk = return_act_chunk
        self.vlm_prompt_style = vlm_prompt_style
        self.token_dropout = token_dropout
        self.action_type_adaln = action_type_adaln
        self.sampling_type = sampling_type
        self.second_view_key = second_view_key
        self.cfg_dropout = cfg_dropout
        self.cfg_lambda = cfg_lambda

    def _setup_vlm(
        self,
        vlm_path: str,
        freeze_vision_tower: bool,
        freeze_florence: bool,
        freeze_embeddings_only: bool,
    ) -> None:
        """
        Loads the pretrained VLM, sets up the processor/tokenizer, adds a prompt token,
        and optionally freezes parameters.
        """
        logger.info(f"Loading VLM from {vlm_path}")
        self.vlm = AutoModelForCausalLM.from_pretrained(
            vlm_path,
            trust_remote_code=True,
            attn_implementation="eager",  # Fix for Florence-2 SDPA issue
        )
        self.train_vlm = not freeze_florence

        if freeze_florence:
            for param in self.vlm.parameters():
                param.requires_grad = False
        elif freeze_embeddings_only:
            embedding_layer = self.vlm.get_input_embeddings()
            for param in embedding_layer.parameters():
                param.requires_grad = False
            if hasattr(self.vlm.language_model, "shared"):
                for param in self.vlm.language_model.shared.parameters():
                    param.requires_grad = False

        if not freeze_vision_tower:
            for param in self.vlm.vision_tower.parameters():
                param.requires_grad = True

        self.processor = AutoProcessor.from_pretrained(vlm_path, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer
        self.prompt_embeds = self._create_prompt_embed("<Flow>").to(self.device)
        self.vlm_token_dropout = nn.Dropout(self.token_dropout)
    
    def _setup_action_tokenizer(self, config: BeastVLAConfig) -> None:
        """Initializes the Beast action tokenizer."""
        self.action_tokenizer = BeastTokenizer(
            action_types=config.action_types,
            num_dof=config.num_dof,
            num_basis=config.num_basis,
            action_bounds=config.action_bounds,
            device=self.device,
        )
        self.update_w_bound = config.update_w_bound


    def _verify_device_consistency(self) -> None:
        """Verifies that all parameters and buffers are on the expected device."""
        expected = self.device
        inconsistent = []
        for name, param in self.named_parameters():
            if param.device != expected:
                inconsistent.append(f"{name}: {param.device} (expected {expected})")
        for name, buf in self.named_buffers():
            if buf.device != expected:
                inconsistent.append(
                    f"{name} (buffer): {buf.device} (expected {expected})"
                )
        if inconsistent:
            logger.warning("Device consistency issues: " + "; ".join(inconsistent))

    def ensure_device_consistency(self) -> None:
        """Moves the entire model (and buffers) to the designated device."""
        self.to(self.device)
        self.vlm.to(self.device)
        if not self.use_rope and hasattr(self, "positional_encoding"):
            self.positional_encoding = self.positional_encoding.to(self.device)
        if self.use_readout_token and hasattr(self, "register_token"):
            self.register_token = self.register_token.to(self.device)
        self._verify_device_consistency()

    def _create_prompt_embed(self, prompt_text: str) -> nn.Parameter:
        """
        Creates a prompt embedding. Adds the prompt token to the tokenizer
        and returns its embedding (frozen).
        """
        self.tokenizer.add_special_tokens({"additional_special_tokens": [prompt_text]})
        self.vlm.resize_token_embeddings(len(self.tokenizer))
        prompt_token_id = self.tokenizer.convert_tokens_to_ids(prompt_text)
        prompt_embed = nn.Parameter(
            self.vlm.get_input_embeddings()(torch.tensor(prompt_token_id)),
            requires_grad=False,
        )
        return prompt_embed.unsqueeze(0).unsqueeze(0)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass (training mode).

        Args:
            batch: Dictionary with observation and action data

        Returns:
            Dictionary with loss and other outputs
        """
        obs_features = self.encode_observations(batch)
        dataset_idx = batch.get("task.dataset_index", None)
        action_loss, losses_dict = self.rf_loss(
            obs_features, batch[self.target_modality], dataset_idx
        )

        return {"loss": action_loss, "loss_dict": losses_dict}

    def encode_observations(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        Encodes primary (and optional second view) image observations and text goals.
        Returns a dictionary with:
            - 'features': Encoder outputs.
            - 'frequency_embeds': Frequency embeddings.
            - 'action_space_embeds': Action space embeddings.
            - 'action_type': Action type indices.
            - 'proprio': Proprioception data (if available).
            - 'attention_mask': Attention mask.
        """
        device = self.device
        default_dtype = next(self.parameters()).dtype
        # Debug: print available keys
        image_tensor = batch["observation.images.right_cam"]
        # Handle both 4D [B, C, H, W] and 5D [B, T, C, H, W] image tensors
        if len(image_tensor.shape) == 4:
            # Shape is [B, C, H, W], add temporal dimension
            B, C, H, W = image_tensor.shape
            T = 1
            image_tensor = image_tensor.unsqueeze(1)  # [B, 1, C, H, W]
        else:
            B, T, C, H, W = image_tensor.shape

        image_features = self.vlm._encode_image(
            image_tensor.view(-1, C, H, W).to(device).to(default_dtype)
        )
        image_features = image_features.view(B, T * image_features.shape[1], -1)

        if self.use_second_view and "observation.images.wrist_cam" in batch:
            image2_tensor = batch["observation.images.wrist_cam"]

            # Handle both 4D and 5D for second view as well
            if len(image2_tensor.shape) == 4:
                image2_tensor = image2_tensor.unsqueeze(1)

            image2_features = self.vlm._encode_image(
                image2_tensor.view(-1, C, H, W).to(device).to(default_dtype)
            )
            image2_features = image2_features.view(B, T * image2_features.shape[1], -1)
            image_features = torch.cat([image_features, image2_features], dim=1)

        if "task" in batch:
            task_text = batch["task"]
            # task_text is already a list of strings
            if not isinstance(task_text, list):
                task_text = [task_text] * B

            # Tokenize the text
            tokenized = self.tokenizer(
                task_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            )
            text_embeds = self.vlm.get_input_embeddings()(
                tokenized["input_ids"].to(device)
            ).to(device)
            lang_attention_mask = tokenized["attention_mask"].to(device)
        else:
            dummy_text = [""] * B
            tokenized = self.tokenizer(
                dummy_text, return_tensors="pt", padding=True, max_length=128
            )
            text_embeds = self.vlm.get_input_embeddings()(
                tokenized["input_ids"].to(device)
            ).to(device)
            lang_attention_mask = tokenized["attention_mask"].to(device)

        # get the flow prompt for florence
        task_prompt = self.prompt_embeds.expand(B, -1, -1)
        merged_embeds = torch.cat(
            [
                task_prompt.to(image_features.device),
                image_features,
                text_embeds.to(image_features.device),
            ],
            dim=1,
        )

        # get attention mask from txt
        # lang_attention_mask was already created during tokenization above
        # define attention mask for image
        vis_attention_mask = torch.ones(
            image_features.shape[:2], device=image_features.device
        )
        prompt_mask = torch.zeros(B, 1, dtype=torch.bool, device=image_features.device)
        attention_mask = torch.cat(
            [prompt_mask, vis_attention_mask, lang_attention_mask], dim=1
        )

        features = self.vlm.get_encoder()(
            inputs_embeds=merged_embeds,
            attention_mask=attention_mask,
        ).last_hidden_state

        features = self.vlm_token_dropout(features)

        return {
            "features": features,
            "proprio": batch["observation.state"].to(device).to(default_dtype)
            if self.use_proprio and "observation.state" in batch
            else None,
            "attention_mask": attention_mask,
        }


    def encode_proprio(
        self, proprio: torch.Tensor, action_type: torch.Tensor, output_shape
    ) -> torch.Tensor:
        """
        Encodes proprioceptive data based on action type.
        Returns a tensor with shape [batch, dit_dim].
        """
        batch_size, _ = output_shape
        dtype = next(self.parameters()).dtype

        if not self.use_proprio:
            return torch.zeros(batch_size, self.dit_dim, device=self.device)

        encoded = (
                    self.proprio_encoder(proprio)
                    .squeeze(1)
                    .to(dtype)
                )

        return encoded
    
    def compute_llm_outputs(self, batch: Dict) -> torch.Tensor:

        features, encoder_attn_mask = self.compute_input_features(batch)

        llm_input_ids = None

        if "actions" in batch.keys():

            action_tokens, params = self.action_tokenizer.encode(batch["actions"], update_bounds=self.update_w_bound)

            llm_label_ids = self.action_tokenizer.tokens_to_llm_tokens(action_tokens)

            input_tokens = self.action_tokenizer.vocab_size//2 * torch.ones_like(llm_label_ids, dtype=torch.long, device=self.device)
            llm_input_ids = self.action_tokenizer.tokens_to_llm_tokens(input_tokens)

            
            ### Sanity Check, check if the reconstructed tokens are correct
            # for i in range(len(batch["actions"])):
                # self.action_tokenizer.visualize_reconstruction_error_with_llm_tokenizer(batch["actions"][i])

        bidirectional_mask = create_bidirectional_mask(
            batch_size=llm_label_ids.shape[0],
            seq_length=llm_label_ids.shape[1],
            device=self.device
        ) 

        decoder_outputs = self.vlm.get_decoder()(
            input_ids=llm_input_ids, # fix this, this should be the empty action token
            encoder_hidden_states=features,
            encoder_attention_mask=encoder_attn_mask,
            attention_mask=bidirectional_mask, # bidirectional attention for the decoder
            use_cache=True,
        )

        lm_logits = self.vlm.language_model.get_output_embeddings()(decoder_outputs[0])
        lm_logits = lm_logits + self.vlm.language_model.final_logits_bias.to(lm_logits.device)

        loss_fct = nn.CrossEntropyLoss()
        masked_lm_loss = loss_fct(
            lm_logits.view(-1, self.vlm.config.vocab_size),
            llm_label_ids.view(-1),
        )

        ### Add compute reconstruction loss
        token_predict_accuracy = 0.0
        action_mse = 0.0
        if "actions" in batch.keys():
            pred_tokens = torch.argmax(lm_logits, dim=-1)
            token_predict_accuracy = self.token_prediction_accuracy(pred_tokens, llm_label_ids)
            reconstruct_traj = self.action_tokenizer.reconstruct_from_llm_tokens(pred_tokens, times=None)
            action_mse = F.mse_loss(reconstruct_traj, batch["actions"])


        return {
            'llm_loss': masked_lm_loss,
            'token_predict_accuarcy': token_predict_accuracy,
            'reconstruct_action_mse': action_mse,
        }


    def llm_generates(self, batch: Dict) -> torch.Tensor:
        """Encode observations using Florence-2"""
        features, encoder_attn_mask = self.compute_input_features(batch)

        input_tokens = self.action_tokenizer.vocab_size//2 * torch.ones((1, self.num_dof, self.num_basis), 
                                                                        dtype=torch.long, device=self.device)
        llm_input_ids = self.action_tokenizer.tokens_to_llm_tokens(input_tokens)

        bidirectional_mask = create_bidirectional_mask(
            batch_size=llm_input_ids.shape[0],
            seq_length=llm_input_ids.shape[1],
            device=self.device
        ) 

        decoder_outputs = self.vlm.get_decoder()(
            input_ids=llm_input_ids, # fix this, this should be the empty action token
            encoder_hidden_states=features,
            encoder_attention_mask=encoder_attn_mask,
            attention_mask=bidirectional_mask, # bidirectional attention for the decoder
            use_cache=True,
        )

        lm_logits = self.vlm.language_model.get_output_embeddings()(decoder_outputs[0])
        lm_logits = lm_logits + self.vlm.language_model.final_logits_bias.to(lm_logits.device)

        output_tokens = torch.argmax(lm_logits, dim=-1)

        return output_tokens



    def reset(self) -> None:
        """
        Resets the rollout state.
        """
        self.rollout_step_counter = 0
        self.pred_action_seq = None
        self.eval()
