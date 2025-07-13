#!/usr/bin/env python3
"""
vla_inferencer.py

Defines VLAController for ROS 2 Humble: given a raw LIBERO‐style observation,
runs the OpenVLA/LoRA‐fine-tuned model and returns one action.
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional, Union
from pathlib import Path

# Import core eval utilities from your existing script
from .run_libero_eval import (
    validate_config,
    initialize_model,
    prepare_observation,
    process_action,
)
from experiments.robot.robot_utils import (
    get_image_resize_size,
    get_action,
    set_seed_everywhere,
)


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = "moojink/openvla-7b-oft-finetuned-libero-10"     # Pretrained checkpoint path

    use_l1_regression: bool = True                   # If True, uses continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, uses continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps_train: int = 50              # (When `diffusion==True`) Number of diffusion steps used for training
    num_diffusion_steps_inference: int = 50          # (When `diffusion==True`) Number of diffusion steps used for inference
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 2                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = True                         # Whether to include proprio state in input

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 8                     # Number of actions to execute open-loop before requerying policy

    lora_rank: int = 32                              # Rank of LoRA weight matrix (MAKE SURE THIS MATCHES TRAINING!)

    unnorm_key: Union[str, Path] = ""                # Action un-normalization key

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "220"
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task
    initial_states_path: str = "DEFAULT"             # "DEFAULT", or path to initial states JSON file
    env_img_res: int = 256                           # Resolution for environment images (not policy input resolution)

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project

    seed: int = 7                                    # Random Seed (for reproducibility)
    h_decoding = True

    # fmt: on


class VLAController:
    """Wraps the inference pipeline from run_libero_eval as a ROS 2 controller."""

    def __init__(self, task_description: str = None,model_path: str = None):
        # 1) Load & validate config
        self.cfg = GenerateConfig()
        self.cfg.pretrained_checkpoint = model_path

        # 2) Fix randomness for reproducibility
        set_seed_everywhere(self.cfg.seed)

        # 3) Choose device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 4) Initialize model + heads + processor
        model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(self.cfg)
        model.to(self.device).eval()

        self.model = model
        self.action_head = action_head
        self.proprio_projector = proprio_projector
        self.noisy_action_projector = noisy_action_projector
        self.processor = processor

        # 5) Compute image‐resize size for policy inputs
        self.resize_size = get_image_resize_size(self.cfg)

        # 6) Task description used as “language prompt” to the policy
        #    Default to the LIBERO suite name if not provided
        if task_description is None:
            td = self.cfg.task_suite_name
            # if it's an Enum, get its value
            task_description = td.value if hasattr(td, "value") else str(td)
        self.task_description = task_description

    def predict(self, obs: dict) -> np.ndarray:
        """
        Given a raw LIBERO‐style observation `obs` (must contain the keys
        "full_image", "wrist_image", and the proprioceptive state under "state"),
        returns a single 1×D action as a NumPy array.
        """
        # 1) Prepare the observation for the VLA model
        observation, _ = prepare_observation(obs, self.resize_size)

        # 2) Run the model to get a chunk of actions
        actions = get_action(
            self.cfg,
            self.model,
            observation,
            self.task_description,
            processor=self.processor,
            action_head=self.action_head,
            proprio_projector=self.proprio_projector,
            noisy_action_projector=self.noisy_action_projector,
            use_film=self.cfg.use_film,
        )

        # 3) Take only the first action in the chunk
        raw_action = actions[0]

        # 4) Post‐process (normalize gripper, flip sign, etc.)
        proc_action = process_action(raw_action, self.cfg.model_family)

        # 5) Return as NumPy array
        return np.array(proc_action)