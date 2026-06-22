"""
Custom Policy configuration template

Usage steps:
1. Copy this file and rename it, e.g. g1_my_rl_policy_cfg.py
2. Modify the class name and parameters
3. Import and use it in g1_custom_cfg.py
"""

from robojudo.config import Config
from robojudo.policy.policy_cfgs import PolicyCfg
from robojudo.tools.tool_cfgs import DoFConfig


# If you use 29 joints (including torso and arms)
class G1MyCustomFullBodyDoF(DoFConfig):
    """29 DoF full-body configuration"""
    
    joint_names: list[str] = [
        "left_hip_pitch_joint",
        "right_hip_pitch_joint",
        "waist_yaw_joint",
        "left_hip_roll_joint",
        "right_hip_roll_joint",
        "waist_roll_joint",
        "left_hip_yaw_joint",
        "right_hip_yaw_joint",
        "waist_pitch_joint",
        "left_knee_joint",
        "right_knee_joint",
        "left_shoulder_pitch_joint",
        "right_shoulder_pitch_joint",
        "left_ankle_pitch_joint",
        "right_ankle_pitch_joint",
        "left_shoulder_roll_joint",
        "right_shoulder_roll_joint",
        "left_ankle_roll_joint",
        "right_ankle_roll_joint",
        "left_shoulder_yaw_joint",
        "right_shoulder_yaw_joint",
        "left_elbow_joint",
        "right_elbow_joint",
        "left_wrist_roll_joint",
        "right_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "right_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_wrist_yaw_joint",
    ]

    default_pos: list[float] = [
        -0.28,  # left_hip_pitch_joint
        -0.28,  # right_hip_pitch_joint
        0.0,    # waist_yaw_joint
        0.0,    # left_hip_roll_joint
        0.0,    # right_hip_roll_joint
        0.0,    # waist_roll_joint
        0.0,    # left_hip_yaw_joint
        0.0,    # right_hip_yaw_joint
        0.0,    # waist_pitch_joint
        0.5,    # left_knee_joint
        0.5,    # right_knee_joint
        0.35,   # left_shoulder_pitch_joint
        0.35,   # right_shoulder_pitch_joint
        -0.23,  # left_ankle_pitch_joint
        -0.23,  # right_ankle_pitch_joint
        0.16,   # left_shoulder_roll_joint
        -0.16,  # right_shoulder_roll_joint
        0.0,    # left_ankle_roll_joint
        0.0,    # right_ankle_roll_joint
        0.0,    # left_shoulder_yaw_joint
        0.0,    # right_shoulder_yaw_joint
        0.87,   # left_elbow_joint
        0.87,   # right_elbow_joint
        0.0,    # left_wrist_roll_joint
        0.0,    # right_wrist_roll_joint
        0.0,    # left_wrist_pitch_joint
        0.0,    # right_wrist_pitch_joint
        0.0,    # left_wrist_yaw_joint
        0.0,    # right_wrist_yaw_joint
    ]

    stiffness: list[float] = [
        40.18,  # left_hip_pitch_joint
        40.18,  # right_hip_pitch_joint
        40.18,  # waist_yaw_joint
        99.10,  # left_hip_roll_joint
        99.10,  # right_hip_roll_joint
        28.50,  # waist_roll_joint
        40.18,  # left_hip_yaw_joint
        40.18,  # right_hip_yaw_joint
        28.50,  # waist_pitch_joint
        99.10,  # left_knee_joint
        99.10,  # right_knee_joint
        14.25,  # left_shoulder_pitch_joint
        14.25,  # right_shoulder_pitch_joint
        28.50,  # left_ankle_pitch_joint
        28.50,  # right_ankle_pitch_joint
        14.25,  # left_shoulder_roll_joint
        14.25,  # right_shoulder_roll_joint
        28.50,  # left_ankle_roll_joint
        28.50,  # right_ankle_roll_joint
        14.25,  # left_shoulder_yaw_joint
        14.25,  # right_shoulder_yaw_joint
        14.25,  # left_elbow_joint
        14.25,  # right_elbow_joint
        14.25,  # left_wrist_roll_joint
        14.25,  # right_wrist_roll_joint
        16.78,  # left_wrist_pitch_joint
        16.78,  # right_wrist_pitch_joint
        16.78,  # left_wrist_yaw_joint
        16.78,  # right_wrist_yaw_joint
    ]

    damping: list[float] = [
        2.56,  # left_hip_pitch_joint
        2.56,  # right_hip_pitch_joint
        2.56,  # waist_yaw_joint
        6.31,  # left_hip_roll_joint
        6.31,  # right_hip_roll_joint
        1.81,  # waist_roll_joint
        2.56,  # left_hip_yaw_joint
        2.56,  # right_hip_yaw_joint
        1.81,  # waist_pitch_joint
        6.31,  # left_knee_joint
        6.31,  # right_knee_joint
        0.91,  # left_shoulder_pitch_joint
        0.91,  # right_shoulder_pitch_joint
        1.81,  # left_ankle_pitch_joint
        1.81,  # right_ankle_pitch_joint
        0.91,  # left_shoulder_roll_joint
        0.91,  # right_shoulder_roll_joint
        1.81,  # left_ankle_roll_joint
        1.81,  # right_ankle_roll_joint
        0.91,  # left_shoulder_yaw_joint
        0.91,  # right_shoulder_yaw_joint
        0.91,  # left_elbow_joint
        0.91,  # right_elbow_joint
        0.91,  # left_wrist_roll_joint
        0.91,  # right_wrist_roll_joint
        1.07,  # left_wrist_pitch_joint
        1.07,  # right_wrist_pitch_joint
        1.07,  # left_wrist_yaw_joint
        1.07,  # right_wrist_yaw_joint
    ]


# ============================================================================
# Policy configuration
# ============================================================================

class G1MyCustomPolicyCfg(PolicyCfg):
    """
    Configuration class for the custom Policy

    All parameters should match those used during training
    """

    # ========== Basic configuration ==========

    # Robot type
    robot: str = "g1"

    # Policy type (must match the class name registered with @policy_registry.register)
    policy_type: str = "MyCustomPolicy"
    policy_suffix: str | None = None

    def _model_suffix(self) -> str:
        if not self.policy_suffix:
            return ""
        suffix = str(self.policy_suffix).strip()
        if not suffix:
            return ""
        return suffix if suffix.startswith("_") else f"_{suffix}"
    
    # Model file path - use @property instead of defining it directly
    @property
    def policy_file(self) -> str:
        """Model file path"""
        return f"assets/models/g1/my_custom/policy{self._model_suffix()}.pt"

    @property
    def vecnorm_file(self) -> str:
        """VecNorm parameters file path"""
        return f"assets/models/g1/my_custom/vecnorm_params{self._model_suffix()}.pt"
    
    # ========== Frequency settings ==========

    # Policy run frequency (Hz)
    # Must match the training frequency! Common values: 25, 50, 100
    freq: int = 50

    # ========== DoF configuration ==========

    # DoF configuration for the observation space - use the 29 DoF full-body configuration
    obs_dof: DoFConfig = G1MyCustomFullBodyDoF()

    # DoF configuration for the action space (usually the same as obs_dof)
    action_dof: DoFConfig = G1MyCustomFullBodyDoF()
    
    action_scales: list[float] = [
        0.5,   # left_hip_pitch_joint
        0.5,   # right_hip_pitch_joint
        0.25,  # waist_yaw_joint
        0.25,  # left_hip_roll_joint
        0.25,  # right_hip_roll_joint
        0.25,  # waist_roll_joint
        0.25,  # left_hip_yaw_joint
        0.25,  # right_hip_yaw_joint
        0.25,  # waist_pitch_joint
        0.5,   # left_knee_joint
        0.5,   # right_knee_joint
        1.0,   # left_shoulder_pitch_joint
        1.0,   # right_shoulder_pitch_joint
        0.5,   # left_ankle_pitch_joint
        0.5,   # right_ankle_pitch_joint
        1.0,   # left_shoulder_roll_joint
        1.0,   # right_shoulder_roll_joint
        0.5,   # left_ankle_roll_joint
        0.5,   # right_ankle_roll_joint
        1.0,   # left_shoulder_yaw_joint
        1.0,   # right_shoulder_yaw_joint
        1.0,   # left_elbow_joint
        1.0,   # right_elbow_joint
        1.0,   # left_wrist_roll_joint
        1.0,   # right_wrist_roll_joint
        1.0,   # left_wrist_pitch_joint
        1.0,   # right_wrist_pitch_joint
        1.0,   # left_wrist_yaw_joint
        1.0,   # right_wrist_yaw_joint
    ]
    
    # Action clipping range
    # None means no clipping, or set to a specific value (e.g. 100.0)
    action_clip: float | None = 100.0

    # Action smoothing coefficient (0-1)
    # Higher value = less smoothing (faster response but may jitter)
    # Lower value = more smoothing (slower response but more stable)
    # Formula: output = (1-beta) * last_action + beta * new_action
    action_beta: float = 0.9  # Matches the alpha parameter used during training

    # Observation history layout.
    # Older 257-dim policies use [0, 1, 2, 3, 4].
    # Newer 286-dim policies from checkpoint_final.pt use [0, 1, 2, 3, 4, 8].
    # Leave as None to let MyCustomPolicy infer from policy/vecnorm dimensions.
    joint_hist_steps: list[int] | None = None
    prev_action_steps: int = 3

    # ========== Communication Delay (equivalent Isaac substep delay) ==========
    # Whether to enable communication delay simulation
    use_communication_delay: bool = True
    # Maximum delay (unit: physics step, per the training configuration)
    max_delay: int = 4
    # Number of physics substeps per control cycle (Isaac commonly uses 4: 0.02 / 0.005)
    # RoboJuDo currently cannot access substep directly; this config is used to construct an equivalent fractional delay.
    comm_delay_decimation: int = 4

    # ========== Alpha Jitter (randomized EMA coefficient) ==========
    # alpha_jit_scale: random perturbation magnitude of the EMA coefficient
    # Set to None to disable jitter, set to a positive number (e.g. 0.025) to enable
    # Jitter is used during training to improve robustness; it can be disabled at deployment for deterministic behavior
    alpha_jit_scale: float | None = 0.025  # Matches the training configuration

    # alpha_wide_range: allowed range of the EMA coefficient (clamped to this range after adding jitter)
    # Format: [min_alpha, max_alpha]
    # Recommended to set as [alpha - 3*jit_scale, alpha + 3*jit_scale]
    alpha_wide_range: tuple[float, float] = (0.825, 0.975)  # Allows ±0.075 fluctuation

    disable_autoload: bool = True
