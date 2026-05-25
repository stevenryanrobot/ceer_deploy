"""
自定义 Policy 配置模板

使用步骤：
1. 复制此文件并重命名，例如 g1_my_rl_policy_cfg.py
2. 修改类名和参数
3. 在 g1_custom_cfg.py 中导入并使用
"""

from robojudo.config import Config
from robojudo.policy.policy_cfgs import PolicyCfg
from robojudo.tools.tool_cfgs import DoFConfig


# 如果你使用 29 个关节（包含躯干和手臂）
class G1MyCustomFullBodyDoF(DoFConfig):
    """29 DoF 全身配置"""
    
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
# Policy 配置
# ============================================================================

class G1MyCustomPolicyCfg(PolicyCfg):
    """
    自定义 Policy 的配置类
    
    所有参数都应该与训练时保持一致
    """
    
    # ========== 基本配置 ==========
    
    # 机器人类型
    robot: str = "g1"
    
    # Policy 类型（必须与 @policy_registry.register 的类名一致）
    policy_type: str = "MyCustomPolicy"
    policy_suffix: str | None = None

    def _model_suffix(self) -> str:
        if not self.policy_suffix:
            return ""
        suffix = str(self.policy_suffix).strip()
        if not suffix:
            return ""
        return suffix if suffix.startswith("_") else f"_{suffix}"
    
    # 模型文件路径 - 使用 @property 而不是直接定义
    @property
    def policy_file(self) -> str:
        """模型文件路径"""
        return f"assets/models/g1/my_custom/policy{self._model_suffix()}.pt"

    @property
    def vecnorm_file(self) -> str:
        """VecNorm 参数文件路径"""
        return f"assets/models/g1/my_custom/vecnorm_params{self._model_suffix()}.pt"
    
    # ========== 频率设置 ==========
    
    # Policy 运行频率（Hz）
    # 必须与训练时一致！常见值：25, 50, 100
    freq: int = 50
    
    # ========== DoF 配置 ==========
    
    # 观测空间的 DoF 配置 - 使用 29 DoF 全身配置
    obs_dof: DoFConfig = G1MyCustomFullBodyDoF()
    
    # 动作空间的 DoF 配置（通常与 obs_dof 相同）
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
    
    # 动作裁剪范围
    # None 表示不裁剪，或者设置为具体值（如 100.0）
    action_clip: float | None = 100.0
    
    # 动作平滑系数（0-1）
    # 更高的值 = 更少平滑（响应更快但可能抖动）
    # 更低的值 = 更多平滑（响应慢但更稳定）
    # 公式：output = (1-beta) * last_action + beta * new_action
    action_beta: float = 0.9  # 匹配训练时的 alpha 参数

    # Observation history layout.
    # Older 257-dim policies use [0, 1, 2, 3, 4].
    # Newer 286-dim policies from checkpoint_final.pt use [0, 1, 2, 3, 4, 8].
    # Leave as None to let MyCustomPolicy infer from policy/vecnorm dimensions.
    joint_hist_steps: list[int] | None = None
    prev_action_steps: int = 3

    # ========== Communication Delay (等效 Isaac 子步延迟) ==========
    # 是否启用通信延迟模拟
    use_communication_delay: bool = True
    # 最大延迟（单位：physics step，按训练配置）
    max_delay: int = 4
    # 控制周期内的 physics 子步数（Isaac 常见为 4：0.02 / 0.005）
    # 当前 RoboJuDo 无法直接访问 substep，本配置用于构造等效分数延迟。
    comm_delay_decimation: int = 4

    # ========== Alpha Jitter (随机化 EMA 系数) ==========
    # alpha_jit_scale: EMA 系数随机扰动幅度
    # 设置为 None 禁用 jitter，设置为正数（如 0.025）启用
    # 训练时使用 jitter 增强鲁棒性，部署时可以禁用以获得确定性行为
    alpha_jit_scale: float | None = 0.025  # 匹配训练配置

    # alpha_wide_range: EMA 系数允许的范围（加上 jitter 后会 clamp 到此范围）
    # 格式：[min_alpha, max_alpha]
    # 推荐设置为 [alpha - 3*jit_scale, alpha + 3*jit_scale]
    alpha_wide_range: tuple[float, float] = (0.825, 0.975)  # 允许 ±0.075 的波动

    disable_autoload: bool = True
