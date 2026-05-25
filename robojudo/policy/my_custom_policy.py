"""
自定义 Policy 模板
复制此文件并修改以创建你自己的 policy

使用步骤：
1. 将此文件复制为你的 policy 名称，例如 my_rl_policy.py
2. 修改类名和 @policy_registry.register 装饰器
3. 实现 get_observation 方法以匹配你的训练观测空间
4. 创建对应的 config 文件
5. 在 __init__.py 中导入
"""

import json
import numpy as np
import torch
import os

from robojudo.policy import Policy, policy_registry
from robojudo.policy.policy_cfgs import PolicyCfg
from robojudo.utils.util_func import get_gravity_orientation, my_quat_rotate_np
from robojudo.tools.dof import DoFAdapter

from collections import deque

import socket
import struct
import threading
import time
from typing import Optional, Tuple
import math
import re

# UDP packet format constants
# Default expected magic; can be overridden via MOTION_TRACKING_UDP_MAGIC (int, accepts 0x...)
MAGIC = b"G6D1"
PACK_FMT = "<4sI" + "f" * 28  # magic + seq + 28 floats
PACK_SIZE = struct.calcsize(PACK_FMT)

# -------------------------
# Quaternion helpers (torch)
# -------------------------
def _quat_normalize_wxyz(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # q: [...,4] (w,x,y,z)
    return q / (torch.linalg.norm(q, dim=-1, keepdim=True) + eps)

def _quat_conj_wxyz(q: torch.Tensor) -> torch.Tensor:
    # q: [...,4] (w,x,y,z)
    return torch.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], dim=-1)

def _quat_mul_wxyz(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # a,b: [...,4] (w,x,y,z)
    aw, ax, ay, az = a.unbind(dim=-1)
    bw, bx, by, bz = b.unbind(dim=-1)
    return torch.stack([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ], dim=-1)

def _quat_rotate_wxyz(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Rotate vector v by quaternion q (both torch).
    q: [...,4] (w,x,y,z), v: [...,3]
    return: [...,3]
    """
    q = _quat_normalize_wxyz(q)
    vq = torch.cat([torch.zeros_like(v[..., :1]), v], dim=-1)  # (0,v)
    return _quat_mul_wxyz(_quat_mul_wxyz(q, vq), _quat_conj_wxyz(q))[..., 1:4]

def _wrap_to_pi_torch(x: torch.Tensor) -> torch.Tensor:
    return (x + math.pi) % (2 * math.pi) - math.pi

def _yaw_from_quat_wxyz(q_wxyz: torch.Tensor) -> torch.Tensor:
    # q_wxyz: [...,4] (w,x,y,z)
    w, x, y, z = q_wxyz.unbind(dim=-1)
    t0 = 2.0 * (w * z + x * y)
    t1 = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(t0, t1)

def yaw_quat(q_wxyz: torch.Tensor) -> torch.Tensor:
    """
    Extract yaw-only quaternion from a full quaternion.
    Input: [..., 4] (w, x, y, z)
    Output: [..., 4] yaw-only quaternion (w, 0, 0, z_new)
    """
    yaw = _yaw_from_quat_wxyz(q_wxyz)
    # Convert yaw angle back to quaternion
    half_yaw = yaw / 2.0
    cos_half = torch.cos(half_yaw)
    sin_half = torch.sin(half_yaw)
    # Yaw quaternion: (cos(yaw/2), 0, 0, sin(yaw/2))
    return torch.stack([
        cos_half,
        torch.zeros_like(cos_half),
        torch.zeros_like(cos_half),
        sin_half
    ], dim=-1)
 
def axis_angle_from_quat(quat: torch.Tensor) -> torch.Tensor:
    quat = quat * (1.0 - 2.0 * (quat[..., 0:1] < 0.0))
    mag = torch.linalg.norm(quat[..., 1:], dim=-1)
    half_angle = torch.atan2(mag, quat[..., 0])
    angle = 2.0 * half_angle
    sin_half_angles_over_angles = torch.where(
        angle.abs() > 1.0e-6, torch.sin(half_angle) / angle, 0.5 - angle * angle / 48
    )
    return quat[..., 1:4] / sin_half_angles_over_angles.unsqueeze(-1)

def quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    xyz = quat[..., 1:]    # (..., 3)
    w = quat[..., :1]      # (..., 1)
    t = torch.cross(xyz, vec, dim=-1) * 2   # (..., 3)
    return vec + w * t + torch.cross(xyz, t, dim=-1)  # (..., 3)

def quat_apply_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    xyz = quat[..., 1:]    # (..., 3)
    w = quat[..., :1]      # (..., 1)
    t = torch.cross(xyz, vec, dim=-1) * 2
    return vec - w * t + torch.cross(xyz, t, dim=-1)



class UdpTeleopReceiver:
    """
    Receive UDP packets: root/head/left/right, each (pos3 + quat4) in WORLD frame.
    Thread updates latest sample (CPU tensors).
    """
    def __init__(self, bind_ip="0.0.0.0", bind_port=15000, timeout=0.2):
        self.bind_ip = bind_ip
        self.bind_port = bind_port
        self.timeout = timeout

        # allow overriding expected magic via env (e.g., MOTION_TRACKING_UDP_MAGIC=0x31443647)
        env_magic = os.getenv("MOTION_TRACKING_UDP_MAGIC", None)
        try:
            self.magic = int(env_magic, 0) if env_magic else MAGIC
        except Exception:
            print(f"[UDP] Invalid MOTION_TRACKING_UDP_MAGIC='{env_magic}', fallback to default {hex(MAGIC)}")
            self.magic = MAGIC
        if env_magic:
            print(f"[UDP] Using magic override: {hex(self.magic)} (env)")
        else:
            self.magic = MAGIC

        # stats for debugging
        self.stats = {"ok": 0, "bad_len": 0, "bad_magic": 0}
        self._last_sender = None

        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._seq: int = -1
        self._t_recv: float = 0.0

        # store latest as torch CPU tensors
        self._root_pos = torch.zeros(3, dtype=torch.float32)
        self._root_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32)
        self._head_pos = torch.zeros(3, dtype=torch.float32)
        self._head_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32)
        self._l_pos = torch.zeros(3, dtype=torch.float32)
        self._l_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32)
        self._r_pos = torch.zeros(3, dtype=torch.float32)
        self._r_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32)

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def _loop(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((self.bind_ip, self.bind_port))
        except OSError as e:
            print(f"[UDP] Bind failed on {self.bind_ip}:{self.bind_port} -> {e}")
            return
        sock.settimeout(self.timeout)

        print(f"[UDP] Listening on {self.bind_ip}:{self.bind_port} (expect len={PACK_SIZE})", flush=True)

        bad_len = bad_magic = ok = 0

        while self._running:
            try:
                data, addr = sock.recvfrom(2048)
                if len(data) != PACK_SIZE:
                    bad_len += 1
                    if bad_len <= 5 or bad_len % 50 == 0:
                        print(f"[UDP] Dropped packet len={len(data)} from {addr}; expected {PACK_SIZE} (count={bad_len})", flush=True)
                    continue
                magic, seq, *floats = struct.unpack(PACK_FMT, data)
                if magic != self.magic:
                    bad_magic += 1
                    if bad_magic <= 5 or bad_magic % 50 == 0:
                        first8 = data[:8].hex()
                        print(
                            f"[UDP] Dropped packet with wrong magic {hex(magic)} (expected {hex(self.magic)}) "
                            f"from {addr} first8={first8} (count={bad_magic})",
                            flush=True,
                        )
                    continue

                # 4 bodies * 7 floats
                vals = torch.tensor(floats, dtype=torch.float32)  # shape (28,)
                root = vals[0:7]
                head = vals[7:14]
                left = vals[14:21]
                right = vals[21:28]

                with self._lock:
                    self._seq = int(seq)
                    self._t_recv = time.time()
                    self._last_sender = addr
                    ok += 1
                    self.stats = {"ok": ok, "bad_len": bad_len, "bad_magic": bad_magic}

                    if ok <= 3 or ok % 100 == 0:
                        print(f"[UDP] Received seq={seq} len={len(data)} from {addr} (ok={ok})", flush=True)

                    self._root_pos = root[0:3].clone()
                    self._root_quat = root[3:7].clone()
                    self._head_pos = head[0:3].clone()
                    self._head_quat = head[3:7].clone()
                    self._l_pos = left[0:3].clone()
                    self._l_quat = left[3:7].clone()
                    self._r_pos = right[0:3].clone()
                    self._r_quat = right[3:7].clone()

            except socket.timeout:
                continue
            except Exception:
                continue

    def get_latest(self) -> Tuple[int, float, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          seq, t_recv, root_pos(3), root_quat(4), head_pos(3), head_quat(4), l_pos(3), l_quat(4), r_pos(3), r_quat(4)
        """
        with self._lock:
            return (
                self._seq, self._t_recv,
                self._root_pos.clone(), self._root_quat.clone(),
                self._head_pos.clone(), self._head_quat.clone(),
                self._l_pos.clone(), self._l_quat.clone(),
                self._r_pos.clone(), self._r_quat.clone(),
            )

def apply_perm_signs(x: np.ndarray, perm=None, signs=None):
    y = x
    if perm is not None:
        perm = np.asarray(perm, dtype=np.int64)
        y = y[perm]
    if signs is not None:
        signs = np.asarray(signs, dtype=np.float32)
        y = y * signs
    return y

def _match_indices(motion_names, asset_names, patterns, name_map=None, device=None, debug=False):
    asset_idx, motion_idx = [], []
    for i, a in enumerate(asset_names):
        if any(re.match(p, a) for p in patterns):
            m = name_map.get(a, a) if name_map else a
            if m in motion_names:
                asset_idx.append(i)
                motion_idx.append(motion_names.index(m))
                if debug:
                    print(f"Matched asset '{a}' (idx {i}) to motion '{m}' (idx {motion_names.index(m)})")
    return torch.tensor(motion_idx, device=device), torch.tensor(asset_idx, device=device)


@policy_registry.register
class MyCustomPolicy(Policy):
    """
    Sim2Sim obs adaptor integrated.
    Output obs matches training layout:
      boot_indicator_state (1)
      command (cmd_dim)
      root_and_wrist_6d (wrist_dim)
      root_ang_vel_history[0] (3)
      projected_gravity_history[0] (3)
      joint_pos_history[0,1,2,3,4] (len(steps)*num_dof)
      prev_actions[0,1,2] (3*action_dim)
    """
    cfg_policy: PolicyCfg

    def _infer_model_obs_dim(self) -> int | None:
        try:
            state = self.model.state_dict()
        except Exception:
            return None

        for key in ("adapt.0.weight", "module.adapt.0.weight"):
            weight = state.get(key)
            if isinstance(weight, torch.Tensor) and weight.ndim == 2:
                return int(weight.shape[1])

        for key, weight in state.items():
            if key.endswith("adapt.0.weight") and isinstance(weight, torch.Tensor) and weight.ndim == 2:
                return int(weight.shape[1])
        return None

    def _resolve_joint_hist_steps(self) -> list[int]:
        cfg_steps = getattr(self.cfg_policy, "joint_hist_steps", None)
        if cfg_steps is not None:
            return [int(x) for x in cfg_steps]

        default_steps = [0, 1, 2, 3, 4]
        target_obs_dim = self.model_obs_dim
        if target_obs_dim is None and self._vecnorm_loc is not None:
            target_obs_dim = int(self._vecnorm_loc.numel())

        if target_obs_dim is None:
            return default_steps

        fixed_dim = 1 + self.cmd_dim + self.wrist_dim + 3 + 3
        prev_action_dim = self.prev_action_steps * self.action_dim
        history_dim = target_obs_dim - fixed_dim - prev_action_dim
        if history_dim <= 0 or history_dim % self.num_dof != 0:
            return default_steps

        history_count = history_dim // self.num_dof
        if history_count == len(default_steps):
            return default_steps

        # Newer checkpoints in this folder use history_steps=[0,1,2,3,4,8].
        if history_count == 6:
            inferred_steps = [0, 1, 2, 3, 4, 8]
        else:
            inferred_steps = list(range(history_count))

        print(
            f"[MyCustomPolicy] Inferred joint_hist_steps={inferred_steps} "
            f"from target obs dim {target_obs_dim}"
        )
        return inferred_steps

    def _check_obs_dim_consistency(self) -> None:
        dims = {"runtime_obs": self.expected_obs_dim}
        if self.model_obs_dim is not None:
            dims["model_obs"] = self.model_obs_dim
        if self._vecnorm_loc is not None:
            dims["vecnorm_loc"] = int(self._vecnorm_loc.numel())
        if self._vecnorm_scale is not None:
            dims["vecnorm_scale"] = int(self._vecnorm_scale.numel())

        unique_dims = set(dims.values())
        if len(unique_dims) == 1:
            print(f"[MyCustomPolicy] Obs dim check OK: {dims}")
            return

        print(f"[MyCustomPolicy] ERROR: obs dim mismatch: {dims}")
        print(
            "[MyCustomPolicy] Set cfg_policy.joint_hist_steps / prev_action_steps "
            "to match the training checkpoint."
        )

    def __init__(self, cfg_policy: PolicyCfg, device: str = "cpu"):
        super().__init__(cfg_policy=cfg_policy, device=device)

        # 手动加载模型（支持非 TorchScript 格式）
        if not self.cfg_policy.disable_autoload:
            # 如果没有禁用自动加载，但模型已经被基类加载了
            pass
        else:
            # 手动加载模型
            policy_file = self.cfg_policy.policy_file
            print(f"[MyCustomPolicy] Loading model from {policy_file}...")
            try:
                # 尝试 TorchScript 格式
                self.model = torch.jit.load(policy_file, map_location=self.device)
                self.model.eval()
                print(f"[MyCustomPolicy] Loaded as TorchScript model")
            except Exception as e:
                # 如果失败，尝试普通 checkpoint
                print(f"[MyCustomPolicy] TorchScript load failed, trying checkpoint: {e}")
                
                # 尝试加载，如果遇到缺失的模块，给出提示
                try:
                    checkpoint = torch.load(policy_file, map_location=self.device, weights_only=False)
                except ModuleNotFoundError as module_err:
                    print(f"[MyCustomPolicy] ERROR: Missing module - {module_err}")
                    print(f"[MyCustomPolicy] Your checkpoint requires additional dependencies.")
                    print(f"[MyCustomPolicy] Please install the required package or convert your model to TorchScript format.")
                    print(f"[MyCustomPolicy] To convert to TorchScript: torch.jit.script(your_model) or torch.jit.trace()")
                    raise
                
                print(f"[MyCustomPolicy] Checkpoint type: {type(checkpoint)}")
                
                if isinstance(checkpoint, dict):
                    print(f"[MyCustomPolicy] Checkpoint keys: {checkpoint.keys()}")
                    # 尝试常见的 key 名称
                    if 'model' in checkpoint:
                        self.model = checkpoint['model']
                        print(f"[MyCustomPolicy] Loaded model from checkpoint['model']")
                    elif 'model_state_dict' in checkpoint:
                        # 需要先定义模型架构，然后加载 state_dict
                        print(f"[MyCustomPolicy] Found model_state_dict, but need model architecture to load")
                        # self.model = YourModelClass(...)
                        # self.model.load_state_dict(checkpoint['model_state_dict'])
                        raise NotImplementedError("Need to define model architecture first")
                    elif 'state_dict' in checkpoint:
                        print(f"[MyCustomPolicy] Found state_dict, but need model architecture to load")
                        raise NotImplementedError("Need to define model architecture first")
                    else:
                        print(f"[MyCustomPolicy] Unknown checkpoint format, using as-is")
                        self.model = checkpoint
                else:
                    # checkpoint 本身就是模型
                    print(f"[MyCustomPolicy] Checkpoint is model directly")
                    self.model = checkpoint
                
                print(f"[MyCustomPolicy] Model loaded successfully: {type(self.model)}")
        
        print(f"[MyCustomPolicy] Final model type: {type(self.model)}")
        print(f"[MyCustomPolicy] Model attributes: {dir(self.model)}")
        self.model_obs_dim = self._infer_model_obs_dim()
        if self.model_obs_dim is not None:
            print(f"[MyCustomPolicy] Inferred model obs dim: {self.model_obs_dim}")
        else:
            print("[MyCustomPolicy] WARNING: could not infer model obs dim from state_dict")

        self.custom_param = getattr(self.cfg_policy, "custom_param", 1.0)

        # ---- load transforms ----
        # tf_path = "/home/dexlab/RoboJuDo/assets/models/g1/my_custom/transforms.json"
        # with open(tf_path, "r") as f:
        #     tfs = json.load(f)

        # self.obs_perm  = tfs["obs_transform"].get("perm", None)
        # self.obs_signs = tfs["obs_transform"].get("signs", None)
        # self.act_perm  = tfs["act_transform"].get("perm", None)
        # self.act_signs = tfs["act_transform"].get("signs", None)

        # ---- load vecnorm stats from .pt file ----
        # Format: dict with params[key]['loc'] and params[key]['scale'] as tensors
        self._vecnorm_loc = None
        self._vecnorm_scale = None
        try:
            vecnorm_path = getattr(
                self.cfg_policy,
                "vecnorm_file",
                os.path.join(os.path.dirname(self.cfg_policy.policy_file), "vecnorm_params.pt"),
            )
            if os.path.isfile(vecnorm_path):
                vecnorm_params = torch.load(vecnorm_path, map_location="cpu", weights_only=False)
                vecnorm_key = "policy"
                
                print(f"[vecnorm] loaded from: {vecnorm_path}")
                print(f"[vecnorm] top-level keys: {list(vecnorm_params.keys())}")
                
                if vecnorm_key not in vecnorm_params:
                    print(f"[vecnorm] ERROR: key '{vecnorm_key}' not found in vecnorm params")
                else:
                    self._vecnorm_loc = vecnorm_params[vecnorm_key]['loc']  # torch.Tensor
                    self._vecnorm_scale = vecnorm_params[vecnorm_key]['scale']  # torch.Tensor
                    print(f"[vecnorm] loc shape: {self._vecnorm_loc.shape}")
                    print(f"[vecnorm] scale shape: {self._vecnorm_scale.shape}")
            else:
                print(f"[vecnorm] file not found: {vecnorm_path}")
        except Exception as e:
            print(f"[vecnorm] Failed to load vecnorm stats: {e}")

        # Runtime print frequency (env override: MY_POLICY_PRINT_EVERY, default 50 steps)
        try:
            _pe = int(os.getenv("MY_POLICY_PRINT_EVERY", "50"))
            self._print_every = max(1, _pe)
        except Exception:
            self._print_every = 50

        # -------- Per-joint action scaling (optional) --------
        # Load per-joint action_scales if provided, otherwise fall back to single action_scale
        self.action_scales = getattr(self.cfg_policy, "action_scales", None)
        if self.action_scales is not None:
            self.action_scales = np.asarray(self.action_scales, dtype=np.float32)
            if len(self.action_scales) != self.num_dofs:
                print(
                    f"[MyCustomPolicy] WARNING: action_scales length {len(self.action_scales)} "
                    f"!= num_dofs {self.num_dofs}. Falling back to scalar action_scale."
                )
                self.action_scales = None
            else:
                print(f"[MyCustomPolicy] Using per-joint action_scales: {self.action_scales}")
        else:
            print(f"[MyCustomPolicy] Using scalar action_scale: {self.action_scale}")

        # -------- dims you MUST set correctly (match training) --------
        self.cmd_dim = getattr(self.cfg_policy, "cmd_dim", 6)        # 6D command: root_height + linvel + heading + force_limit
        self.wrist_dim = getattr(self.cfg_policy, "wrist_dim", 12)   # root_and_wrist_6d dim
        self.action_dim = int(self.num_actions)                      # action dim from base Policy
        self.num_dof = int(self.num_dofs)                            # dof count from env/policy base
        self.num_envs = int(getattr(self, "num_envs", 1))            # multi-env support
        
        # Debug (can be toggled as needed)
        # print(f"[MyCustomPolicy] Dimension settings:")
        # print(f"  num_dof: {self.num_dof} (should be 29 for full body)")
        # print(f"  action_dim: {self.action_dim} (should be 29)")
        # print(f"  num_envs: {self.num_envs}")
        # print(f"  cmd_dim: {self.cmd_dim}")
        # print(f"  wrist_dim: {self.wrist_dim}")

        # -------- history config (match training) --------
        # 注意：这里必须与训练配置完全一致！
        self.prev_action_steps = int(getattr(self.cfg_policy, "prev_action_steps", 3))
        self.joint_hist_steps = self._resolve_joint_hist_steps()
        self.max_joint_hist = max(self.joint_hist_steps)
        
        # 计算预期的观测维度
        expected_joint_hist_dim = len(self.joint_hist_steps) * self.num_dof
        expected_prev_action_dim = self.prev_action_steps * self.action_dim
        expected_total_dim = (
            1 +  # boot_indicator
            self.cmd_dim +  # command
            self.wrist_dim +  # root_and_wrist_6d
            3 +  # base_ang_vel
            3 +  # projected_gravity
            expected_joint_hist_dim +  # joint_pos_history
            expected_prev_action_dim  # prev_actions
        )
        self.expected_obs_dim = expected_total_dim
        self._check_obs_dim_consistency()
        
        # boot indicator (optional, match training behavior if any)
        self.boot_max = float(getattr(self.cfg_policy, "boot_max", 25))
        self.boot_indicator = self.boot_max

        # buffers (torch, on device)
        self._joint_pos_hist = None   # [H, N, D]
        self._prev_actions = None     # [A, N, action_dim]

        # single UDP receiver (avoid double bind)
        self._teleop = UdpTeleopReceiver(bind_port=15000) 
        self._teleop.start()

        # Optional UDP broadcaster for root state (disabled by default).
        # Enable by setting environment variable MOTION_TRACKING_UDP_BROADCAST to "host:port" (e.g. 127.0.0.1:15001)
        # bodies for teleoperation (head and wrists / hands) used for 6D teleop input
        # self.teleop_body_patterns = ["head_mimic", ".*_hand_mimic", ".*wrist_roll_link.*"]
        # self.teleop_idx_motion, self.teleop_idx_asset = _match_indices(
        #     self.dataset.body_names,
        #     self.asset.body_names,
        #     self.teleop_body_patterns,
        #     name_map=self.keypoint_map,
        #     device=self.device,
        #     debug=False,
        # )
        
        self._udp_broadcast_enabled = False
        self._udp_broadcast_addr = ("127.0.0.1", 15001)
        self._udp_broadcast_sock = None
        
        try:
            b_cfg = os.getenv("MOTION_TRACKING_UDP_BROADCAST", "127.0.0.1:15001")
            print(f"[MOTION_TRACKING] UDP broadcaster config: {b_cfg}", flush=True)
            if b_cfg and b_cfg.strip().lower() not in {"0", "false", "off", "none", "disabled"}:
                parts = b_cfg.split(":")
                host = parts[0]
                port = int(parts[1]) if len(parts) > 1 else 15001
                self._udp_broadcast_addr = (host, port)
                self._udp_broadcast_enabled = True
                # create non-blocking UDP socket
                try:
                    self._udp_broadcast_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    self._udp_broadcast_sock.setblocking(False)
                    self._udp_broadcast_seq = 0
                    print(f"[MOTION_TRACKING] UDP broadcaster enabled -> {self._udp_broadcast_addr}", flush=True)
                    # Print debug send every N sends (option B). Configure via env var:
                    # MOTION_TRACKING_UDP_BROADCAST_PRINT_EVERY (integer > 0). Default 10.
                    try:
                        _pe = os.getenv("MOTION_TRACKING_UDP_BROADCAST_PRINT_EVERY", "10")
                        self._udp_broadcast_print_every = max(1, int(_pe))
                    except Exception:
                        self._udp_broadcast_print_every = 10
                    print(f"[MOTION_TRACKING] UDP broadcaster debug: printing every {self._udp_broadcast_print_every} sends", flush=True)
                except Exception:
                    self._udp_broadcast_sock = None
                    self._udp_broadcast_enabled = False
        except Exception:
            self._udp_broadcast_enabled = False

        # Create DoFAdapter for env->policy joint order conversion
        # Assumes env uses G1_29DoF order (from g1_env_cfg.py)
        env_joint_order = [
            "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
            "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
            "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
            "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
            "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
            "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
        ]
        policy_joint_order = self.cfg_obs_dof.joint_names
        self._dof_adapter = DoFAdapter(env_joint_order, policy_joint_order)
        print(f"[MyCustomPolicy] Created DoFAdapter: {len(env_joint_order)} env joints -> {len(policy_joint_order)} policy joints")

        # -------- Alpha Jitter (随机化 EMA 平滑系数) --------
        self.alpha_jit_scale = getattr(self.cfg_policy, "alpha_jit_scale", None)
        self.alpha_wide_range = getattr(self.cfg_policy, "alpha_wide_range", (0.0, 1.0))

        if self.alpha_jit_scale is not None:
            # 初始化 jitter tensor（每次 get_action 时会重新采样）
            self.alpha_jit = torch.zeros(1, device=self.device, dtype=torch.float32)
            print(f"[MyCustomPolicy] Alpha jitter enabled: scale={self.alpha_jit_scale}, range={self.alpha_wide_range}")
        else:
            self.alpha_jit = None
            print(f"[MyCustomPolicy] Alpha jitter disabled (deterministic EMA)")

        # -------- Communication delay emulation --------
        self.use_communication_delay = bool(getattr(self.cfg_policy, "use_communication_delay", False))
        self.max_delay = int(max(0, getattr(self.cfg_policy, "max_delay", 0)))
        self.comm_delay_decimation = int(max(1, getattr(self.cfg_policy, "comm_delay_decimation", 1)))
        self._delay_steps = 0
        self._action_buf = None
        self.applied_action = None

        if self.use_communication_delay and self.max_delay > 0:
            print(
                f"[MyCustomPolicy] Communication delay enabled: "
                f"max_delay={self.max_delay} physics-steps, decimation={self.comm_delay_decimation}"
            )
        else:
            print("[MyCustomPolicy] Communication delay disabled")

        self.reset()

    def reset(self):
        self.timestep = 0
        self.boot_indicator = self.boot_max
        self._action_count = 0

        # history buffers
        H = self.max_joint_hist + 1
        N = self.num_envs
        D = self.num_dof
        A = self.prev_action_steps

        self._last_udp_seq = -1
        self._last_udp_t = None
        self._last_root_pos_w = None
        self._last_root_yaw = None

        self._joint_pos_hist = torch.zeros((H, N, D), device=self.device, dtype=torch.float32)
        self._prev_actions = torch.zeros((A, N, self.action_dim), device=self.device, dtype=torch.float32)
        self._history_initialized = False

        # 注意：base class 里可能把 last_action 当 numpy；这里统一用 torch 存，再转 numpy 输出
        self.last_action = torch.zeros((N, self.action_dim), device=self.device, dtype=torch.float32)
        
        # Store raw network output (clamped) for prev_actions history (to match training)
        self.last_raw_action = torch.zeros((N, self.action_dim), device=self.device, dtype=torch.float32)

        # delay state
        if self.use_communication_delay and self.max_delay > 0:
            self._delay_steps = int(torch.randint(0, self.max_delay + 1, (1,), device="cpu").item())
            buf_len = int(math.ceil(self.max_delay / self.comm_delay_decimation)) + 2
            self._action_buf = torch.zeros((buf_len, self.action_dim), device=self.device, dtype=torch.float32)
            self.applied_action = torch.zeros((self.action_dim,), device=self.device, dtype=torch.float32)
            print(f"[MyCustomPolicy] Sampled communication delay: {self._delay_steps} physics-steps")
        else:
            self._delay_steps = 0
            self._action_buf = None
            self.applied_action = torch.zeros((self.action_dim,), device=self.device, dtype=torch.float32)

    def post_step_callback(self, commands=None):
        self.timestep += 1
        # 你也可以在这里做 logging

    def debug_viz(self, visualizer, env_data, ctrl_data, extras):
        if not extras.get("udp_valid", False):
            return

        root_quat_xyzw = extras.get("viz_root_target_yaw_xyzw")
        left_ee_pos = extras.get("viz_left_ee_target_pos_b")
        right_ee_pos = extras.get("viz_right_ee_target_pos_b")
        root_speed_xy = extras.get("viz_root_target_speed_xy")

        if root_quat_xyzw is None:
            return

        viewer_data = getattr(getattr(visualizer, "viewer", None), "data", None)
        if viewer_data is not None:
            qpos = np.asarray(viewer_data.qpos, dtype=np.float32)
            root_pos = qpos[:3].copy()
            base_quat_xyzw = qpos[3:7][[1, 2, 3, 0]].copy()
        elif env_data.base_pos is not None and env_data.base_quat is not None:
            root_pos = np.asarray(env_data.base_pos, dtype=np.float32)
            base_quat_xyzw = np.asarray(env_data.base_quat, dtype=np.float32)
        else:
            return

        root_quat_xyzw = np.asarray(root_quat_xyzw, dtype=np.float32)
        left_ee_pos = np.asarray(left_ee_pos, dtype=np.float32) if left_ee_pos is not None else None
        right_ee_pos = np.asarray(right_ee_pos, dtype=np.float32) if right_ee_pos is not None else None
        root_speed_xy = (
            np.asarray(root_speed_xy, dtype=np.float32)
            if root_speed_xy is not None
            else np.zeros(2, dtype=np.float32)
        )

        if not np.all(np.isfinite(root_pos)) or not np.all(np.isfinite(base_quat_xyzw)):
            return

        root_marker_pos = root_pos + np.array([0.0, 0.0, 0.12], dtype=np.float32)

        if left_ee_pos is not None and np.all(np.isfinite(left_ee_pos)):
            left_ee_pos = root_pos + my_quat_rotate_np(base_quat_xyzw, left_ee_pos)
        if right_ee_pos is not None and np.all(np.isfinite(right_ee_pos)):
            right_ee_pos = root_pos + my_quat_rotate_np(base_quat_xyzw, right_ee_pos)

        speed = float(np.linalg.norm(root_speed_xy))
        arrow_len = float(np.clip(max(speed, 0.25), 0.25, 0.6))

        visualizer.set_mocap_pose("viz_root_target", root_marker_pos)
        if left_ee_pos is not None and np.all(np.isfinite(left_ee_pos)):
            visualizer.set_mocap_pose("viz_left_ee_target", left_ee_pos)
        else:
            visualizer.hide_mocap("viz_left_ee_target")
        if right_ee_pos is not None and np.all(np.isfinite(right_ee_pos)):
            visualizer.set_mocap_pose("viz_right_ee_target", right_ee_pos)
        else:
            visualizer.hide_mocap("viz_right_ee_target")

        visualizer.set_mocap_pose("viz_root_dir", root_marker_pos, quat_xyzw=root_quat_xyzw)
        visualizer.set_arrow_length("viz_root_dir_shaft", "viz_root_dir_tip", arrow_len)

    # -------------------------
    # UDP control hook
    # -------------------------
    def _get_udp_control(self):
        """
        Returns:
        root_command:   [N,7]  = [root_pos_w(3), target_yaw_quat_wxyz(4)]
        udp_command:    [N,3]  = [vx, vy, vyaw]  (estimated from UDP root pose delta)
        ee_cmd_12:      [N,12] = EE-only command for policy's root_and_wrist_6d (name legacy)
        extra: dict
        """

        N = self.num_envs
        device = self.device

        root_command = torch.zeros((N, 7), device=device, dtype=torch.float32)
        udp_command  = torch.zeros((N, 3), device=device, dtype=torch.float32)
        ee_cmd_12    = torch.zeros((N, 12), device=device, dtype=torch.float32)
        extra = {"udp_valid": False}

        if not hasattr(self, "_teleop") or self._teleop is None:
            print("[MyCustomPolicy] UDP teleop receiver not initialized")
            return root_command, udp_command, ee_cmd_12, extra

        seq, t_recv, \
            root_pos_w, root_quat_xyzw, \
            head_pos_w, head_quat_xyzw, \
            l_pos_w, l_quat_xyzw, \
            r_pos_w, r_quat_xyzw = self._teleop.get_latest()
        
        seq, t_recv, \
            root_pos_unused, root_quat_unused, \
            head_pos_b, head_quat_b, \
            l_pos_b, l_quat_b, \
            r_pos_b, r_quat_b = self._teleop.get_latest()
        
        # ---- pack positions (already root frame) ----
        pos_sel_b = torch.stack([l_pos_b, r_pos_b], dim=0).to(self.device)   # [2, 3]
        pos_sel_b = pos_sel_b.unsqueeze(0).expand(self.num_envs, -1, -1)     # [N, 2, 3]

        # ---- pack orientations (already root-relative) ----
        quat_sel_b = torch.stack([l_quat_b, r_quat_b], dim=0).to(self.device)  # [2, 4]
        # optional safety normalize
        quat_sel_b = quat_sel_b / (torch.norm(quat_sel_b, dim=-1, keepdim=True) + 1e-8)
        quat_sel_b = quat_sel_b.unsqueeze(0).expand(self.num_envs, -1, -1)     # [N, 2, 4]

        axis_ang_b = axis_angle_from_quat(quat_sel_b)                          # [N, 2, 3]

        ee_cmd_12 = torch.cat(
            [pos_sel_b.reshape(self.num_envs, -1),   # [N, 6]
             axis_ang_b.reshape(self.num_envs, -1)], # [N, 6]
            dim=-1
        )  # [N, 12]

        extra["udp_seq"] = int(seq)
        extra["udp_t_recv"] = float(t_recv)
        extra["udp_stats"] = getattr(self._teleop, "stats", None)
        extra["udp_last_sender"] = getattr(self._teleop, "_last_sender", None)

        if seq < 0:
            print("[MyCustomPolicy] No valid UDP packet received yet")
            return root_command, udp_command, ee_cmd_12, extra

        # -------- root quat: xyzw -> wxyz --------
        root_quat_xyzw = root_quat_xyzw.to(device).float()  # [4]
        root_quat_wxyz = torch.stack(
            [root_quat_xyzw[3], root_quat_xyzw[0], root_quat_xyzw[1], root_quat_xyzw[2]],
            dim=-1
        )
        root_quat_wxyz = root_quat_wxyz / (torch.linalg.norm(root_quat_wxyz) + 1e-8)

        # -------- root_command: pos_w + yaw_quat(wxyz) --------
        target_yaw_quat = yaw_quat(root_quat_wxyz.unsqueeze(0)).squeeze(0)  # [4], wxyz
        root_pos_wN = root_pos_w.to(device).float().view(1, 3).repeat(N, 1)          # [N,3]
        target_yaw_quatN = target_yaw_quat.view(1, 4).repeat(N, 1)                   # [N,4]
        root_command = torch.cat([root_pos_wN, target_yaw_quatN], dim=-1)             # [N,7]
        target_yaw_quat_xyzw = torch.stack(
            [target_yaw_quat[1], target_yaw_quat[2], target_yaw_quat[3], target_yaw_quat[0]],
            dim=-1,
        )

        # -------- udp_command: [vx, vy, vyaw] from pose delta --------
        yaw_now = _yaw_from_quat_wxyz(root_quat_wxyz).view(1).repeat(N)  # [N]
        if (self._last_udp_t is not None) and (seq != self._last_udp_seq):
            dt = float(t_recv - self._last_udp_t)
            if 1e-4 < dt < 0.5:
                dp = root_pos_wN - self._last_root_pos_w  # [N,3]
                vx = dp[:, 0] / dt
                vy = dp[:, 1] / dt
                dyaw = _wrap_to_pi_torch(yaw_now - self._last_root_yaw)
                vyaw = dyaw / dt
                udp_command = torch.stack([vx, vy, vyaw], dim=-1)  # [N,3]


        # -------- update memory --------
        self._last_udp_seq = int(seq)
        self._last_udp_t = float(t_recv)
        self._last_root_pos_w = root_pos_wN.detach()
        self._last_root_yaw = yaw_now.detach()

        extra["udp_valid"] = True
        extra["viz_root_target_yaw_xyzw"] = target_yaw_quat_xyzw.detach().cpu().numpy()
        extra["viz_left_ee_target_pos_b"] = l_pos_b.detach().cpu().numpy()
        extra["viz_right_ee_target_pos_b"] = r_pos_b.detach().cpu().numpy()
        extra["viz_root_target_speed_xy"] = udp_command[0, :2].detach().cpu().numpy()

        # debug print
        # print(f'udp_command: {root_command[0].cpu().numpy()}')

        return root_command, udp_command, ee_cmd_12, extra

    def _broadcast_env_state(self, env_data):
        if not self._udp_broadcast_enabled or self._udp_broadcast_sock is None:
            return

        base_pos = getattr(env_data, "base_pos", None)
        base_quat = getattr(env_data, "base_quat", None)
        if base_pos is None or base_quat is None:
            return

        base_pos = np.asarray(base_pos, dtype=np.float32).reshape(-1)
        base_quat_xyzw = np.asarray(base_quat, dtype=np.float32).reshape(-1)
        if base_pos.shape[0] < 3 or base_quat_xyzw.shape[0] < 4:
            return
        if not np.all(np.isfinite(base_pos[:3])) or not np.all(np.isfinite(base_quat_xyzw[:4])):
            return

        qx, qy, qz, qw = base_quat_xyzw[:4]
        dynamic_objects = []
        for obj_name, obj_pos, obj_quat_xyzw in (getattr(env_data, "dynamic_objects", []) or []):
            obj_pos = np.asarray(obj_pos, dtype=np.float32).reshape(-1)
            obj_quat_xyzw = np.asarray(obj_quat_xyzw, dtype=np.float32).reshape(-1)
            if obj_pos.shape[0] < 3 or obj_quat_xyzw.shape[0] < 4:
                continue
            if not np.all(np.isfinite(obj_pos[:3])) or not np.all(np.isfinite(obj_quat_xyzw[:4])):
                continue
            dynamic_objects.append((obj_name, obj_pos, obj_quat_xyzw))

        fields = [
            f"{time.time():.6f}",
            f"{base_pos[0]:.6f}",
            f"{base_pos[1]:.6f}",
            f"{base_pos[2]:.6f}",
            f"{qw:.6f}",
            f"{qx:.6f}",
            f"{qy:.6f}",
            f"{qz:.6f}",
            str(len(dynamic_objects)),
        ]
        for obj_name, obj_pos, obj_quat_xyzw in dynamic_objects:
            fields.extend([
                str(obj_name),
                f"{obj_pos[0]:.6f}",
                f"{obj_pos[1]:.6f}",
                f"{obj_pos[2]:.6f}",
                f"{obj_quat_xyzw[0]:.6f}",
                f"{obj_quat_xyzw[1]:.6f}",
                f"{obj_quat_xyzw[2]:.6f}",
                f"{obj_quat_xyzw[3]:.6f}",
            ])
        payload = ",".join(fields).encode("ascii")

        try:
            self._udp_broadcast_sock.sendto(payload, self._udp_broadcast_addr)
            self._udp_broadcast_seq += 1
        except Exception as e:
            print(f"[MOTION_TRACKING] UDP broadcast send failed: {e}", flush=True)

    # -------------------------
    # Buffer update
    # -------------------------
    def _step_update_buffers(self, env_data):
        """
        Call this inside get_observation AFTER you computed current action/last_action update.
        Updates history buffers to match training configuration:
        - joint_pos_history: absolute joint positions (no offset subtraction)
        - prev_actions: clamped network output (before smoothing and scaling)
        """
        # joint pos push (newest at index 0)
        # Convert from env joint order to policy joint order
        dof_pos_env = np.asarray(env_data.dof_pos, dtype=np.float32)
        # dof_pos_policy = self._dof_adapter.fit(dof_pos_env)
        dof_pos_policy = dof_pos_env
        dof_pos = torch.as_tensor(dof_pos_policy, device=self.device, dtype=torch.float32)
        if dof_pos.ndim == 1:
            dof_pos = dof_pos.unsqueeze(0)  # [1,D]
        # expect [N,D]
        assert dof_pos.shape[0] == self.num_envs and dof_pos.shape[1] == self.num_dof, \
            f"dof_pos shape {dof_pos.shape} != ({self.num_envs},{self.num_dof})"

        # Use absolute joint positions (no offset subtraction)
        dof_pos_absolute = dof_pos
        # print(f"[MyCustomPolicy] dof_pos_absolute: {dof_pos_absolute}")

        # on first call, fill entire history with current joint pos to match training init
        if not getattr(self, "_history_initialized", False):
            for k in range(self.max_joint_hist + 1):
                self._joint_pos_hist[k] = dof_pos_absolute
            # prev_actions already zero-initialized as desired
            self._history_initialized = True

        self._joint_pos_hist = torch.roll(self._joint_pos_hist, shifts=1, dims=0)
        self._joint_pos_hist[0] = dof_pos_absolute

        # prev actions push (newest at index 0)
        # Use raw network output (clamped) to match training
        self._prev_actions = torch.roll(self._prev_actions, shifts=1, dims=0)
        self._prev_actions[0] = self.last_raw_action

        # boot indicator decay
        self.boot_indicator = max(0.0, self.boot_indicator - 1.0)

    # -------------------------
    # Main: get_observation
    # -------------------------
    def get_observation(self, env_data, ctrl_data) -> tuple[np.ndarray, dict]:
        """
        Build obs that matches training exactly.
        Returns:
          obs: np.ndarray, shape [N, obs_dim]
          info: dict
        """

        N = self.num_envs

        self._broadcast_env_state(env_data)
        root_cmd7, udp_cmd3, ee12, extra = self._get_udp_control()

        # ====== 构建 6 维 command ======
        # Output format (6 dimensions, matches training):
        #   - root_height: [N, 1] - current frame root height
        #   - target_linvel_b: [N, 2] - target xy linear velocity in body frame
        #   - target_heading_b: [N, 2] - target heading direction in body frame (cos, sin)
        #   - force_safe_limit: [N, 1] - force limit
        
        # 1) root_height - 0.79
        base_pos = torch.as_tensor(env_data.base_pos, device=self.device, dtype=torch.float32)
        if base_pos.ndim == 1:
            base_pos = base_pos.unsqueeze(0)  # [1, 3]
        root_height = torch.full((N, 1), 0.79, device=self.device, dtype=torch.float32)  # [N, 1] - fixed height
        
        # 2) target_linvel_b - 从 udp_cmd3 获取 vx, vy (已经是 body frame)
        target_linvel_b = root_cmd7[:, 0:2]

        # use target yaw from UDP root_command (already wxyz in root_cmd7[:, 3:7])
        target_yaw_quat = root_cmd7[:, 3:7]
        target_yaw_quat = target_yaw_quat / (torch.norm(target_yaw_quat, dim=-1, keepdim=True) + 1e-8)

        # current robot yaw from env_data.base_quat (xyzw unless cfg says wxyz)
        # base_quat = torch.as_tensor(env_data.base_quat, device=self.device, dtype=torch.float32)
        base_quat_xyzw = torch.as_tensor(env_data.base_quat, device=self.device, dtype=torch.float32)
        if base_quat_xyzw.ndim == 1:
            base_quat_xyzw = base_quat_xyzw.unsqueeze(0)
        base_quat_wxyz = torch.stack([base_quat_xyzw[:,3], base_quat_xyzw[:,0], base_quat_xyzw[:,1], base_quat_xyzw[:,2]], dim=-1)
        base_quat_wxyz = _quat_normalize_wxyz(base_quat_wxyz)
        # print(f'base_quat_wxyz: {base_quat_wxyz}')
        base_quat = base_quat_wxyz
        
        current_yaw_quat = yaw_quat(base_quat)  # [N, 4]
        
        # Target heading in world frame (x-axis of target frame)
        heading_vec = torch.tensor([1.0, 0.0, 0.0], device=self.device)
        target_heading_w = quat_apply(target_yaw_quat, heading_vec.unsqueeze(0).expand(self.num_envs, -1))  # [N, 3]
        
        # Convert heading to current robot's body frame
        target_heading_b = quat_apply_inverse(current_yaw_quat, target_heading_w)  # [N, 3]
        target_heading_b_xy = target_heading_b[:, :2]  # [N, 2]
        
        # 4) force_safe_limit - 默认值（可以从 cfg 读取）
        force_safe_limit = torch.full((N, 1), 15.0, device=self.device, dtype=torch.float32)
        
        # 拼接成 6 维 command
        command = torch.cat([
            root_height,         # [N, 1]
            target_linvel_b,     # [N, 2]
            target_heading_b_xy,    # [N, 2]
            force_safe_limit,    # [N, 1]
        ], dim=-1)  # [N, 6]
        # print(f"[DEBUG] command: {command.shape}, udp_cmd3: {udp_cmd3.shape}, ee12: {ee12.shape}")

        root_and_wrist_6d = ee12 

        # 2) root ang vel (history_steps=[0])
        # Convert ang_vel from world frame to body frame (MujocoEnv provides world frame)
        base_ang_vel_world = torch.as_tensor(env_data.base_ang_vel, device=self.device, dtype=torch.float32)
        if base_ang_vel_world.ndim == 1:
            base_ang_vel_world = base_ang_vel_world.unsqueeze(0)  # [1,3]
        
        # Convert to body frame using inverse quaternion rotation
        base_ang_vel = _quat_rotate_wxyz(_quat_conj_wxyz(base_quat), base_ang_vel_world)  # [N,3]
        assert base_ang_vel.shape == (N, 3), f"base_ang_vel shape {base_ang_vel.shape} != ({N},3)"
        # print(f'base_ang_vel (body frame): {base_ang_vel}')

        # 3) projected gravity
        g_w = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=torch.float32).view(1, 3).repeat(N, 1)
        projected_gravity = _quat_rotate_wxyz(_quat_conj_wxyz(base_quat), g_w)  # [N,3]

        # 4) joint pos history
        self._step_update_buffers(env_data)

        joint_hist_list = [self._joint_pos_hist[k] for k in self.joint_hist_steps]  # each [N,D]
        joint_pos_history = torch.cat(joint_hist_list, dim=-1)  # [N, len(steps)*D]

        # 5) prev actions (steps=3)
        prev_actions = torch.cat([self._prev_actions[i] for i in range(self.prev_action_steps)], dim=-1)  # [N, 3*action_dim]

        # 6) boot indicator
        boot = torch.full((N, 1), float(self.boot_indicator / self.boot_max), device=self.device, dtype=torch.float32)

        # print(f'command: {command[0].cpu().numpy()}')
        # 7) concat in exact order
        obs = torch.cat([
            boot,                 # [N,1] 没问题
            command,              # [N,cmd_dim] 没问题
            root_and_wrist_6d,    # [N,wrist_dim] 没问题
            base_ang_vel,         # [N,3] 没问题
            projected_gravity,    # [N,3] 没问题
            joint_pos_history,    # [N,len(steps)*D]
            prev_actions,         # [N,3*action_dim]
        ], dim=-1)

        info = {
            "timestep": self.timestep,
            "boot_indicator": self.boot_indicator,
            "command": command.detach().cpu().numpy(),
            "root_and_wrist_6d": root_and_wrist_6d.detach().cpu().numpy(),
            "base_ang_vel": base_ang_vel.detach().cpu().numpy(),
            "projected_gravity": projected_gravity.detach().cpu().numpy(),
            "udp": extra,
        }

        return obs.detach().cpu().numpy().astype(np.float32), info

    # -------------------------
    # Action API (update last_action)
    # -------------------------
    def _compute_alpha(self) -> float:
        alpha = float(self.action_beta)
        if self.alpha_jit is not None and self.alpha_jit_scale is not None:
            jit = (torch.rand(1, device=self.device) * 2.0 - 1.0) * float(self.alpha_jit_scale)
            alpha += float(jit.item())
        alpha_min, alpha_max = self.alpha_wide_range
        alpha = float(np.clip(alpha, alpha_min, alpha_max))
        alpha = float(np.clip(alpha, 0.0, 1.0))
        return alpha

    def _apply_communication_delay(self, action_now: torch.Tensor, alpha: float) -> torch.Tensor:
        """
        Approximate substep-level communication delay when substep index is unavailable.
        Delay is specified in physics steps and converted to a fractional control-step delay.
        """
        if self._action_buf is None or not self.use_communication_delay or self.max_delay <= 0:
            self.applied_action.lerp_(action_now, alpha)
            return self.applied_action

        # push current action into history (index 0 is newest)
        self._action_buf = torch.roll(self._action_buf, shifts=1, dims=0)
        self._action_buf[0] = action_now

        delay_ctrl = float(self._delay_steps) / float(self.comm_delay_decimation)
        idx = int(math.floor(delay_ctrl))
        frac = float(delay_ctrl - idx)
        idx_next = min(idx + 1, self._action_buf.shape[0] - 1)

        delayed_action = (1.0 - frac) * self._action_buf[idx] + frac * self._action_buf[idx_next]
        self.applied_action.lerp_(delayed_action, alpha)
        return self.applied_action

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        # ---- raw obs ----
        obs_raw = np.asarray(obs, dtype=np.float32).reshape(-1)
#         test_raw_str = """9.6000e-01,  7.9000e-01,  0.0000e+00,  0.0000e+00, -9.3493e-01,
# 3.5484e-01,  1.0000e+01,  2.5000e-01,  1.8000e-01,  1.5000e-01,
# 2.5000e-01, -1.8000e-01,  1.5000e-01,  0.0000e+00,  0.0000e+00,
# 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00, -2.8821e-01,
# 2.7058e+00,  5.4152e-01,  9.5908e-02,  2.5870e-03, -9.9539e-01,
# 3.1021e-02, -3.5741e-02, -7.2553e-03, -9.7945e-02,  2.0932e-03,
# -8.2345e-03, -1.2099e-01, -4.2813e-02,  1.6532e-02,  1.0571e-01,
# 1.8581e-01,  1.2712e-01,  1.6606e-01, -3.0418e-01, -3.3349e-01,
# 1.0780e+00, -7.8892e-01,  7.4404e-03,  1.2566e-01,  3.6537e-01,
# -4.5183e-02,  5.2884e-01,  1.1232e+00,  1.0430e-01, -1.1349e-01,
# -6.8449e-02,  2.1818e-02,  4.0483e-01,  1.3530e-01,  7.0194e-02,
# -4.8344e-03, -4.6772e-03, -9.6884e-02,  2.1312e-03, -1.1738e-02,
# -1.3571e-01, -3.6741e-02,  2.2618e-02,  5.8351e-02,  1.2798e-01,
# 1.5097e-01,  1.6300e-01, -2.8051e-01, -3.1908e-01,  1.0881e+00,
# -7.9651e-01,  6.8098e-04,  1.2942e-01,  3.6525e-01, -5.1942e-02,
# 5.2283e-01,  1.1315e+00,  1.2083e-01, -1.3800e-01, -5.7984e-02,
# 3.0080e-02,  4.3327e-01,  1.4359e-01,  7.0194e-02, -4.8344e-03,
# -4.6772e-03, -9.6884e-02,  2.1312e-03, -1.1738e-02, -1.3571e-01,
# -3.6741e-02,  2.2618e-02,  5.8351e-02,  1.2798e-01,  1.5097e-01,
# 1.6300e-01, -2.8051e-01, -3.1908e-01,  1.0881e+00, -7.9651e-01,
# 6.8098e-04,  1.2942e-01,  3.6525e-01, -5.1942e-02,  5.2283e-01,
# 1.1315e+00,  1.2083e-01, -1.3800e-01, -5.7984e-02,  3.0080e-02,
# 4.3327e-01,  1.4359e-01,  7.0194e-02, -4.8344e-03, -4.6772e-03,
# -9.6884e-02,  2.1312e-03, -1.1738e-02, -1.3571e-01, -3.6741e-02,
# 2.2618e-02,  5.8351e-02,  1.2798e-01,  1.5097e-01,  1.6300e-01,
# -2.8051e-01, -3.1908e-01,  1.0881e+00, -7.9651e-01,  6.8098e-04,
# 1.2942e-01,  3.6525e-01, -5.1942e-02,  5.2283e-01,  1.1315e+00,
# 1.2083e-01, -1.3800e-01, -5.7984e-02,  3.0080e-02,  4.3327e-01,
# 1.4359e-01,  7.0194e-02, -4.8344e-03, -4.6772e-03, -9.6884e-02,
# 2.1312e-03, -1.1738e-02, -1.3571e-01, -3.6741e-02,  2.2618e-02,
# 5.8351e-02,  1.2798e-01,  1.5097e-01,  1.6300e-01, -2.8051e-01,
# -3.1908e-01,  1.0881e+00, -7.9651e-01,  6.8098e-04,  1.2942e-01,
# 3.6525e-01, -5.1942e-02,  5.2283e-01,  1.1315e+00,  1.2083e-01,
# -1.3800e-01, -5.7984e-02,  3.0080e-02,  4.3327e-01,  1.4359e-01,
# -2.6854e-01, -2.9400e-01,  2.4846e-03,  5.5790e-01, -5.7061e-01,
# 1.1616e-01,  7.2048e-03,  5.5418e-02, -1.6751e+00,  1.6877e+00,
# 1.6305e+00, -1.5085e+00, -1.4872e+00, -4.7070e-01, -4.8921e-01,
# -3.0785e-01,  2.4810e-01, -1.2617e-01,  1.1586e-01, -2.4713e-01,
# 2.3440e-01, -6.8112e-01, -6.0130e-01, -1.7137e-01,  2.1012e-01,
# -3.8728e-01, -3.3385e-01, -1.5184e-01,  2.1171e-01,  0.0000e+00,
# 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
# 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
# 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
# 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
# 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
# 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
# 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
# 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
# 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
# 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
# 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
# 0.0000e+00,  0.0000e+00"""
        # obs_raw = np.fromstring(test_raw_str, sep=',', dtype=np.float32)
        # ---- normalize obs_raw ----
        # 使用与 torchrl ObservationNorm._apply_transform() 相同的归一化方式
        # 当 standard_normal=True 时: (obs - loc) / scale
        # 用 torch tensor 计算以保持精度一致
        if self._vecnorm_loc is not None and self._vecnorm_scale is not None:
            obs_raw_t = torch.from_numpy(obs_raw)
            obs_norm_t = (obs_raw_t - self._vecnorm_loc) / self._vecnorm_scale
            obs_norm = obs_norm_t.numpy().astype(np.float32).reshape(-1)
            obs_tensor = torch.from_numpy(obs_norm).unsqueeze(0).to(self.device)
        else:
            # no normalization
            obs_tensor = torch.as_tensor(obs_raw, dtype=torch.float32, device=self.device).unsqueeze(0)

        # print(f'norm obs: {obs_tensor[:8].cpu().numpy()}')
        # obs_in = apply_perm_signs(obs_norm, self.obs_perm, self.obs_signs)   # 257 -> 257
        # inp = torch.from_numpy(obs_in).to(self.device).unsqueeze(0)
        # ---- network output (raw action) ----
        with torch.no_grad():
            out = self.model(obs_tensor)
        act_net = out[0] if isinstance(out, (list, tuple)) else out
        act_net = act_net.squeeze(0).detach()  # [29] on device
        # act_net = apply_perm_signs(act_net, self.act_perm, self.act_signs)

        # ---- clamp raw network output ----
        act_clamped = torch.clamp(act_net, -10.0, 10.0)

        # prev_actions history
        self.last_raw_action = act_clamped.detach()

        # ---- smooth in RAW space ----
        last_raw = torch.as_tensor(getattr(self, "last_action_raw", torch.zeros_like(act_clamped)),
                                dtype=torch.float32, device=self.device)
        act_smooth = (1 - self.action_beta) * last_raw + self.action_beta * act_clamped
        # print(f'smoothed action (raw space): {act_smooth[:8].cpu().numpy()}')

        # ✅ update EMA state in RAW space
        self.last_action_raw = act_smooth.detach()

        act_smooth = act_clamped
        # ---- scale (convert RAW -> applied delta space) ----
        if self.action_scales is not None:
            action_scales_t = torch.as_tensor(self.action_scales, device=self.device, dtype=torch.float32)
            act_final = act_smooth * action_scales_t
        else:
            act_final = act_smooth * self.action_scale

        # ---- communication delay emulation in action space ----
        alpha = self._compute_alpha()
        act_applied = self._apply_communication_delay(act_final, alpha)

        # print(f"action", act_final.detach().cpu().numpy().reshape(-1))
        return act_applied.detach().cpu().numpy().reshape(-1)
