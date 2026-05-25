#!/usr/bin/env python3
"""Export deployment policy.pt and vecnorm_params.pt from a training checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import sys
import types
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn


class ActorStudent(nn.Module):
    def __init__(
        self,
        actor_input_dim: int,
        actor_hidden0: int,
        actor_hidden1: int,
        actor_hidden2: int,
        action_dim: int,
    ) -> None:
        super().__init__()
        self.fc0 = nn.Linear(actor_input_dim, actor_hidden0)
        self.ln0 = nn.LayerNorm(actor_hidden0)
        self.fc1 = nn.Linear(actor_hidden0, actor_hidden1)
        self.ln1 = nn.LayerNorm(actor_hidden1)
        self.fc2 = nn.Linear(actor_hidden1, actor_hidden2)
        self.ln2 = nn.LayerNorm(actor_hidden2)
        self.mean = nn.Linear(actor_hidden2, action_dim)
        self.act = nn.Mish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.ln0(self.fc0(x)))
        x = self.act(self.ln1(self.fc1(x)))
        x = self.act(self.ln2(self.fc2(x)))
        return self.mean(x)


class StudentRollout(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        adapt_hidden0: int,
        adapt_hidden1: int,
        latent_dim: int,
        adapt_joint_hidden0: int,
        adapt_joint_hidden1: int,
        joint_target_dim: int,
        actor_hidden0: int,
        actor_hidden1: int,
        actor_hidden2: int,
        action_dim: int,
    ) -> None:
        super().__init__()
        self.adapt = nn.Sequential(
            nn.Linear(obs_dim, adapt_hidden0),
            nn.LayerNorm(adapt_hidden0),
            nn.Mish(),
            nn.Linear(adapt_hidden0, adapt_hidden1),
            nn.LayerNorm(adapt_hidden1),
            nn.Mish(),
            nn.Linear(adapt_hidden1, latent_dim),
        )
        self.adapt_joint = nn.Sequential(
            nn.Linear(obs_dim, adapt_joint_hidden0),
            nn.LayerNorm(adapt_joint_hidden0),
            nn.Mish(),
            nn.Linear(adapt_joint_hidden0, adapt_joint_hidden1),
            nn.LayerNorm(adapt_joint_hidden1),
            nn.Mish(),
            nn.Linear(adapt_joint_hidden1, joint_target_dim),
        )
        self.actor = ActorStudent(
            actor_input_dim=obs_dim + latent_dim + joint_target_dim,
            actor_hidden0=actor_hidden0,
            actor_hidden1=actor_hidden1,
            actor_hidden2=actor_hidden2,
            action_dim=action_dim,
        )

    def forward(self, obs_key: torch.Tensor) -> torch.Tensor:
        latent = self.adapt(obs_key)
        joint_target = self.adapt_joint(obs_key)
        return self.actor(torch.cat([obs_key, latent, joint_target], dim=-1))


def ensure_module(name: str) -> types.ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    module = types.ModuleType(name)
    sys.modules[name] = module
    if "." in name:
        parent_name, child_name = name.rsplit(".", 1)
        parent = ensure_module(parent_name)
        setattr(parent, child_name, module)
    return module


def install_checkpoint_shims() -> None:
    ppo = ensure_module("active_adaptation.learning.ppo.ppo")

    class PPOConfig:
        pass

    class PPOPolicy:
        pass

    PPOConfig.__module__ = "active_adaptation.learning.ppo.ppo"
    PPOPolicy.__module__ = "active_adaptation.learning.ppo.ppo"
    ppo.PPOConfig = PPOConfig
    ppo.PPOPolicy = PPOPolicy


def sha256sum(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_signature(tensor: torch.Tensor) -> tuple[tuple[int, ...], str, str]:
    contiguous = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(contiguous.numpy().tobytes()).hexdigest()
    return tuple(contiguous.shape), str(contiguous.dtype), digest


def flatten_tensors(value: Any, prefix: str = "") -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    if isinstance(value, torch.Tensor):
        tensors[prefix or "<tensor>"] = value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            tensors.update(flatten_tensors(item, name))
    elif isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            name = f"{prefix}.{idx}" if prefix else str(idx)
            tensors.update(flatten_tensors(item, name))
    return tensors


def load_checkpoint(path: Path) -> Mapping[str, Any]:
    install_checkpoint_shims()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Expected checkpoint mapping, got {type(checkpoint).__name__}")
    return checkpoint


def require_tensor(mapping: Mapping[str, Any], key: str) -> torch.Tensor:
    if key in mapping:
        value = mapping[key]
    else:
        value = mapping
        parts = key.split(".")
        idx = 0
        while idx < len(parts):
            if not isinstance(value, Mapping):
                raise KeyError(key)
            remaining = ".".join(parts[idx:])
            if remaining in value:
                value = value[remaining]
                idx = len(parts)
                break
            part = parts[idx]
            if part not in value:
                raise KeyError(key)
            value = value[part]
            idx += 1
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected tensor at key {key}, got {type(value).__name__}")
    return value.detach().cpu()


def build_policy_state(checkpoint: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    policy = checkpoint["policy"]
    if not isinstance(policy, Mapping):
        raise TypeError("checkpoint['policy'] is not a mapping")

    mapping = {
        "adapt.0.weight": "adapt_module.module.0.0.weight",
        "adapt.0.bias": "adapt_module.module.0.0.bias",
        "adapt.1.weight": "adapt_module.module.0.1.weight",
        "adapt.1.bias": "adapt_module.module.0.1.bias",
        "adapt.3.weight": "adapt_module.module.0.3.weight",
        "adapt.3.bias": "adapt_module.module.0.3.bias",
        "adapt.4.weight": "adapt_module.module.0.4.weight",
        "adapt.4.bias": "adapt_module.module.0.4.bias",
        "adapt.6.weight": "adapt_module.module.1.weight",
        "adapt.6.bias": "adapt_module.module.1.bias",
        "adapt_joint.0.weight": "adapt_joint_module.module.0.0.weight",
        "adapt_joint.0.bias": "adapt_joint_module.module.0.0.bias",
        "adapt_joint.1.weight": "adapt_joint_module.module.0.1.weight",
        "adapt_joint.1.bias": "adapt_joint_module.module.0.1.bias",
        "adapt_joint.3.weight": "adapt_joint_module.module.0.3.weight",
        "adapt_joint.3.bias": "adapt_joint_module.module.0.3.bias",
        "adapt_joint.4.weight": "adapt_joint_module.module.0.4.weight",
        "adapt_joint.4.bias": "adapt_joint_module.module.0.4.bias",
        "adapt_joint.6.weight": "adapt_joint_module.module.1.weight",
        "adapt_joint.6.bias": "adapt_joint_module.module.1.bias",
        "actor.fc0.weight": "actor_student.module.0.module.1.module.0.weight",
        "actor.fc0.bias": "actor_student.module.0.module.1.module.0.bias",
        "actor.ln0.weight": "actor_student.module.0.module.1.module.1.weight",
        "actor.ln0.bias": "actor_student.module.0.module.1.module.1.bias",
        "actor.fc1.weight": "actor_student.module.0.module.1.module.3.weight",
        "actor.fc1.bias": "actor_student.module.0.module.1.module.3.bias",
        "actor.ln1.weight": "actor_student.module.0.module.1.module.4.weight",
        "actor.ln1.bias": "actor_student.module.0.module.1.module.4.bias",
        "actor.fc2.weight": "actor_student.module.0.module.1.module.6.weight",
        "actor.fc2.bias": "actor_student.module.0.module.1.module.6.bias",
        "actor.ln2.weight": "actor_student.module.0.module.1.module.7.weight",
        "actor.ln2.bias": "actor_student.module.0.module.1.module.7.bias",
        "actor.mean.weight": "actor_student.module.0.module.2.module.actor_mean.weight",
        "actor.mean.bias": "actor_student.module.0.module.2.module.actor_mean.bias",
    }
    return {target: require_tensor(policy, source) for target, source in mapping.items()}


def build_student_rollout_from_state(state: Mapping[str, torch.Tensor]) -> StudentRollout:
    obs_dim = state["adapt.0.weight"].shape[1]
    adapt_hidden0 = state["adapt.0.weight"].shape[0]
    adapt_hidden1 = state["adapt.3.weight"].shape[0]
    latent_dim = state["adapt.6.weight"].shape[0]

    adapt_joint_hidden0 = state["adapt_joint.0.weight"].shape[0]
    adapt_joint_hidden1 = state["adapt_joint.3.weight"].shape[0]
    joint_target_dim = state["adapt_joint.6.weight"].shape[0]

    actor_input_dim = state["actor.fc0.weight"].shape[1]
    expected_actor_input_dim = obs_dim + latent_dim + joint_target_dim
    if actor_input_dim != expected_actor_input_dim:
        raise ValueError(
            "Unexpected actor input dimension: "
            f"actor.fc0 expects {actor_input_dim}, but obs+latent+joint is {expected_actor_input_dim}"
        )

    return StudentRollout(
        obs_dim=obs_dim,
        adapt_hidden0=adapt_hidden0,
        adapt_hidden1=adapt_hidden1,
        latent_dim=latent_dim,
        adapt_joint_hidden0=adapt_joint_hidden0,
        adapt_joint_hidden1=adapt_joint_hidden1,
        joint_target_dim=joint_target_dim,
        actor_hidden0=state["actor.fc0.weight"].shape[0],
        actor_hidden1=state["actor.fc1.weight"].shape[0],
        actor_hidden2=state["actor.fc2.weight"].shape[0],
        action_dim=state["actor.mean.weight"].shape[0],
    )


def export_policy(checkpoint: Mapping[str, Any], output_path: Path) -> StudentRollout:
    state = build_policy_state(checkpoint)
    model = build_student_rollout_from_state(state).eval()
    model.load_state_dict(state, strict=True)
    scripted = torch.jit.script(model)
    scripted.save(str(output_path))
    return model


def export_vecnorm(checkpoint: Mapping[str, Any], output_path: Path) -> dict[str, dict[str, torch.Tensor]]:
    env = checkpoint["env"]
    if not isinstance(env, Mapping):
        raise TypeError("checkpoint['env'] is not a mapping")

    transforms = {
        "policy": "transforms.2",
        "joint_target": "transforms.3",
        "priv": "transforms.4",
        "priv_critic": "transforms.5",
    }
    vecnorm = {
        name: {
            "loc": require_tensor(env, f"{prefix}.loc"),
            "scale": require_tensor(env, f"{prefix}.scale"),
        }
        for name, prefix in transforms.items()
    }
    torch.save(vecnorm, output_path)
    return vecnorm


def load_policy_state(path: Path) -> dict[str, torch.Tensor]:
    policy = torch.jit.load(str(path), map_location="cpu")
    return dict(policy.state_dict())


def compare_state_dicts(label: str, exported: Mapping[str, torch.Tensor], gt: Mapping[str, torch.Tensor]) -> bool:
    exported_keys = set(exported)
    gt_keys = set(gt)
    print(f"\n[{label}]")
    print(f"  exported tensors: {len(exported)}")
    print(f"  gt tensors: {len(gt)}")
    print(f"  keys only in exported: {len(exported_keys - gt_keys)}")
    print(f"  keys only in gt: {len(gt_keys - exported_keys)}")

    mismatches: list[str] = []
    for key in sorted(exported_keys & gt_keys):
        left = exported[key].detach().cpu()
        right = gt[key].detach().cpu()
        if tuple(left.shape) != tuple(right.shape) or left.dtype != right.dtype:
            mismatches.append(
                f"{key}: metadata differs "
                f"exported shape={tuple(left.shape)}, dtype={left.dtype}; "
                f"gt shape={tuple(right.shape)}, dtype={right.dtype}"
            )
        elif not torch.equal(left, right):
            diff = (left.to(torch.float64) - right.to(torch.float64)).abs()
            mismatches.append(f"{key}: max_abs={diff.max().item() if diff.numel() else 0.0:.8g}")

    print(f"  value mismatches: {len(mismatches)}")
    for item in mismatches[:20]:
        print(f"    - {item}")
    return not (exported_keys - gt_keys or gt_keys - exported_keys or mismatches)


def compare_vecnorm(exported: Mapping[str, Mapping[str, torch.Tensor]], gt_path: Path) -> bool:
    gt = torch.load(gt_path, map_location="cpu", weights_only=False)
    flat_exported = flatten_tensors(exported)
    flat_gt = flatten_tensors(gt)
    return compare_state_dicts("vecnorm compare with gt", flat_exported, flat_gt)


def compare_policy_outputs(exported_path: Path, gt_path: Path) -> bool:
    exported = torch.jit.load(str(exported_path), map_location="cpu").eval()
    gt = torch.jit.load(str(gt_path), map_location="cpu").eval()

    obs_dim = exported.state_dict()["adapt.0.weight"].shape[1]
    gt_obs_dim = gt.state_dict()["adapt.0.weight"].shape[1]
    if obs_dim != gt_obs_dim:
        print("\n[policy forward compare with gt]")
        print(f"  skipped: exported obs_dim={obs_dim}, gt obs_dim={gt_obs_dim}")
        return False

    torch.manual_seed(0)
    obs = torch.randn(7, obs_dim)
    with torch.no_grad():
        left = exported(obs)
        right = gt(obs)
    max_abs = (left - right).abs().max().item()
    same = torch.equal(left, right)
    close = torch.allclose(left, right, atol=0.0, rtol=0.0)
    print("\n[policy forward compare with gt]")
    print(f"  output shape: {tuple(left.shape)}")
    print(f"  torch.equal: {same}")
    print(f"  exact allclose: {close}")
    print(f"  max_abs_diff: {max_abs:.8g}")
    return bool(same)


def compare_policy_by_content(exported_path: Path, gt_path: Path) -> bool:
    exported = load_policy_state(exported_path)
    gt = load_policy_state(gt_path)
    exported_sigs = {tensor_signature(tensor) for tensor in exported.values()}
    gt_sigs = {tensor_signature(tensor) for tensor in gt.values()}
    print("\n[policy tensor content compare with gt]")
    print(f"  exported tensor contents in gt: {len(exported_sigs & gt_sigs)} / {len(exported_sigs)}")
    return exported_sigs == gt_sigs


def model_suffix(value: str | None) -> str:
    if value is None:
        return ""
    suffix = value.strip()
    if not suffix:
        return ""
    return suffix if suffix.startswith("_") else f"_{suffix}"


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    model_dir = repo_root / "assets/models/g1/my_custom"
    parser = argparse.ArgumentParser(
        description="Export policy.pt and vecnorm_params.pt from checkpoint_final.pt.",
        allow_abbrev=False,
    )
    parser.add_argument("--checkpoint", type=Path, default=model_dir / "checkpoint_final.pt")
    parser.add_argument("--policy-suffix", type=str, default=None, help="Export policy_<suffix>.pt and vecnorm_params_<suffix>.pt")
    parser.add_argument("--policy-out", type=Path, default=None, help="Explicit policy output path.")
    parser.add_argument("--vecnorm-out", type=Path, default=None, help="Explicit vecnorm output path.")
    parser.add_argument("--gt-policy", type=Path, default=model_dir / "policy.pt")
    parser.add_argument("--gt-vecnorm", type=Path, default=model_dir / "vecnorm_params.pt")
    parser.add_argument("--overwrite", action="store_true", default=True, help="Allow overwriting output files.")
    parser.add_argument("--no-overwrite", dest="overwrite", action="store_false", help="Refuse to overwrite output files.")
    parser.add_argument("--skip-gt-check", action="store_true", help="Only export files; do not compare against gt files.")
    parser.add_argument(
        "--strict-gt-check",
        action="store_true",
        help="Return non-zero when export succeeds but gt consistency check fails.",
    )
    args, unknown = parser.parse_known_args()

    suffix_flags = []
    for item in unknown:
        if item.startswith("--") and len(item) > 2:
            suffix_flags.append(item[2:])
        else:
            parser.error(f"unrecognized argument: {item}")

    if args.policy_suffix is not None and suffix_flags:
        parser.error("use either --policy-suffix or one shortcut suffix flag like --gt, not both")
    if len(suffix_flags) > 1:
        parser.error(f"expected at most one shortcut suffix flag, got: {', '.join('--' + x for x in suffix_flags)}")
    if suffix_flags:
        args.policy_suffix = suffix_flags[0]

    suffix = model_suffix(args.policy_suffix)
    if args.policy_out is None:
        args.policy_out = model_dir / f"policy{suffix}.pt"
    if args.vecnorm_out is None:
        args.vecnorm_out = model_dir / f"vecnorm_params{suffix}.pt"

    return args


def main() -> int:
    args = parse_args()
    for path in (args.checkpoint,):
        if not path.exists():
            print(f"Missing file: {path}", file=sys.stderr)
            return 2

    if not args.skip_gt_check:
        for path in (args.gt_policy, args.gt_vecnorm):
            if not path.exists():
                print(f"GT file not found; export will continue without gt check: {path}")
                args.skip_gt_check = True
                break

    for path in (args.policy_out, args.vecnorm_out):
        if path.exists() and not args.overwrite:
            print(f"Output exists, pass --overwrite to replace it: {path}", file=sys.stderr)
            return 2

    checkpoint = load_checkpoint(args.checkpoint)
    export_policy(checkpoint, args.policy_out)
    exported_vecnorm = export_vecnorm(checkpoint, args.vecnorm_out)

    print("[exported files]")
    for label, path in (("policy", args.policy_out), ("vecnorm", args.vecnorm_out)):
        print(f"  {label}: {path}")
        print(f"    size: {path.stat().st_size:,} bytes")
        print(f"    sha256: {sha256sum(path)}")

    if args.skip_gt_check:
        print("\n[Conclusion]")
        print("  Export succeeded. GT consistency check was skipped.")
        return 0

    policy_state_ok = compare_state_dicts(
        "policy state_dict compare with gt",
        load_policy_state(args.policy_out),
        load_policy_state(args.gt_policy),
    )
    policy_content_ok = compare_policy_by_content(args.policy_out, args.gt_policy)
    policy_output_ok = compare_policy_outputs(args.policy_out, args.gt_policy)
    vecnorm_ok = compare_vecnorm(exported_vecnorm, args.gt_vecnorm)

    print("\n[Conclusion]")
    if policy_state_ok and policy_content_ok and policy_output_ok and vecnorm_ok:
        print("  Export succeeded. Exported policy and vecnorm are exactly consistent with the existing gt files.")
        return 0

    print("  Export finished, but at least one gt consistency check failed. See sections above.")
    return 1 if args.strict_gt_check else 0


if __name__ == "__main__":
    raise SystemExit(main())
