#!/usr/bin/env python3
"""
Test script for validating a custom policy configuration

Usage:
    python scripts/test_custom_policy.py --config g1_my_rl
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np


def test_policy_config(config_name: str):
    """Test whether the policy configuration is correct"""
    
    print("=" * 60)
    print(f"Testing Policy Configuration: {config_name}")
    print("=" * 60)
    
    # 1. Test configuration loading
    print("\n[1/6] Loading configuration...")
    try:
        from robojudo.config.config_manager import ConfigManager
        config_manager = ConfigManager(config_name=config_name)
        cfg = config_manager.get_cfg()
        print("✓ Configuration loaded successfully")
        print(f"  - Robot: {cfg.robot}")
        print(f"  - Pipeline: {cfg.pipeline_type}")
    except Exception as e:
        print(f"✗ Failed to load configuration: {e}")
        return False
    
    # 2. Test whether the policy class can be imported
    print("\n[2/6] Checking policy class...")
    try:
        import robojudo.policy
        policy_type = cfg.policy.policy_type
        policy_class = getattr(robojudo.policy, policy_type, None)
        if policy_class is None:
            print(f"✗ Policy class '{policy_type}' not found")
            print(f"  Available policies: {dir(robojudo.policy)}")
            return False
        print(f"✓ Policy class '{policy_type}' found")
    except Exception as e:
        print(f"✗ Failed to import policy: {e}")
        return False
    
    # 3. Test whether the model file exists
    print("\n[3/6] Checking model file...")
    model_path = Path(cfg.policy.policy_file)
    if not model_path.exists():
        print(f"✗ Model file not found: {cfg.policy.policy_file}")
        print(f"  Please ensure your model file is at this location")
        return False
    print(f"✓ Model file exists: {cfg.policy.policy_file}")
    print(f"  File size: {model_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # 4. Test policy initialization
    print("\n[4/6] Initializing policy...")
    try:
        policy = policy_class(cfg_policy=cfg.policy, device="cpu")
        print("✓ Policy initialized successfully")
        print(f"  - Observation DoFs: {policy.num_dofs}")
        print(f"  - Action DoFs: {policy.num_actions}")
        print(f"  - Frequency: {policy.freq} Hz")
        print(f"  - History length: {policy.history_length}")
    except Exception as e:
        print(f"✗ Failed to initialize policy: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 5. Test the observation space
    print("\n[5/6] Testing observation space...")
    try:
        # Create mock environment data
        class MockEnvData:
            def __init__(self, num_dofs):
                self.base_pos = np.array([0.0, 0.0, 0.79], dtype=np.float32)
                self.base_ang_vel = np.zeros(3)
                self.base_lin_vel = np.zeros(3)
                self.base_quat = np.array([1, 0, 0, 0])  # w, x, y, z
                self.dof_pos = np.zeros(num_dofs)
                self.dof_vel = np.zeros(num_dofs)
                self.imu_acc = np.zeros(3)
                self.imu_gyro = np.zeros(3)
        
        env_data = MockEnvData(policy.num_dofs)
        ctrl_data = {}  # empty control data
        
        obs, info = policy.get_observation(env_data, ctrl_data)
        print("✓ Observation generated successfully")
        print(f"  - Observation shape: {obs.shape}")
        print(f"  - Observation range: [{obs.min():.3f}, {obs.max():.3f}]")
        
        # Check for NaN or Inf values
        if np.isnan(obs).any():
            print("  ⚠ Warning: Observation contains NaN values")
        if np.isinf(obs).any():
            print("  ⚠ Warning: Observation contains Inf values")
    except Exception as e:
        print(f"✗ Failed to generate observation: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 6. Test action inference
    print("\n[6/6] Testing action inference...")
    try:
        action = policy.get_action(obs)
        print("✓ Action generated successfully")
        print(f"  - Action shape: {action.shape}")
        print(f"  - Action range: [{action.min():.3f}, {action.max():.3f}]")
        print(f"  - Action mean: {action.mean():.3f}")
        print(f"  - Action std: {action.std():.3f}")
        
        # Check whether the action is reasonable
        if np.abs(action).max() > 100:
            print("  ⚠ Warning: Actions seem very large (> 100)")
        if np.isnan(action).any():
            print("  ⚠ Warning: Action contains NaN values")
            return False
        if np.isinf(action).any():
            print("  ⚠ Warning: Action contains Inf values")
            return False
    except Exception as e:
        print(f"✗ Failed to generate action: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Summary
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
    print("\nYour policy is ready to run!")
    print(f"\nRun with: python scripts/run_pipeline.py --config {config_name}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Test custom policy configuration"
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        required=True,
        help="Name of the config to test"
    )
    args = parser.parse_args()
    
    success = test_policy_config(args.config)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
