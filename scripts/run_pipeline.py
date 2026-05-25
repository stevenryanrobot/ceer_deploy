# Fix OMP perfmance issue on ARM platform (Jetson)
import os
import platform
import sys
from pathlib import Path

if platform.machine().startswith("aarch64"):
    os.environ["OMP_NUM_THREADS"] = "1"

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
import logging
import time

import robojudo.pipeline
from robojudo.config.config_manager import ConfigManager
from robojudo.pipeline.pipeline_cfgs import RlPipelineCfg
from robojudo.pipeline.rl_pipeline import RlPipeline

logger = logging.getLogger("robojudo")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="g1_my_rl",
        help="Name of the config class to use",
    )
    parser.add_argument(
        "--policy-suffix",
        type=str,
        default=None,
        help="Load policy_<suffix>.pt and vecnorm_params_<suffix>.pt",
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

    return args


def main():
    args = parse_args()
    logger.info(f"Using config: {args.config}")
    config_manager = ConfigManager(config_name=args.config)

    cfg: RlPipelineCfg = config_manager.get_cfg()
    if args.policy_suffix:
        if not hasattr(cfg.policy, "policy_suffix"):
            raise AttributeError(f"Policy config {type(cfg.policy).__name__} does not support policy_suffix")
        cfg.policy.policy_suffix = args.policy_suffix
        print(
            "[run_pipeline] policy suffix="
            f"{args.policy_suffix!r}; policy_file={cfg.policy.policy_file}; "
            f"vecnorm_file={getattr(cfg.policy, 'vecnorm_file', None)}",
            flush=True,
        )
    if hasattr(cfg, "env"):
        print(
            "[run_pipeline] env="
            f"{getattr(cfg.env, 'env_type', None)} "
            f"camera_capture_enabled={getattr(cfg.env, 'camera_capture_enabled', None)} "
            f"camera_udp={getattr(cfg.env, 'camera_udp_host', None)}:"
            f"{getattr(cfg.env, 'camera_udp_port', None)}",
            flush=True,
        )

    pipeline_type = cfg.pipeline_type

    pipeline_class: type[RlPipeline] = getattr(robojudo.pipeline, pipeline_type)
    logger.info(f"Using pipeline: {pipeline_type} -> {pipeline_class}")

    pipeline = pipeline_class(cfg=cfg)

    if not cfg.env.is_sim:
        pipeline.prepare()

    try:
        while True:
            time_start = time.time()
            pipeline.step()
            time_end = time.time()
            time_diff = time_end - time_start

            # keep the pipeline running at the desired frequency
            if not cfg.run_fullspeed:
                time_diff = pipeline.dt - time_diff
                if time_diff > 0:
                    time.sleep(time_diff)
                else:
                    if not cfg.env.is_sim:
                        logger.error(f"Warning: frame drop -> {time_diff}")
                        if time_diff < -0.2:
                            logger.critical("Exiting due to excessive frame drop")
                            time.sleep(10)
                            break
    except KeyboardInterrupt:
        logger.info("Interrupted, shutting down pipeline.")
    finally:
        pipeline.env.shutdown()


if __name__ == "__main__":
    main()
