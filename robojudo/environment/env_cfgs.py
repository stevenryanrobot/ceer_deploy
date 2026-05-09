from robojudo.config import Config
from robojudo.tools.tool_cfgs import DoFConfig, ForwardKinematicCfg


class EnvCfg(Config):
    env_type: str  # name of the environment class
    is_sim: bool = False

    urdf: str | None = None
    xml: str
    body_names: list[str] | None = None

    dof: DoFConfig

    forward_kinematic: ForwardKinematicCfg | None = None
    update_with_fk: bool = False
    """Whether to update info from fk"""
    torso_name: str = "torso_link"
    """Name of the torso link, used in fk info extraction"""

    born_place_align: bool = True
    """Whether to align the born place to zero position and heading"""


class MujocoEnvCfg(EnvCfg):
    env_type: str = "MujocoEnv"
    is_sim: bool = True
    # ====== ENV CONFIGURATION ======
    sim_duration: float = 60.0
    sim_dt: float = 0.001
    sim_decimation: int = 20

    visualize_extras: bool = True  # TODO: remove

    camera_capture_enabled: bool = False
    camera_name: str = "robot_pov"
    camera_capture_interval_s: float = 1.0
    camera_image_width: int = 640
    camera_image_height: int = 480
    camera_output_dir: str = "logs/robot_pov"
    camera_udp_host: str = "127.0.0.1"
    camera_udp_port: int = 15002
    camera_jpeg_quality: int = 75
    camera_udp_max_chunk_size: int = 8192

    viewer_record_enabled: bool = False
    viewer_record_fps: float = 30.0
    viewer_record_output_dir: str = "/Users/ajayvikram/Desktop/Duke/Robotics/FailureRecovery/videos"
