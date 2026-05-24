import logging
import os
import shutil
import socket
import struct
import subprocess
import sys
import time
from io import BytesIO
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
VIEWER_ROOT = REPO_ROOT / "third_party" / "mujoco_viewer"
if str(VIEWER_ROOT) not in sys.path:
    sys.path.insert(0, str(VIEWER_ROOT))

import mujoco
import mujoco_viewer
import numpy as np

from robojudo.environment import Environment, env_registry
from robojudo.environment.env_cfgs import MujocoEnvCfg
from robojudo.environment.utils.mujoco_viz import MujocoVisualizer
from robojudo.utils.util_func import quat_rotate_inverse_np, quatToEuler

logger = logging.getLogger(__name__)

VLM_IMAGE_MAGIC = b"VLM1"
VLM_IMAGE_HEADER_FMT = "<4sIQHH"
VLM_IMAGE_HEADER_SIZE = struct.calcsize(VLM_IMAGE_HEADER_FMT)


@env_registry.register
class MujocoEnv(Environment):
    cfg_env: MujocoEnvCfg

    def __init__(self, cfg_env: MujocoEnvCfg, device="cpu"):
        super().__init__(cfg_env=cfg_env, device=device)

        self.sim_duration = cfg_env.sim_duration
        self.sim_dt = cfg_env.sim_dt
        self.sim_decimation = cfg_env.sim_decimation
        self.control_dt = self.sim_dt * self.sim_decimation

        logger.info("Loading Mujoco XML: %s", cfg_env.xml)
        self.model = mujoco.MjModel.from_xml_path(cfg_env.xml)  # pyright: ignore[reportAttributeAccessIssue]
        self._dof_qpos_indices, self._dof_qvel_indices = self._resolve_dof_indices()
        self.model.opt.timestep = self.sim_dt
        self.data = mujoco.MjData(self.model)  # pyright: ignore[reportAttributeAccessIssue]
        # mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_step(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]

        self.viewer = mujoco_viewer.MujocoViewer(
            self.model,
            self.data,
            width=1200,
            height=900,
            hide_menus=True,
            diable_key_callbacks=True,
        )
        self.viewer.cam.distance = 3.0
        self.viewer.cam.elevation = -10.0
        self.viewer.cam.azimuth = 180.0
        # self.viewer._paused = True

        if cfg_env.visualize_extras:
            self.visualizer = MujocoVisualizer(self.viewer)
        else:
            self.visualizer = None

        self._init_viewer_recording()
        self._init_camera_capture()

        self.last_time = time.time()

        self.update()  # get initial state

    def reborn(self, init_qpos=None):
        if init_qpos is not None:
            self.data.qpos[0:7] = init_qpos
            self.data.qvel[:] = 0.0
            self.data.ctrl[:] = 0.0
        else:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)  # pyright: ignore[reportAttributeAccessIssue]
        mujoco.mj_forward(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]

    def reset(self):
        if self.born_place_align:  # TODO: merge
            self.born_place_align = False  # disable during reset
            self.update()
            self.born_place_align = True  # enable after reset
            self.set_born_place()
            self.update()

    def set_gains(self, stiffness, damping):
        assert len(stiffness) == self.num_dofs and len(damping) == self.num_dofs
        self.stiffness = np.asarray(stiffness)
        self.damping = np.asarray(damping)

    def self_check(self):
        pass

    def _resolve_dof_indices(self):
        qpos_indices = []
        qvel_indices = []
        for joint_name in self.joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id == -1:
                raise ValueError(f"Joint {joint_name} not found in Mujoco model.")
            qpos_indices.append(int(self.model.jnt_qposadr[joint_id]))
            qvel_indices.append(int(self.model.jnt_dofadr[joint_id]))

        return np.asarray(qpos_indices, dtype=np.int32), np.asarray(qvel_indices, dtype=np.int32)

    def _env_flag(self, name: str, default: bool) -> bool:
        value = os.getenv(name)
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "on"}

    def _init_viewer_recording(self):
        self.viewer_record_enabled = self._env_flag(
            "MUJOCO_VIEWER_RECORD",
            bool(getattr(self.cfg_env, "viewer_record_enabled", False)),
        )
        self.viewer_record_writer = None
        self.viewer_record_path = None
        self.viewer_record_backend = None
        self.viewer_record_width = 0
        self.viewer_record_height = 0
        self.viewer_record_frame_count = 0
        self.viewer_record_next_time = time.time()
        self.viewer_record_fps = float(
            os.getenv(
                "MUJOCO_VIEWER_RECORD_FPS",
                str(getattr(self.cfg_env, "viewer_record_fps", 30.0)),
            )
        )
        self.viewer_record_period = 1.0 / max(self.viewer_record_fps, 1e-3)

        if not self.viewer_record_enabled:
            return

        output_dir = Path(
            os.getenv(
                "MUJOCO_VIEWER_RECORD_DIR",
                str(getattr(self.cfg_env, "viewer_record_output_dir", "logs/videos")),
            )
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        self.viewer_record_path = output_dir / f"viewer_{time.strftime('%Y%m%d_%H%M%S')}.mp4"

        writer_error = None
        opencv_error = None
        try:
            import imageio.v2 as imageio

            self.viewer_record_writer = imageio.get_writer(
                self.viewer_record_path.as_posix(),
                fps=self.viewer_record_fps,
                macro_block_size=None,
            )
            self.viewer_record_backend = "imageio"
        except Exception as e:
            writer_error = e

        if self.viewer_record_writer is None:
            try:
                import cv2

                width, height = self.viewer.viewport.width, self.viewer.viewport.height
                if width <= 0 or height <= 0:
                    width, height = 1200, 900
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self.viewer_record_writer = cv2.VideoWriter(
                    self.viewer_record_path.as_posix(),
                    fourcc,
                    self.viewer_record_fps,
                    (int(width), int(height)),
                )
                if not self.viewer_record_writer.isOpened():
                    self.viewer_record_writer.release()
                    self.viewer_record_writer = None
                    raise RuntimeError("OpenCV VideoWriter failed to open")
                self.viewer_record_backend = "opencv"
            except Exception as e:
                opencv_error = e

        if self.viewer_record_writer is None:
            try:
                ffmpeg_bin = os.getenv("FFMPEG") or shutil.which("ffmpeg")
                if ffmpeg_bin is None:
                    raise RuntimeError("ffmpeg not found on PATH")

                width, height = self.viewer.viewport.width, self.viewer.viewport.height
                if width <= 0 or height <= 0:
                    width, height = 1200, 900
                width = int(width) - (int(width) % 2)
                height = int(height) - (int(height) % 2)
                self.viewer_record_width = width
                self.viewer_record_height = height

                self.viewer_record_writer = subprocess.Popen(
                    [
                        ffmpeg_bin,
                        "-y",
                        "-f",
                        "rawvideo",
                        "-vcodec",
                        "rawvideo",
                        "-pix_fmt",
                        "rgb24",
                        "-s",
                        f"{width}x{height}",
                        "-r",
                        str(self.viewer_record_fps),
                        "-i",
                        "-",
                        "-an",
                        "-vcodec",
                        "libx264",
                        "-preset",
                        "ultrafast",
                        "-tune",
                        "zerolatency",
                        "-pix_fmt",
                        "yuv420p",
                        self.viewer_record_path.as_posix(),
                    ],
                    stdin=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                )
                self.viewer_record_backend = "ffmpeg"
            except Exception as e:
                logger.warning(
                    "Viewer recording disabled: imageio=%s, opencv=%s, ffmpeg=%s",
                    writer_error,
                    opencv_error,
                    e,
                )
                self.viewer_record_enabled = False
                return

        print(
            f"[viewer record] saving {self.viewer_record_path} "
            f"({self.viewer_record_backend})",
            flush=True,
        )

    def _maybe_record_viewer_frame(self):
        if not self.viewer_record_enabled or self.viewer_record_writer is None:
            return

        now = time.time()
        if now < self.viewer_record_next_time:
            return
        self.viewer_record_next_time = now + self.viewer_record_period

        width = int(self.viewer.viewport.width)
        height = int(self.viewer.viewport.height)
        if width <= 0 or height <= 0:
            return

        img = np.zeros((height, width, 3), dtype=np.uint8)
        mujoco.mjr_readPixels(img, None, self.viewer.viewport, self.viewer.ctx)
        img = np.ascontiguousarray(np.flipud(img))
        if self.viewer_record_backend == "opencv":
            self.viewer_record_writer.write(img[:, :, ::-1])
            self.viewer_record_frame_count += 1
        elif self.viewer_record_backend == "ffmpeg":
            if self.viewer_record_writer.poll() is not None or self.viewer_record_writer.stdin is None:
                logger.warning("Viewer recording stopped: ffmpeg exited.")
                self.viewer_record_enabled = False
                return
            img = img[: self.viewer_record_height, : self.viewer_record_width]
            if img.shape[:2] != (self.viewer_record_height, self.viewer_record_width):
                return
            try:
                self.viewer_record_writer.stdin.write(img.tobytes())
                self.viewer_record_frame_count += 1
            except OSError as e:
                logger.warning("Viewer recording stopped: ffmpeg write failed: %s", e)
                self.viewer_record_enabled = False
        else:
            self.viewer_record_writer.append_data(img)
            self.viewer_record_frame_count += 1

    def _init_camera_capture(self):
        self.camera_capture_enabled = bool(self.cfg_env.camera_capture_enabled)
        self.camera_renderer = None
        self.camera_image_writer = None
        self.camera_udp_sock = None
        self.camera_udp_addr = None
        self.camera_frame_i = 0
        self.camera_udp_printed_first_frame = False
        self.next_camera_capture_time = time.time()

        print(
            "[CEER camera] init "
            f"enabled={self.camera_capture_enabled} "
            f"name={self.cfg_env.camera_name} "
            f"host={getattr(self.cfg_env, 'camera_udp_host', None)} "
            f"port={getattr(self.cfg_env, 'camera_udp_port', None)}",
            flush=True,
        )

        if not self.camera_capture_enabled:
            return

        camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.cfg_env.camera_name)
        if camera_id == -1:
            logger.warning("Camera capture disabled: camera '%s' not found.", self.cfg_env.camera_name)
            self.camera_capture_enabled = False
            return

        try:
            import imageio.v2 as imageio
        except ImportError:
            logger.warning("Camera capture disabled: install imageio to encode JPEG frames.")
            self.camera_capture_enabled = False
            return

        self.camera_image_writer = imageio
        self.camera_udp_addr = (
            str(self.cfg_env.camera_udp_host),
            int(self.cfg_env.camera_udp_port),
        )
        self.camera_udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.camera_udp_sock.setblocking(False)
        self.camera_renderer = mujoco.Renderer(
            self.model,
            height=int(self.cfg_env.camera_image_height),
            width=int(self.cfg_env.camera_image_width),
        )
        logger.info(
            "Sending camera '%s' every %.2fs to udp://%s:%d",
            self.cfg_env.camera_name,
            self.cfg_env.camera_capture_interval_s,
            self.camera_udp_addr[0],
            self.camera_udp_addr[1],
        )
        print(
            f"[CEER camera] sending '{self.cfg_env.camera_name}' every "
            f"{self.cfg_env.camera_capture_interval_s:.2f}s to "
            f"udp://{self.camera_udp_addr[0]}:{self.camera_udp_addr[1]}",
            flush=True,
        )

    def _encode_camera_jpeg(self, rgb: np.ndarray) -> bytes:
        buf = BytesIO()
        quality = max(1, min(100, int(self.cfg_env.camera_jpeg_quality)))
        self.camera_image_writer.imwrite(buf, rgb, format="jpeg", quality=quality)
        return buf.getvalue()

    def _send_camera_jpeg(self, jpeg: bytes, stamp_ms: int):
        if self.camera_udp_sock is None or self.camera_udp_addr is None:
            return

        max_payload = 9000 - VLM_IMAGE_HEADER_SIZE
        max_chunk = max(512, min(int(self.cfg_env.camera_udp_max_chunk_size), max_payload))
        chunk_count = max(1, (len(jpeg) + max_chunk - 1) // max_chunk)
        if chunk_count > 65535:
            logger.warning("Camera JPEG too large for UDP chunk protocol: %d bytes", len(jpeg))
            return

        frame_id = self.camera_frame_i & 0xFFFFFFFF
        for chunk_index in range(chunk_count):
            start = chunk_index * max_chunk
            chunk = jpeg[start:start + max_chunk]
            header = struct.pack(
                VLM_IMAGE_HEADER_FMT,
                VLM_IMAGE_MAGIC,
                frame_id,
                stamp_ms,
                chunk_index,
                chunk_count,
            )
            try:
                self.camera_udp_sock.sendto(header + chunk, self.camera_udp_addr)
            except BlockingIOError:
                return
            except OSError as e:
                logger.warning("Camera UDP send failed: %s", e)
                return
        logger.info(
            "Sent camera frame %d: %d bytes in %d UDP chunks",
            frame_id,
            len(jpeg),
            chunk_count,
        )
        if not self.camera_udp_printed_first_frame:
            print(
                f"[CEER camera] sent first frame {frame_id}: "
                f"{len(jpeg)} bytes in {chunk_count} UDP chunks",
                flush=True,
            )
            self.camera_udp_printed_first_frame = True

    def _maybe_capture_camera(self):
        if not self.camera_capture_enabled:
            return

        now = time.time()
        if now < self.next_camera_capture_time:
            return

        interval_s = max(float(self.cfg_env.camera_capture_interval_s), 1e-3)
        self.next_camera_capture_time = now + interval_s
        self.camera_frame_i += 1

        self.camera_renderer.update_scene(self.data, camera=self.cfg_env.camera_name)
        rgb = self.camera_renderer.render()

        stamp_ms = int(now * 1000)
        jpeg = self._encode_camera_jpeg(rgb)
        self._send_camera_jpeg(jpeg, stamp_ms)

    def set_born_place(self, quat: np.ndarray | None = None, pos: np.ndarray | None = None):
        quat_ = self.base_quat if quat is None else quat
        pos_ = self.base_pos if pos is None else pos
        super().set_born_place(quat_, pos_)

    def update(self, simple=False):  # TODO: clean sensors in xml
        """simple: only update dof pos & vel"""
        dof_pos = self.data.qpos[self._dof_qpos_indices].astype(np.float32)
        dof_vel = self.data.qvel[self._dof_qvel_indices].astype(np.float32)

        self._dof_pos = dof_pos.copy()
        self._dof_vel = dof_vel.copy()

        if simple:
            return

        quat = self.data.qpos.astype(np.float32)[3:7][[1, 2, 3, 0]]
        ang_vel = self.data.qvel.astype(np.float32)[3:6]
        base_pos = self.data.qpos.astype(np.float32)[:3]
        lin_vel = self.data.qvel.astype(np.float32)[0:3]

        if self.born_place_align:
            quat, base_pos = self.base_align.align_transform(quat, base_pos)

        lin_vel = quat_rotate_inverse_np(quat, lin_vel)
        rpy = quatToEuler(quat)

        self._base_rpy = rpy.copy()
        self._base_quat = quat.copy()
        self._base_ang_vel = ang_vel.copy()

        self._base_pos = base_pos.copy()
        self._base_lin_vel = lin_vel.copy()

        if self.update_with_fk:
            fk_info = self.fk()
            self._fk_info = fk_info.copy()
            self._torso_ang_vel = fk_info[self._torso_name]["ang_vel"]
            self._torso_quat = fk_info[self._torso_name]["quat"]
            self._torso_pos = fk_info[self._torso_name]["pos"]

        self._dynamic_objects = self._get_dynamic_objects()

    def _get_dynamic_objects(self):
        objects = []
        for body_name in ("box",):
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id == -1:
                continue

            pos = self.data.xpos[body_id].astype(np.float32).copy()
            quat_wxyz = self.data.xquat[body_id].astype(np.float32).copy()
            quat_xyzw = quat_wxyz[[1, 2, 3, 0]]

            if self.born_place_align:
                quat_xyzw, pos = self.base_align.align_transform(quat_xyzw, pos)
                quat_xyzw = quat_xyzw.astype(np.float32)
                pos = pos.astype(np.float32)

            objects.append((body_name, pos, quat_xyzw))
        return objects

    def step(self, pd_target, hand_pose=None):
        assert len(pd_target) == self.num_dofs, "pd_target len should be num_dofs of env"

        # print(f'pd_target: {pd_target}')

        if hand_pose is not None:
            logger.info("Hand pose-->", hand_pose)

        self.viewer.cam.lookat = self.data.qpos.astype(np.float32)[:3]
        if self.viewer.is_alive:
            mujoco.mj_forward(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]
            self.viewer.render()
            self._maybe_record_viewer_frame()

        for _ in range(self.sim_decimation):
            torque = (pd_target - self.dof_pos) * self.stiffness - self.dof_vel * self.damping
            torque = np.clip(torque, -self.torque_limits, self.torque_limits)

            self.data.ctrl = torque

            mujoco.mj_step(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]
            self.update(simple=True)
        self.update(simple=False)
        self._maybe_capture_camera()

    def shutdown(self):
        if self.viewer_record_writer is not None:
            if self.viewer_record_backend == "opencv":
                self.viewer_record_writer.release()
            elif self.viewer_record_backend == "ffmpeg":
                if self.viewer_record_writer.stdin is not None:
                    self.viewer_record_writer.stdin.close()
                try:
                    close_timeout = float(os.getenv("MUJOCO_VIEWER_RECORD_CLOSE_TIMEOUT", "30.0"))
                    self.viewer_record_writer.wait(timeout=close_timeout)
                except subprocess.TimeoutExpired:
                    logger.warning(
                        "Viewer recording did not finalize within timeout; video may be invalid."
                    )
                    self.viewer_record_writer.kill()
                    self.viewer_record_writer.wait()
            else:
                self.viewer_record_writer.close()
            self.viewer_record_writer = None
            if self.viewer_record_path is not None:
                print(
                    f"[viewer record] saved {self.viewer_record_path} "
                    f"({self.viewer_record_frame_count} frames)",
                    flush=True,
                )
        if self.camera_renderer is not None and hasattr(self.camera_renderer, "close"):
            self.camera_renderer.close()
        if self.camera_udp_sock is not None:
            self.camera_udp_sock.close()
        self.viewer.close()


if __name__ == "__main__":
    from robojudo.config.g1.env.g1_mujuco_env_cfg import G1MujocoEnvCfg

    mujoco_env = MujocoEnv(cfg_env=G1MujocoEnvCfg())
    mujoco_env.viewer._paused = False

    while True:
        # mujoco_env.update()
        mujoco_env.step(np.zeros(mujoco_env.num_dofs))
        time.sleep(0.02)
