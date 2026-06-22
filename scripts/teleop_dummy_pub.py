#!/usr/bin/env python3
"""Keyboard teleop dummy publisher.

Sends root/head/left/right target poses (28 float32, world frame, quat xyzw)
over UDP so the deploy pipeline can be driven from the keyboard.
"""
import math
import numpy as np
import socket
import struct
import time
import argparse
import sys
import termios
import tty
import select
import pathlib
import json

# Repo root (this file lives in <repo>/scripts/). Anchor outputs and imports to
# it so the script works no matter the current working directory.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MAGIC = b"G6D1"
# magic(4s) + seq(u32) + 28 float32 (4 bodies * (pos3+quat4))
PACK_FMT = "<4sI" + "f" * 28


class SignalSendLogger:
    """Append-only logger for sent robot command packets."""

    def __init__(self, mode: int, root_dir: str = str(REPO_ROOT / "logs/signal_send")):
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.dir = pathlib.Path(root_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / f"teleop_mode{mode}_{ts}.csv"
        self.f = self.path.open("w", buffering=1)
        header = ["timestamp", "seq", "mode"] + [f"v{i}" for i in range(28)]
        self.f.write(",".join(header) + "\n")

    def log(self, t_sec: float, seq: int, mode: int, cmd28):
        vals = ",".join(f"{float(v):.6f}" for v in cmd28)
        self.f.write(f"{t_sec:.6f},{int(seq)},{int(mode)},{vals}\n")

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass


def euler_to_quat(roll: float, pitch: float, yaw: float):
    """
    Convert Euler angles (roll, pitch, yaw) to quaternion (x, y, z, w).
    Angles are in radians. Uses the z-y'-x'' (yaw-pitch-roll) convention.
    """
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return (x, y, z, w)


def vr_to_target_quat(vr_quat):
    """
    Convert VR quaternion to target coordinate system.
    Rotation axes should match 1:1 (VR X->Robot X, VR Y->Robot Y, VR Z->Robot Z).
    Apply 180° rotations around Z and Y to match robot EE initial orientation,
    with axis correction and direction inversion.
    """
    qx, qy, qz, qw = vr_quat

    # Axis mapping with X and Z direction inverted (flip from previous)
    target_qx = -qy  # VR Y -> Target X (negated)
    target_qy = -qx  # VR X -> Target Y (negated)
    target_qz = qz   # VR Z -> Target Z
    target_qw = qw

    # Quaternion multiplication: q1 * q2
    # First: rotate 180° around Z axis: q_z180 = (0, 0, 1, 0)
    z180_x, z180_y, z180_z, z180_w = 0.0, 0.0, 1.0, 0.0

    temp_w = z180_w * target_qw - z180_x * target_qx - z180_y * target_qy - z180_z * target_qz
    temp_x = z180_w * target_qx + z180_x * target_qw + z180_y * target_qz - z180_z * target_qy
    temp_y = z180_w * target_qy - z180_x * target_qz + z180_y * target_qw + z180_z * target_qx
    temp_z = z180_w * target_qz + z180_x * target_qy - z180_y * target_qx + z180_z * target_qw

    # Second: rotate 180° around Y axis: q_y180 = (0, 1, 0, 0)
    y180_x, y180_y, y180_z, y180_w = 0.0, 1.0, 0.0, 0.0

    result_w = y180_w * temp_w - y180_x * temp_x - y180_y * temp_y - y180_z * temp_z
    result_x = y180_w * temp_x + y180_x * temp_w + y180_y * temp_z - y180_z * temp_y
    result_y = y180_w * temp_y - y180_x * temp_z + y180_y * temp_w + y180_z * temp_x
    result_z = y180_w * temp_z + y180_x * temp_y - y180_y * temp_x + y180_z * temp_w

    return np.array([result_x, result_y, result_z, result_w], dtype=float)


def euler_to_target_quat(roll: float, pitch: float, yaw: float):
    """
    Convert Euler angles (roll,pitch,yaw) to a quaternion in the target
    coordinate system by first building a quaternion in the source (Euler)
    frame and then applying the same axis/flip mapping we use for VR
    quaternions (vr_to_target_quat).
    """
    q = euler_to_quat(roll, pitch, yaw)
    return vr_to_target_quat(q)


class ILRecorder:
    """
    Records:
      - commands: seq + 28 floats (root/head/left/right pose) per step
      - rgb video frames (mp4)
    Writes per-episode folder with commands.npz + video_rgb.mp4 + manifest.json
    """
    def __init__(self, root_dir=str(REPO_ROOT / "dataset/episodes"), hz=30.0):
        self.hz = float(hz)
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.ep_dir = pathlib.Path(root_dir) / f"ep_{ts}"
        self.ep_dir.mkdir(parents=True, exist_ok=True)

        self._t = []
        self._seq = []
        self._cmd = []

        self._cap = None
        self._writer = None
        self._rgb_path = str(self.ep_dir / "video_rgb.mp4")
        self._w = None
        self._h = None

    def start_camera(self, cam_index_or_path=4, width=None, height=None, fps=None):
        import cv2
        fps = float(fps if fps is not None else self.hz)

        # prefer V4L2 on linux
        self._cap = cv2.VideoCapture(cam_index_or_path, cv2.CAP_V4L2)

        if width is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        if height is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        self._cap.set(cv2.CAP_PROP_FPS, float(fps))

        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open camera: {cam_index_or_path}")

        self._w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH) or (width or 640))
        self._h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or (height or 480))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(self._rgb_path, fourcc, fps, (self._w, self._h))
        if not self._writer.isOpened():
            raise RuntimeError("Failed to open video writer (mp4v).")

        return self._w, self._h, fps

    def step(self, t_sec: float, seq: int, cmd28):
        # cmd28: iterable of 28 floats
        self._t.append(float(t_sec))
        self._seq.append(int(seq))
        self._cmd.append(np.asarray(cmd28, dtype=np.float32).reshape(28))

    def write_frame(self):
        if self._cap is None or self._writer is None:
            return False
        ret, frame = self._cap.read()
        if ret:
            self._writer.write(frame)
        return bool(ret)

    def close(self, conventions=None):
        if self._cap is not None:
            self._cap.release()
        if self._writer is not None:
            self._writer.release()

        t = np.asarray(self._t, dtype=np.float64)
        seq = np.asarray(self._seq, dtype=np.int64)
        cmd = np.stack(self._cmd, axis=0) if len(self._cmd) > 0 else np.zeros((0, 28), np.float32)

        np.savez_compressed(str(self.ep_dir / "commands.npz"), t=t, seq=seq, cmd=cmd)

        manifest = {
            "episode_dir": str(self.ep_dir),
            "steps": int(cmd.shape[0]),
            "hz": self.hz,
            "commands": {
                "file": "commands.npz",
                "fields": {
                    "t": "float64 seconds",
                    "seq": "int64",
                    "cmd": "[T,28] float32 = root(7)+head(7)+left(7)+right(7)"
                },
                "frame": "WORLD",
                "quat_order": "xyzw",
            },
            "rgb": {
                "file": "video_rgb.mp4",
                "size": [int(self._h or 0), int(self._w or 0)],
                "fps": self.hz,
                "codec": "mp4v",
            },
            "conventions": conventions or {},
        }
        with open(self.ep_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        return manifest


class KeyboardController:
    """Non-blocking keyboard input handler for terminal."""

    def __init__(self):
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)

    def __enter__(self):
        tty.setraw(self.fd)
        return self

    def __exit__(self, *args):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)

    def get_key(self):
        """Return key pressed or None if no key pressed. Non-blocking."""
        if select.select([sys.stdin], [], [], 0)[0]:
            ch = sys.stdin.read(1)
            # Handle arrow keys (escape sequences)
            if ch == '\x1b':
                # Wait a bit for the rest of the escape sequence
                if select.select([sys.stdin], [], [], 0.05)[0]:
                    ch2 = sys.stdin.read(1)
                    if ch2 == '[':
                        if select.select([sys.stdin], [], [], 0.05)[0]:
                            ch3 = sys.stdin.read(1)
                            if ch3 == 'A': return 'UP'
                            elif ch3 == 'B': return 'DOWN'
                            elif ch3 == 'C': return 'RIGHT'
                            elif ch3 == 'D': return 'LEFT'
                        return None  # incomplete sequence, ignore
                    return None  # incomplete sequence, ignore
                return 'ESC'  # only ESC pressed alone (no following chars after timeout)
            return ch
        return None


def main():
    ap = argparse.ArgumentParser(description="Keyboard teleop dummy UDP publisher")
    ap.add_argument("--dst_ip", type=str, default="127.0.0.1")
    ap.add_argument("--dst_port", type=int, default=15000)
    ap.add_argument("--hz", type=float, default=30.0)
    ap.add_argument("-r", "--record", action="store_true",
                    help="Record camera video alongside the sent commands")
    args = ap.parse_args()
    print_every = max(1, int(args.hz))

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Fixed body offsets for head/left/right hands
    head_off_b = (0.0, 0.0, 0.77)
    lhand_off_b = (0.25,  0.18, 0.15)
    rhand_off_b = (0.25, -0.18, 0.15)

    print("[teleop_dummy_udp_sender] keyboard interactive")
    print(f"[teleop_dummy_udp_sender] sending to {args.dst_ip}:{args.dst_port} at {args.hz} Hz")
    print("[packet] bodies = root, head, left, right; each = pos(xyz) + quat(xyzw) in WORLD frame")
    send_logger = SignalSendLogger(mode=0)
    print(f"[log] signal send log -> {send_logger.path}")
    print()
    print("=== Keyboard Controls ===")
    print("  Hands:  I/K = X +/-,  J/L = Y (apart/together),  U/O = Z +/-")
    print("  Root speed:  W/S = forward/back +/-,  A/D = left/right +/-  (each press = 0.2 m/s)")
    print("  Root:   F/H = Z +/-,  Q/E = rotate left/right (10°),  SPACE = stop (zero speed)")
    print("  ESC or Ctrl+C: quit")
    print("=========================")
    print()

    seq = 0
    period = 1.0 / args.hz
    next_t = time.time()
    step = 0.01  # 1cm step size
    vel_step = 0.2  # root speed increment per key press (m/s)
    vel_limit = 2.0  # clamp accumulated root speed (m/s)
    rot_step = math.radians(10)  # 10 degrees in radians
    root_yaw = 0.0  # current root yaw angle
    root_vx = 0.0  # accumulated root speed along X (m/s)
    root_vy = 0.0  # accumulated root speed along Y (m/s)

    rec = None
    record_enabled = False
    if args.record:
        try:
            rec = ILRecorder(root_dir=str(REPO_ROOT / "dataset/episodes"), hz=args.hz)

            # RealSense color node is /dev/video4; keep index=4
            w, h, fps = rec.start_camera(cam_index_or_path=4, width=1280, height=720, fps=args.hz)

            record_enabled = True
            print(f"[record] episode dir: {rec.ep_dir}")
            print(f"[record] rgb -> {rec._rgb_path} ({w}x{h}@{fps}fps)")
            print(f"[record] cmd -> {rec.ep_dir/'commands.npz'}")
        except Exception as e:
            print(f"[record] Recording unavailable: {e}")
            rec = None
            record_enabled = False

    # Fixed poses: root at origin, head/left/right at configured offsets
    # Quaternion (1, 0, 0, 0) = 180° rotation around X axis, flipping Y and Z
    root_x, root_y, root_z = (0.0, 0.0, 0.79)
    rqx, rqy, rqz, rqw = (1.0, 0.0, 0.0, 0.0)

    hx, hy, hz_pos = head_off_b
    hqx, hqy, hqz, hqw = (1.0, 0.0, 0.0, 0.0)

    lx, ly, lz = list(lhand_off_b)
    lqx, lqy, lqz, lqw = (1.0, 0.0, 0.0, 0.0)

    rx, ry, rz = list(rhand_off_b)
    rrqx, rrqy, rrqz, rrqw = (1.0, 0.0, 0.0, 0.0)

    with KeyboardController() as kb:
        try:
            while True:
                now = time.time()
                if now < next_t:
                    time.sleep(max(0.0, min(next_t - now, 0.01)))

                # Check for keyboard input
                key = kb.get_key()
                if key:
                    if key == 'ESC' or key == '\x03':  # ESC or Ctrl+C
                        print("\r\n[quit] Exiting...\r\n")
                        break
                    # Hands control
                    elif key in ('i', 'I'):
                        lx += step
                        rx += step
                    elif key in ('k', 'K'):
                        lx -= step
                        rx -= step
                    elif key in ('j', 'J'):
                        ly += step   # left hand Y+
                        ry -= step   # right hand Y- (opposite direction)
                    elif key in ('l', 'L'):
                        ly -= step   # left hand Y-
                        ry += step   # right hand Y+ (opposite direction)
                    elif key in ('u', 'U'):
                        lz += step
                        rz += step
                    elif key in ('o', 'O'):
                        lz -= step
                        rz -= step
                    # Root locomotion: W/S/A/D change the *speed* (it persists
                    # after release); SPACE stops (zeroes the speed).
                    elif key in ('w', 'W'):
                        root_vx = max(-vel_limit, root_vx - vel_step)
                    elif key in ('s', 'S'):
                        root_vx = min(vel_limit, root_vx + vel_step)
                    elif key in ('a', 'A'):
                        root_vy = max(-vel_limit, root_vy - vel_step)
                    elif key in ('d', 'D'):
                        root_vy = min(vel_limit, root_vy + vel_step)
                    elif key == ' ':
                        root_vx = 0.0
                        root_vy = 0.0
                    elif key in ('f', 'F'):
                        root_z += step
                    elif key in ('h', 'H'):
                        root_z -= step
                    # Root rotation (yaw)
                    elif key in ('q', 'Q'):
                        root_yaw += rot_step  # counter-clockwise (left)
                    elif key in ('e', 'E'):
                        root_yaw -= rot_step  # clockwise (right)

                # Update root quaternion from yaw (convert via target mapping)
                rqx, rqy, rqz, rqw = euler_to_target_quat(0.0, 0.0, root_yaw)

                if now >= next_t:
                    next_t += period

                    # Integrate the accumulated root speed into the root position
                    # so a held/released speed keeps the robot moving.
                    root_x += root_vx * period
                    root_y += root_vy * period

                    # Build the command to send (28 floats)
                    root = (root_x, root_y, root_z, rqx, rqy, rqz, rqw)
                    head = (hx, hy, hz_pos, hqx, hqy, hqz, hqw)
                    left = (lx, ly, lz, lqx, lqy, lqz, lqw)
                    right = (rx, ry, rz, rrqx, rrqy, rrqz, rrqw)
                    floats = root + head + left + right  # 28 floats

                    # Record cmd + RGB at the same instant
                    if record_enabled and rec is not None:
                        rec.step(t_sec=now, seq=seq, cmd28=floats)
                        rec.write_frame()   # skip the frame on failure, cmd is still recorded

                    # Send packet
                    send_seq = seq
                    pkt = struct.pack(PACK_FMT, MAGIC, send_seq, *map(float, floats))
                    sock.sendto(pkt, (args.dst_ip, args.dst_port))
                    send_logger.log(t_sec=now, seq=send_seq, mode=0, cmd28=floats)
                    seq = (send_seq + 1) & 0xFFFFFFFF

                    # low-rate print (every second)
                    if seq % print_every == 0:
                        yaw_deg = math.degrees(root_yaw)
                        print(f"\r[seq={seq:6d}] root=({root_x:+.3f},{root_y:+.3f},{root_z:+.3f},yaw={yaw_deg:+.1f}°) vel=({root_vx:+.2f},{root_vy:+.2f}) left=({lx:+.3f},{ly:+.3f},{lz:+.3f}) right=({rx:+.3f},{ry:+.3f},{rz:+.3f})   ", end='', flush=True)
        finally:
            if record_enabled and rec is not None:
                conventions = {
                    "packet": "root/head/left/right each: pos(xyz)+quat(xyzw) in WORLD",
                    "quat_order": "xyzw",
                    "axes": "world frame (as sent)",
                    "hz": args.hz,
                }
                manifest = rec.close(conventions=conventions)
                print(f"\n[record] saved episode: {manifest['episode_dir']}")
            send_logger.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
