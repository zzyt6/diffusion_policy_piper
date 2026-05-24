"""
Run a trained image Diffusion Policy on an AgileX/Songling Piper arm.

The policy predicts absolute 6-DoF joint qpos in radians. This script executes
those targets with Piper JointCtrl using a higher-rate interpolated command
stream to reduce stop-and-go motion.
"""

from __future__ import annotations

import argparse
import json
import math
import select
import signal
import sys
import termios
import threading
import time
import tty
from collections import deque
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional

import cv2
import dill
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from piper_sdk import C_PiperInterface_V2

from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace


DEG_PER_RAD = 180.0 / math.pi
JOINT_LIMITS_RAD = np.asarray(
    [
        [-2.6179, 2.6179],
        [0.0, 3.14],
        [-2.967, 0.0],
        [-1.745, 1.745],
        [-1.22, 1.22],
        [-2.09439, 2.09439],
    ],
    dtype=np.float64,
)


class CameraWorker:
    def __init__(self, name: str, camera_id: int | str, width: int, height: int, fps: int):
        self.name = name
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self._lock = threading.Lock()
        self._latest: Optional[np.ndarray] = None
        self._timestamp = 0.0
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._cap = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._cap is not None:
            self._cap.release()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def latest(self) -> tuple[Optional[np.ndarray], float]:
        with self._lock:
            image = None if self._latest is None else self._latest.copy()
            return image, self._timestamp

    def _run(self) -> None:
        cap = open_video_capture(self.camera_id)
        self._cap = cap
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        if not cap.isOpened():
            raise RuntimeError(f"{self.name}: cannot open camera {self.camera_id}")

        while self._running:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if (rgb.shape[1] != self.width) or (rgb.shape[0] != self.height):
                rgb = cv2.resize(rgb, (self.width, self.height), interpolation=cv2.INTER_AREA)
            rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
            with self._lock:
                self._latest = rgb
                self._timestamp = time.time()

        cap.release()
        self._cap = None


class PiperRobot:
    def __init__(self, can_name: str, no_can_judge: bool):
        self.interface = C_PiperInterface_V2(can_name, judge_flag=not no_can_judge)
        self.connected = False

    def connect(self) -> None:
        self.interface.ConnectPort()
        self.connected = True

    def enable(self, timeout_s: float) -> bool:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if bool(self.interface.EnablePiper()):
                return True
            time.sleep(0.02)
        return False

    def read_qpos(self) -> np.ndarray:
        msg = self.interface.GetArmJointMsgs()
        joint_state = getattr(msg, "joint_state")
        return np.asarray(
            [
                sdk_joint_to_rad(int(getattr(joint_state, f"joint_{idx}")))
                for idx in range(1, 7)
            ],
            dtype=np.float32,
        )

    def read_eef_pose(self) -> np.ndarray:
        msg = self.interface.GetArmEndPoseMsgs()
        end_pose = getattr(msg, "end_pose")
        return np.asarray(
            [
                getattr(end_pose, "X_axis") / 1000.0,
                getattr(end_pose, "Y_axis") / 1000.0,
                getattr(end_pose, "Z_axis") / 1000.0,
                getattr(end_pose, "RX_axis") / 1000.0,
                getattr(end_pose, "RY_axis") / 1000.0,
                getattr(end_pose, "RZ_axis") / 1000.0,
            ],
            dtype=np.float32,
        )

    def send_joint_qpos(self, qpos_rad: np.ndarray, speed_percent: int) -> None:
        units = qpos_rad_to_sdk_units(qpos_rad)
        self.interface.MotionCtrl_2(0x01, 0x01, int(speed_percent), 0x00)
        self.interface.JointCtrl(*[int(x) for x in units])

    def standby(self) -> None:
        try:
            self.interface.MotionCtrl_2(0x00, 0x01, 0, 0x00)
        except Exception:
            pass

    def emergency_stop(self) -> None:
        self.interface.EmergencyStop(0x01)

    def disconnect(self) -> None:
        if self.connected:
            self.interface.DisconnectPort()
            self.connected = False


class KeyboardStopper:
    def __init__(self):
        self.enabled = False
        self.fd = None
        self.old_settings = None

    def __enter__(self):
        if sys.stdin.isatty():
            self.fd = sys.stdin.fileno()
            self.old_settings = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
            self.enabled = True
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.enabled and self.fd is not None and self.old_settings is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)

    def should_stop(self) -> bool:
        if not self.enabled:
            return False
        ready, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not ready:
            return False
        char = sys.stdin.read(1)
        return char in ("q", "Q", "\x1b")


def parse_camera_id(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return value


def open_video_capture(camera_id: int | str):
    attempts = [None, cv2.CAP_V4L2]
    last_cap = None
    for backend in attempts:
        cap = cv2.VideoCapture(camera_id) if backend is None else cv2.VideoCapture(camera_id, backend)
        last_cap = cap
        if cap.isOpened():
            return cap
        cap.release()
    return last_cap


def sdk_joint_to_rad(value: int) -> float:
    return (value / 1000.0) / DEG_PER_RAD


def qpos_rad_to_sdk_units(qpos_rad: np.ndarray) -> np.ndarray:
    qpos = np.asarray(qpos_rad, dtype=np.float64)
    if qpos.shape != (6,):
        raise ValueError(f"Expected qpos shape (6,), got {qpos.shape}")
    return np.round(qpos * 1000.0 * DEG_PER_RAD).astype(np.int32)


def clip_joint_limits(qpos_rad: np.ndarray) -> np.ndarray:
    return np.clip(qpos_rad, JOINT_LIMITS_RAD[:, 0], JOINT_LIMITS_RAD[:, 1])


def limit_delta(target: np.ndarray, reference: np.ndarray, max_delta: float) -> np.ndarray:
    if max_delta <= 0:
        return target
    return reference + np.clip(target - reference, -max_delta, max_delta)


def smoothstep(alpha: float) -> float:
    alpha = min(1.0, max(0.0, alpha))
    return alpha * alpha * (3.0 - 2.0 * alpha)


def load_policy(ckpt_path: Path, device: torch.device, num_inference_steps: int):
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=("optimizer",), include_keys=None)

    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    policy.num_inference_steps = int(num_inference_steps)
    policy.eval().to(device)
    return policy, cfg


def image_to_tensor_history(images: Iterable[np.ndarray]) -> np.ndarray:
    arr = np.stack(list(images), axis=0).astype(np.float32) / 255.0
    return np.moveaxis(arr, -1, 1)


def build_obs_dict(
    wrist_hist: deque,
    global_hist: deque,
    qpos_hist: deque,
    eef_hist: deque,
) -> Dict[str, torch.Tensor]:
    return {
        "camera_wrist": torch.from_numpy(image_to_tensor_history(wrist_hist)),
        "camera_global": torch.from_numpy(image_to_tensor_history(global_hist)),
        "robot_qpos": torch.from_numpy(np.stack(qpos_hist, axis=0).astype(np.float32)),
        "robot_eef_pose": torch.from_numpy(np.stack(eef_hist, axis=0).astype(np.float32)),
    }


def wait_for_cameras(cameras: list[CameraWorker], timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if all(camera.latest()[0] is not None for camera in cameras):
            return
        time.sleep(0.05)
    missing = [camera.name for camera in cameras if camera.latest()[0] is None]
    raise RuntimeError(f"Timed out waiting for cameras: {missing}")


def append_observation(
    wrist_cam: CameraWorker,
    global_cam: CameraWorker,
    robot: PiperRobot,
    wrist_hist: deque,
    global_hist: deque,
    qpos_hist: deque,
    eef_hist: deque,
) -> bool:
    wrist_img, _ = wrist_cam.latest()
    global_img, _ = global_cam.latest()
    if wrist_img is None or global_img is None:
        return False
    wrist_hist.append(wrist_img)
    global_hist.append(global_img)
    qpos_hist.append(robot.read_qpos())
    eef_hist.append(robot.read_eef_pose())
    return True


def reset_to_initial_pose(
    robot: PiperRobot,
    target_qpos: np.ndarray,
    speed_percent: int,
    duration_s: float,
    send_hz: float,
    enable_motion: bool,
) -> np.ndarray:
    start_qpos = robot.read_qpos().astype(np.float64)
    target_qpos = clip_joint_limits(target_qpos.astype(np.float64))
    dt = 1.0 / send_hz
    t0 = time.monotonic()
    next_t = t0
    last_cmd = start_qpos.copy()
    while True:
        now = time.monotonic()
        alpha = min(1.0, (now - t0) / max(duration_s, 0.1))
        qpos = start_qpos + (target_qpos - start_qpos) * smoothstep(alpha)
        qpos = clip_joint_limits(qpos)
        if enable_motion:
            robot.send_joint_qpos(qpos, speed_percent)
        last_cmd = qpos
        if alpha >= 1.0:
            break
        next_t += dt
        precise_wait(next_t)
    return last_cmd


def interpolate_and_send(
    robot: PiperRobot,
    waypoints: np.ndarray,
    last_command: np.ndarray,
    frequency: float,
    send_hz: float,
    speed_percent: int,
    max_delta_per_send: float,
    max_delta_per_policy_step: float,
    enable_motion: bool,
    after_policy_step: Optional[Callable[[], None]] = None,
) -> np.ndarray:
    waypoints = np.asarray(waypoints, dtype=np.float64)
    if waypoints.ndim != 2 or waypoints.shape[1] != 6:
        raise ValueError(f"Expected waypoints shape [T,6], got {waypoints.shape}")

    dt_send = 1.0 / send_hz
    sends_per_policy_step = max(1, int(round(send_hz / frequency)))
    current = last_command.astype(np.float64)
    next_t = time.monotonic()

    for raw_waypoint in waypoints:
        target = clip_joint_limits(raw_waypoint)
        target = limit_delta(target, current, max_delta_per_policy_step)
        for idx in range(1, sends_per_policy_step + 1):
            alpha = idx / sends_per_policy_step
            qpos = current * (1.0 - alpha) + target * alpha
            qpos = limit_delta(qpos, last_command, max_delta_per_send)
            qpos = clip_joint_limits(qpos)
            if enable_motion:
                robot.send_joint_qpos(qpos, speed_percent)
            last_command = qpos
            next_t += dt_send
            precise_wait(next_t)
        current = last_command
        if after_policy_step is not None:
            after_policy_step()
    return last_command


def make_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Piper real-robot Diffusion Policy inference.")
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--initial-pose-json", type=Path, required=True)
    parser.add_argument("--can", default="can0")
    parser.add_argument("--camera-wrist", default="10")
    parser.add_argument("--camera-global", default="4")
    parser.add_argument("--camera-width", type=int, default=320)
    parser.add_argument("--camera-height", type=int, default=240)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--frequency", type=float, default=10.0)
    parser.add_argument("--send-hz", type=float, default=50.0)
    parser.add_argument("--steps-per-inference", type=int, default=2)
    parser.add_argument("--speed-percent", type=int, default=10)
    parser.add_argument("--reset-duration", type=float, default=5.0)
    parser.add_argument("--max-duration", type=float, default=60.0)
    parser.add_argument("--num-inference-steps", type=int, default=16)
    parser.add_argument("--max-joint-delta-rad-per-send", type=float, default=0.01)
    parser.add_argument("--max-joint-delta-rad-per-policy-step", type=float, default=0.04)
    parser.add_argument("--enable-timeout", type=float, default=5.0)
    parser.add_argument("--camera-timeout", type=float, default=5.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-can-judge", action="store_true")
    parser.add_argument("--enable-motion", action="store_true")
    parser.add_argument("--emergency-stop-on-exit", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.frequency <= 0:
        raise ValueError("--frequency must be > 0")
    if args.send_hz <= 0:
        raise ValueError("--send-hz must be > 0")
    if args.send_hz < args.frequency:
        raise ValueError("--send-hz must be >= --frequency")
    if args.steps_per_inference <= 0:
        raise ValueError("--steps-per-inference must be > 0")
    if not args.ckpt.is_file():
        raise FileNotFoundError(args.ckpt)
    if not args.initial_pose_json.is_file():
        raise FileNotFoundError(args.initial_pose_json)


def main() -> None:
    parser = make_arg_parser()
    args = parser.parse_args()
    validate_args(args)

    stop_event = threading.Event()

    def _request_stop(signum, frame):
        stop_event.set()

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    cv2.setNumThreads(1)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Loading policy from {args.ckpt}")
    policy, cfg = load_policy(args.ckpt, device=device, num_inference_steps=args.num_inference_steps)
    n_obs_steps = int(cfg.n_obs_steps)
    action_dim = int(cfg.shape_meta.action.shape[0])
    if action_dim != 6:
        raise ValueError(f"Expected action_dim=6, got {action_dim}")
    print(f"Policy ready on {device}; n_obs_steps={n_obs_steps}")

    initial_pose = json.loads(args.initial_pose_json.read_text())
    reset = initial_pose.get("reset", {})
    reset_qpos = reset.get("joints_rad") or initial_pose.get("joints_rad")
    if reset_qpos is None:
        units = reset.get("piper_sdk_joint_units") or initial_pose.get("piper_sdk_joint_units")
        if units is None:
            raise ValueError("Initial pose JSON must contain joints_rad or piper_sdk_joint_units")
        reset_qpos = np.asarray(units, dtype=np.float64) / (1000.0 * DEG_PER_RAD)
    reset_qpos = np.asarray(reset_qpos, dtype=np.float64)
    if reset_qpos.shape != (6,):
        raise ValueError(f"Reset qpos must have shape (6,), got {reset_qpos.shape}")

    wrist_cam = CameraWorker(
        "wrist",
        parse_camera_id(args.camera_wrist),
        width=args.camera_width,
        height=args.camera_height,
        fps=args.camera_fps,
    )
    global_cam = CameraWorker(
        "global",
        parse_camera_id(args.camera_global),
        width=args.camera_width,
        height=args.camera_height,
        fps=args.camera_fps,
    )
    robot = PiperRobot(args.can, no_can_judge=args.no_can_judge)

    last_command = None
    try:
        wrist_cam.start()
        global_cam.start()
        wait_for_cameras([wrist_cam, global_cam], timeout_s=args.camera_timeout)
        print("Cameras ready.")

        robot.connect()
        print("Piper connected.")
        if args.enable_motion:
            if not robot.enable(args.enable_timeout):
                raise RuntimeError("EnablePiper timed out")
            print("Piper enabled.")
        else:
            print("Dry run: --enable-motion not set; no JointCtrl commands will be sent.")

        last_command = reset_to_initial_pose(
            robot=robot,
            target_qpos=reset_qpos,
            speed_percent=args.speed_percent,
            duration_s=args.reset_duration,
            send_hz=args.send_hz,
            enable_motion=args.enable_motion,
        )
        print(f"Reset complete. qpos={np.round(last_command, 4)}")

        wrist_hist = deque(maxlen=n_obs_steps)
        global_hist = deque(maxlen=n_obs_steps)
        qpos_hist = deque(maxlen=n_obs_steps)
        eef_hist = deque(maxlen=n_obs_steps)

        t_start = time.monotonic()
        next_policy_t = t_start
        with KeyboardStopper() as keyboard:
            while not stop_event.is_set():
                if keyboard.should_stop():
                    print("Keyboard stop requested.")
                    break
                if time.monotonic() - t_start > args.max_duration:
                    print("Max duration reached.")
                    break

                if not append_observation(
                    wrist_cam=wrist_cam,
                    global_cam=global_cam,
                    robot=robot,
                    wrist_hist=wrist_hist,
                    global_hist=global_hist,
                    qpos_hist=qpos_hist,
                    eef_hist=eef_hist,
                ):
                    time.sleep(0.01)
                    continue

                if len(qpos_hist) < n_obs_steps:
                    time.sleep(1.0 / args.frequency)
                    continue

                obs_np = build_obs_dict(wrist_hist, global_hist, qpos_hist, eef_hist)
                obs = dict_apply(obs_np, lambda x: x.unsqueeze(0).to(device))
                with torch.no_grad():
                    result = policy.predict_action(obs)
                    action = result["action"][0].detach().cpu().numpy()
                if action.shape[-1] != 6:
                    raise ValueError(f"Policy returned action shape {action.shape}, expected [...,6]")
                action = action[: args.steps_per_inference]
                if len(action) == 0:
                    raise RuntimeError("Policy returned no executable actions")

                last_command = interpolate_and_send(
                    robot=robot,
                    waypoints=action,
                    last_command=last_command,
                    frequency=args.frequency,
                    send_hz=args.send_hz,
                    speed_percent=args.speed_percent,
                    max_delta_per_send=args.max_joint_delta_rad_per_send,
                    max_delta_per_policy_step=args.max_joint_delta_rad_per_policy_step,
                    enable_motion=args.enable_motion,
                    after_policy_step=lambda: append_observation(
                        wrist_cam=wrist_cam,
                        global_cam=global_cam,
                        robot=robot,
                        wrist_hist=wrist_hist,
                        global_hist=global_hist,
                        qpos_hist=qpos_hist,
                        eef_hist=eef_hist,
                    ),
                )
                print(
                    "policy_action",
                    np.round(action[-1], 4),
                    "cmd",
                    np.round(last_command, 4),
                    "sdk",
                    qpos_rad_to_sdk_units(last_command).tolist(),
                )

                next_policy_t += args.steps_per_inference / args.frequency
                precise_wait(next_policy_t)

    finally:
        if args.emergency_stop_on_exit and args.enable_motion:
            print("Emergency stop on exit.")
            robot.emergency_stop()
        elif args.enable_motion:
            print("Entering standby.")
            robot.standby()
        robot.disconnect()
        wrist_cam.stop()
        global_cam.stop()
        cv2.destroyAllWindows()
        print("Shutdown complete.")


if __name__ == "__main__":
    main()
