"""EXP-0005 helpers shared by the mixture builder and the slow-only check, so training data and closed loop build
LIBERO observations the same way: rendered in our sim at IMG_SIZE, rotated 180° (openpi's LIBERO convention), state =
eef pos + axis-angle + gripper qpos."""

import math

import numpy as np

IMG = ("observation.images.image", "observation.images.wrist_image")
IMG_SIZE = 224  # pi0_fast's input resolution; other sources are resized with padding to it
LIBERO_CAMS = ("agentview_image", "robot0_eye_in_hand_image")


def quat2axisangle(quat):
    """openpi `examples/libero/main.py` `_quat2axisangle` (robosuite transform_utils); quat = (x, y, z, w)."""
    quat = np.array(quat, dtype=np.float64)
    quat[3] = min(max(quat[3], -1.0), 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def libero_state(obs) -> np.ndarray:
    return np.concatenate([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                           obs["robot0_gripper_qpos"]]).astype(np.float32)


def libero_images(obs) -> tuple[np.ndarray, np.ndarray]:
    """Both cameras, rotated 180° (the sim renders in the OpenGL convention), HWC uint8."""
    return tuple(np.ascontiguousarray(obs[c][::-1, ::-1]) for c in LIBERO_CAMS)


def resize_pad(img: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """HWC uint8 → size × size, aspect kept, black padding (as openpi's resize_with_pad)."""
    from PIL import Image

    h, w = img.shape[:2]
    if (h, w) == (size, size):
        return img
    s = size / max(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    out = np.zeros((size, size, 3), np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    out[top: top + nh, left: left + nw] = np.asarray(Image.fromarray(img).resize((nw, nh), Image.BILINEAR))
    return out
