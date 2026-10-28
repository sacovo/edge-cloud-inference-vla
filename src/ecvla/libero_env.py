"""LIBERO environment setup with a project-local config.

LIBERO reads its paths from $LIBERO_CONFIG_PATH/config.yaml (default ~/.libero), which may point at another
project's install. We always write our own config pointing at this venv's bddl/init files and a shared assets dir.
"""

import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / ".libero"
ASSETS_DIR = Path(os.environ.get("ECVLA_LIBERO_ASSETS", Path.home() / ".cache/libero/assets"))


def _ensure_config() -> None:
    import importlib.util

    pkg = Path(importlib.util.find_spec("libero").submodule_search_locations[0]) / "libero"
    config = {
        "benchmark_root": str(pkg),
        "bddl_files": str(pkg / "bddl_files"),
        "init_states": str(pkg / "init_files"),
        "datasets": str(PROJECT_ROOT / "data/libero"),
        "assets": str(ASSETS_DIR),
    }
    CONFIG_DIR.mkdir(exist_ok=True)
    (CONFIG_DIR / "config.yaml").write_text(yaml.safe_dump(config))
    os.environ["LIBERO_CONFIG_PATH"] = str(CONFIG_DIR)


_ensure_config()
os.environ.setdefault("MUJOCO_GL", "egl")

from libero.libero import get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402


def bddl_path(dataset_bddl_name: str) -> str:
    """Map the bddl path stored in a LIBERO hdf5 ('libero/libero/bddl_files/<suite>/<task>.bddl') to this install."""
    rel = dataset_bddl_name.split("bddl_files/", 1)[1]
    return str(Path(get_libero_path("bddl_files")) / rel)


def make_env(bddl_file: str, image_size: int = 128) -> OffScreenRenderEnv:
    return OffScreenRenderEnv(bddl_file_name=bddl_file, camera_heights=image_size, camera_widths=image_size)
