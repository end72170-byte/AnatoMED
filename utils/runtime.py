import contextlib
from pathlib import Path
import shutil
import warnings

import torch


def _configure_warnings(suppress: bool = True) -> None:
    if not suppress:
        return
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message=r"`torch\.cuda\.amp\.autocast\(args\.\.\.\)` is deprecated",
    )
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message=r"`torch\.cuda\.amp\.GradScaler\(args\.\.\.\)` is deprecated",
    )
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message=r"`torch_dtype` is deprecated",
    )


@contextlib.contextmanager
def autocast_ctx(device: torch.device):
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.amp.autocast(device_type="cuda", dtype=dtype):
            yield
    else:
        yield


def _check_free_space(path: Path, need_gb: float, what: str) -> None:
    usage = shutil.disk_usage(str(path))
    free_gb = usage.free / (1024**3)
    if free_gb < need_gb:
        raise RuntimeError(
            f"Not enough disk space to save {what}: need ~{need_gb:.1f}GB free, "
            f"but only {free_gb:.2f}GB available under {path}. "
            "Delete/move old checkpoints or change --out_dir to a bigger filesystem."
        )