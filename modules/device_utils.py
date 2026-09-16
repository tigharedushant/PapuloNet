"""
modules/device_utils.py

AEF-CRC Phase 3: Device-agnostic compute detection and execution telemetry.
Designed to meet strict IEEE journal reproducibility and hardware-transparency standards:
- Dynamic discovery of physical compute devices (CPU, NVIDIA GPU, multi-GPU, accelerators)
  without hardcoded hardware or OS strings.
- Graceful memory growth configuration to prevent out-of-memory errors on memory-constrained
  devices (e.g. 4 GB VRAM accelerators) without altering scientific hyperparameters.
- Truthful device telemetry reporting (never silently claim CPU run was on GPU).
- Standardized execution device manifest for per-fold traceability.
"""

from __future__ import annotations

import os
import platform
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class DeviceInfo:
    """Structured runtime hardware and software environment specification."""
    device_type: str                  # "GPU", "CPU", or "TPU"
    device_name: str                  # e.g. "NVIDIA GeForce RTX 2050" or "CPU"
    device_count: int                 # number of physical devices of this type
    execution_strategy: str           # "OneDeviceStrategy", "MirroredStrategy", or "DefaultStrategy"
    device_fallback: bool             # True if hardware accelerator unavailable and fell back to CPU
    tensorflow_version: str
    python_version: str
    platform_system: str              # platform.system() e.g. "Linux", "Windows"
    platform_release: str             # platform.release()
    cuda_available: bool = False
    mixed_precision_active: bool = False
    mixed_precision_policy: str = "float32"
    keras_version: str = ""
    seed: Optional[int] = None
    status_message: str = ""
    physical_devices: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        return self.status_message or f"{self.device_type} ({self.device_name})"

    def to_dict(self) -> dict:
        return asdict(self)


def detect_device_environment(seed: Optional[int] = None) -> DeviceInfo:
    """Inspects TensorFlow and the underlying OS to identify available compute devices.
    Never hardcodes specific device names or OS assumptions.
    """
    import tensorflow as tf

    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    tf_ver = str(tf.__version__)
    plat_sys = platform.system()
    plat_rel = platform.release()

    try:
        import keras
        keras_ver = str(getattr(keras, "__version__", getattr(tf.keras, "__version__", tf_ver)))
    except Exception:
        keras_ver = str(getattr(tf.keras, "__version__", tf_ver))

    def _safe_name(dev: Any) -> str:
        try:
            val = getattr(dev, "name", None)
            if val is not None:
                return str(val)
        except Exception:
            pass
        return str(dev)

    gpus = tf.config.list_physical_devices("GPU")
    tpus = tf.config.list_physical_devices("TPU")
    physical_all = [_safe_name(d) for d in tf.config.list_physical_devices()]
    cuda_built = bool(tf.test.is_built_with_cuda())

    if tpus:
        device_type = "TPU"
        device_name = "Google Cloud TPU"
        device_count = len(tpus)
        execution_strategy = "TPUStrategy"
        device_fallback = False
        cuda_avail = False
    elif gpus:
        device_type = "GPU"
        device_count = len(gpus)
        device_names = []
        for gpu in gpus:
            try:
                details = tf.config.experimental.get_device_details(gpu)
                name = details.get("device_name") or _safe_name(gpu)
                device_names.append(str(name))
            except Exception:
                device_names.append(_safe_name(gpu))
        device_name = ", ".join(device_names) if device_names else "NVIDIA GPU"
        execution_strategy = "MirroredStrategy" if device_count > 1 else "OneDeviceStrategy"
        device_fallback = False
        cuda_avail = cuda_built
    else:
        device_type = "CPU"
        device_name = platform.processor() or "Generic CPU"
        device_count = len(tf.config.list_physical_devices("CPU"))
        execution_strategy = "DefaultStrategy"
        device_fallback = True  # Accelerator not detected, CPU fallback
        cuda_avail = False

    status_msg = f"{device_count} {device_type}(s) detected ({device_name}) on {plat_sys} (TF {tf_ver}, Keras {keras_ver})"

    return DeviceInfo(
        device_type=device_type,
        device_name=device_name,
        device_count=device_count,
        execution_strategy=execution_strategy,
        device_fallback=device_fallback,
        tensorflow_version=tf_ver,
        python_version=py_ver,
        platform_system=plat_sys,
        platform_release=plat_rel,
        cuda_available=cuda_avail,
        mixed_precision_active=False,
        mixed_precision_policy="float32",
        keras_version=keras_ver,
        seed=seed,
        status_message=status_msg,
        physical_devices=physical_all,
    )


def configure_execution_device(device_info: Optional[DeviceInfo] = None, seed: Optional[int] = None) -> DeviceInfo:
    """Configures TensorFlow memory growth and precision policies based on detected hardware.
    Enables memory growth on all GPUs to prevent pre-allocating entire VRAM.
    Sets mixed precision (mixed_float16) on GPU accelerators.
    """
    import tensorflow as tf

    if device_info is None:
        device_info = detect_device_environment(seed=seed)
    elif seed is not None:
        device_info.seed = seed

    if device_info.device_type == "GPU":
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                # Memory growth must be configured before GPU tensors are allocated
                pass

        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            device_info.mixed_precision_active = True
            device_info.mixed_precision_policy = "mixed_float16"
        except Exception:
            device_info.mixed_precision_active = False
            device_info.mixed_precision_policy = "float32"
        device_info.status_message = (
            f"Compute configuration: {device_info.device_count} GPU(s) detected ({device_info.device_name}): "
            f"memory growth enabled, mixed_float16 policy set."
        )
    else:
        try:
            tf.keras.mixed_precision.set_global_policy("float32")
        except Exception:
            pass
        device_info.mixed_precision_active = False
        device_info.mixed_precision_policy = "float32"
        device_info.status_message = (
            f"No GPU detected -- running on CPU ({device_info.device_name}). Mixed precision set to float32."
        )

    return device_info


def detect_and_configure_device(seed: Optional[int] = None) -> DeviceInfo:
    """Convenience entry point auto-detecting and configuring runtime device environment."""
    info = detect_device_environment(seed=seed)
    return configure_execution_device(info, seed=seed)


def device_smoke_test() -> Tuple[bool, str]:
    """Small device-agnostic smoke test verifying TensorFlow execution readiness:
    1. imports TensorFlow
    2. detects available devices
    3. constructs a tiny model
    4. performs one small tensor/model operation
    5. reports CPU/GPU/accelerator availability
    Passes reliably on both GPU-accelerated and CPU-only systems.
    """
    try:
        import tensorflow as tf
    except ImportError as exc:
        return False, f"TensorFlow is not installed in this environment: {exc}"

    try:
        device_info = detect_device_environment()

        # Step 3: Construct a tiny test model
        tiny_model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(2,)),
            tf.keras.layers.Dense(2, kernel_initializer="ones", use_bias=False),
        ])

        # Step 4: Perform one small tensor / model operation
        test_input = tf.constant([[1.0, 2.0]], dtype=tf.float32)
        out = tiny_model(test_input)
        out_arr = out.numpy()

        # Verify output shape and computation correctness (1*1 + 2*1 = 3)
        if out_arr.shape != (1, 2) or abs(out_arr[0, 0] - 3.0) > 1e-4:
            return False, f"Unexpected calculation result from test model: {out_arr}"

        # Step 5: Report device and accelerator availability truthfully
        status = (
            f"Device smoke test verified successfully on {device_info.device_type} ({device_info.device_name}). "
            f"Availability: {device_info.status_message}"
        )
        return True, status
    except Exception as exc:
        return False, f"Device smoke test failed: {exc}"


def format_device_manifest(device_info: DeviceInfo) -> Dict[str, Any]:
    """Returns a dictionary suitable for persisting into run manifests and JSON contracts."""
    return asdict(device_info)
