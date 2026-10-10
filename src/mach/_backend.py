"""Selection of the compiled beamforming backend.

mach can be built with a CUDA backend (``mach._cuda_impl``, compiled by NVCC), a
SYCL backend (``mach._sycl_impl``, compiled by Intel oneAPI DPC++), or both.
Both expose the same ``beamform`` overloads and ``InterpolationType`` enum.

Set the ``MACH_BACKEND`` environment variable to ``"cuda"`` or ``"sycl"`` to pin
a specific backend; otherwise the first importable one in :data:`BACKEND_MODULES`
order is used.
"""

import os
from importlib import import_module
from types import ModuleType

from mach import _dpctl
from mach._array_api import DLPackDevice

#: Extension module implementing each backend, in the order they are tried.
BACKEND_MODULES: dict[str, str] = {
    "cuda": "mach._cuda_impl",
    "sycl": "mach._sycl_impl",
}

#: DLPack device that each backend can beamform on without a host round-trip.
BACKEND_DEVICES: dict[str, DLPackDevice] = {
    "cuda": DLPackDevice.CUDA,
    "sycl": DLPackDevice.ONEAPI,
}

ENV_VAR = "MACH_BACKEND"


def _load() -> tuple[str, ModuleType]:
    requested = os.environ.get(ENV_VAR, "").strip().lower()
    if requested and requested not in BACKEND_MODULES:
        raise ValueError(f"{ENV_VAR}={requested!r} is not a known backend. Choose one of {sorted(BACKEND_MODULES)}.")

    candidates = [requested] if requested else list(BACKEND_MODULES)
    failures: dict[str, str] = {}
    for name in candidates:
        try:
            # Broad, because a backend can also fail on an unusable driver or
            # device: in a dual-backend install the other one may still work.
            module = import_module(BACKEND_MODULES[name])
        except Exception as err:
            failures[name] = f"{type(err).__name__}: {err}"
        else:
            return name, module

    detail = "\n".join(f"  {name}: {message}" for name, message in failures.items())
    raise ImportError(
        f"No mach beamforming backend could be imported. Tried:\n{detail}\n"
        "Reinstall mach with a backend enabled, e.g.\n"
        "  pip install --no-build-isolation -ve . -Ccmake.define.MACH_BUILD_SYCL=ON"
    )


BACKEND, impl = _load()

#: DLPack device type that the active backend beamforms on.
BACKEND_DEVICE: DLPackDevice = BACKEND_DEVICES[BACKEND]

#: Array library that allocates on the active backend's device, in import order.
DEVICE_NAMESPACES: dict[str, tuple[str, ...]] = {
    "cuda": ("cupy",),
    "sycl": ("dpnp",),
}

# USM allocations are context-scoped, so sharing dpnp's context is what makes its
# arrays usable without a staging copy. No-op unless dpnp is installed.
#: Why mach is not sharing dpnp's SYCL context, or None when it is.
DPCTL_ERROR: str | None = _dpctl.adopt_context(impl) if BACKEND == "sycl" else "the active backend is not SYCL"

#: Whether dpnp arrays can be beamformed without a staging copy.
SHARES_DPCTL_CONTEXT: bool = DPCTL_ERROR is None


def device_namespace() -> ModuleType | None:
    """The array library that allocates on the active backend's device, if installed.

    CuPy for CUDA, dpnp for SYCL. Returns None when none of the candidates can be
    imported, or when mach is not sharing dpnp's SYCL context and so could not use
    its arrays anyway; :data:`DPCTL_ERROR` says which.
    """
    if BACKEND == "sycl" and not SHARES_DPCTL_CONTEXT:
        return None
    for name in DEVICE_NAMESPACES[BACKEND]:
        try:
            return import_module(name)
        except ImportError:
            continue
    return None


InterpolationType = impl.InterpolationType
beamform = impl.beamform
