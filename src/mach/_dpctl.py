"""Interop between the SYCL backend and dpctl/dpnp.

dpnp is Intel's NumPy-like array library and dpctl is the device-management layer
underneath it, the rough equivalent of CuPy's two halves. dpnp allocates USM
memory, which is only meaningful inside the SYCL context it was allocated in, so
mach has to run in that context for dpnp arrays to be usable without a copy.

:func:`adopt_context` arranges that at import time. It is a no-op when dpnp is
not installed, in which case dpnp arrays are simply never seen.

Two version notes:

* Tensor functionality used to live in ``dpctl.tensor``, which was deprecated in
  dpctl 0.21 and removed in 0.22 in favour of dpnp. Only dpnp is used here.
* Joining the context means handing dpctl a raw ``sycl::queue`` pointer, which is
  only meaningful if mach and dpnp are using the same SYCL runtime. dpnp >= 0.20
  requires ``dpcpp-cpp-rt >= 2026.0``, so mach has to be built with a 2026 oneAPI
  compiler. :func:`_runtime_mismatch` checks that and refuses to hand the pointer
  over otherwise, turning a likely crash into a clean fallback to NumPy staging.
"""

import os
import re
from importlib.metadata import PackageNotFoundError, version
from types import ModuleType
from typing import Any

#: Shared with the device selector in kernel_sycl.cpp.
DEVICE_ENV_VAR = "MACH_SYCL_DEVICE"

#: Set to 1 to hand over the queue pointer even when the runtimes look mismatched.
OVERRIDE_ENV_VAR = "MACH_ALLOW_SYCL_RUNTIME_MISMATCH"

#: MACH_SYCL_DEVICE value -> dpctl filter selector string (None means dpnp's default).
_DEVICE_FILTERS: dict[str, str | None] = {"gpu": "gpu", "cpu": "cpu", "default": None}

_SYCL_LIBRARY = re.compile(r"^libsycl\.so")


def _requested_filter() -> str | None:
    """The dpctl filter selector matching MACH_SYCL_DEVICE."""
    requested = os.environ.get(DEVICE_ENV_VAR, "").strip().lower()
    if not requested:
        return None
    if requested not in _DEVICE_FILTERS:
        raise ValueError(f"{DEVICE_ENV_VAR}={requested!r} is not valid. Choose one of {sorted(_DEVICE_FILTERS)}.")
    return _DEVICE_FILTERS[requested]


def _probe_queue(dpnp: ModuleType) -> Any:
    """The queue dpnp allocates on for the device mach was asked for.

    Read off a real allocation rather than built from a device, so that the
    context is guaranteed to be the one dpnp arrays will actually carry, and so
    that dpnp's SYCL runtime is loaded before :func:`_runtime_mismatch` looks.
    """
    probe = dpnp.empty(1, dtype="uint8", device=_requested_filter())
    return probe.sycl_queue


def _loaded_sycl_runtimes() -> set[str]:
    """Distinct libsycl images mapped into this process.

    More than one means mach and dpnp each brought their own SYCL runtime, which
    is the case the queue handoff cannot survive. Returns an empty set where
    there is no procfs to read, i.e. everywhere except Linux.
    """
    try:
        with open("/proc/self/maps", encoding="utf-8", errors="replace") as maps:
            lines = maps.read().splitlines()
    except OSError:
        return set()

    paths = set()
    for line in lines:
        fields = line.split()
        # Mapped files are the sixth field; anonymous mappings have only five
        if len(fields) < 6:
            continue
        path = fields[-1]
        if path.startswith("/") and _SYCL_LIBRARY.match(os.path.basename(path)):
            paths.add(os.path.realpath(path))
    return paths


def _runtime_mismatch(impl: ModuleType) -> str | None:
    """Why mach's SYCL runtime and dpnp's cannot be assumed to be the same, if so."""
    runtimes = _loaded_sycl_runtimes()
    if len(runtimes) > 1:
        return (
            "mach and dpnp loaded different SYCL runtimes (" + ", ".join(sorted(runtimes)) + "), so passing "
            "a sycl::queue between them is unsafe. Build mach with the oneAPI release that dpnp's "
            "dpcpp-cpp-rt provides, or install dpnp from the channel matching your compiler."
        )

    # Portable fallback for platforms without procfs, and a clearer message when
    # the runtime came from a pip wheel: compare oneAPI release years.
    try:
        runtime_version = version("dpcpp-cpp-rt")
    except PackageNotFoundError:
        # The runtime came from setvars.sh rather than pip, so there is no
        # version to compare against; the procfs check above is the real guard.
        return None

    built_version = getattr(impl, "__sycl_version__", "") or ""
    built_year = built_version.partition(".")[0]
    runtime_year = runtime_version.partition(".")[0]
    if built_year and runtime_year and built_year != runtime_year:
        return (
            f"mach was built with oneAPI {built_version} but dpnp uses dpcpp-cpp-rt {runtime_version}. "
            "Passing a sycl::queue between two oneAPI releases is unsafe. Rebuild mach with a "
            f"{runtime_year} compiler ('source /opt/intel/oneapi/setvars.sh')."
        )
    return None


def adopt_context(impl: ModuleType) -> str | None:
    """Move the SYCL backend onto dpnp's SYCL context.

    Args:
        impl: The compiled backend module, i.e. ``mach._sycl_impl``.

    Returns:
        None once mach is sharing dpnp's context, otherwise a short explanation
        of why it is not. Failing here is not fatal: it only means dpnp arrays
        will be rejected, and NumPy input still works.

    Note:
        Deliberately returns the reason rather than warning. This runs during
        ``import mach``, and a warning raised there becomes a hard error for
        anyone who has turned warnings into errors.
    """
    adopt = getattr(impl, "adopt_sycl_queue", None)
    if adopt is None:
        return "this build of the backend has no adopt_sycl_queue()"

    try:
        import dpnp
    except ImportError as err:
        # Reports the underlying message rather than just "not installed": an
        # installed dpnp that cannot find the oneAPI runtime fails the same way.
        return f"dpnp could not be imported ({err})"

    try:
        # Broad: this can fail on a missing device, a dpnp built against a
        # different oneAPI runtime, or an API change. None of those should stop
        # mach from loading, since the NumPy path does not need dpnp.
        queue = _probe_queue(dpnp)
    except Exception as err:
        return f"{type(err).__name__}: {err}"

    # Checked after the probe, which forces dpnp's runtime to load, and before
    # the handoff, which is the only step that crosses between the two runtimes.
    if os.environ.get(OVERRIDE_ENV_VAR, "").strip() not in {"1", "true", "yes"}:
        mismatch = _runtime_mismatch(impl)
        if mismatch is not None:
            return f"{mismatch} Set {OVERRIDE_ENV_VAR}=1 to try anyway."

    try:
        adopt(queue.addressof_ref())
    except Exception as err:
        return f"{type(err).__name__}: {err}"

    return None
