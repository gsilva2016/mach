"""
GPU-accelerated 3D ultrasound beamforming with GPU array support.

This package provides GPU-accelerated beamforming functionality for ultrasound data,
through either a CUDA backend (NVIDIA GPUs) or a SYCL/DPC++ backend (Intel GPUs and CPUs).
It supports input arrays from NumPy, CuPy, and JAX through the DLPack protocol.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mach")
except PackageNotFoundError:
    # Update via bump-my-version, not manually
    __version__ = "0.1.2"

# Import main modules to make them available as mach.module_name
from . import geometry, kernel, wavefront
from ._array_api import Array

# Import the main beamform function to the top level for convenience
from .kernel import __backend__, beamform

# Define what gets imported with "from mach import *"
__all__ = [
    "Array",
    "__backend__",
    "__version__",
    "beamform",
    "geometry",
    "kernel",
    "wavefront",
]
