"""Test the array API utilities."""

from array_api_compat import is_cupy_namespace, is_jax_namespace, is_numpy_namespace

from mach._array_api import Array, DLPackDevice


def test_array_protocol(xp):
    """Test that the Array protocol accurately describes supported array libraries."""
    arr = xp.array([[1, 2], [3, 4]])

    # Type checking - verify arr matches protocol
    assert isinstance(arr, Array)

    # Check device type
    device_type, _ = arr.__dlpack_device__()
    if is_numpy_namespace(xp):
        assert device_type == DLPackDevice.CPU
    elif is_jax_namespace(xp):
        import jax

        # arr lives on jax's default backend, so that is what decides the DLPack device.
        # Asking jax.devices("gpu") instead would raise on a jax[cpu] install.
        jax_platforms = {"cuda": DLPackDevice.CUDA, "gpu": DLPackDevice.CUDA, "oneapi": DLPackDevice.ONEAPI}
        assert device_type == jax_platforms.get(jax.default_backend(), DLPackDevice.CPU)
    elif is_cupy_namespace(xp):
        assert device_type == DLPackDevice.CUDA
