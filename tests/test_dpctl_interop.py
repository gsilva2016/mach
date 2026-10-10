"""Tests for zero-copy interop with dpnp on the SYCL backend.

USM memory is only valid inside the SYCL context that allocated it, so mach
joins dpnp's context at import time rather than creating one of its own. These
tests check that dpnp arrays really are beamformed in place, and that the result
matches the NumPy path.
"""

import warnings

import numpy as np
import pytest

from mach import _backend, kernel

# Exercises whichever backend is installed, so no NVIDIA GPU is implied
pytestmark = pytest.mark.no_cuda

N_RX = 16
N_SAMPLES = 256
N_FRAMES = 2
N_VOXELS = 64
PITCH_M = 0.3e-3
SAMPLING_FREQ_HZ = 40e6
SOUND_SPEED_M_S = 1540.0
MODULATION_FREQ_HZ = 5e6
F_NUMBER = 1.5

requires_dpnp = pytest.mark.skipif(
    not (_backend.BACKEND == "sycl" and _backend.SHARES_DPCTL_CONTEXT),
    reason="needs the SYCL backend with dpnp installed and its SYCL context shared",
)


def _make_inputs() -> dict:
    rng = np.random.default_rng(0)
    channel_data = np.empty((N_RX, N_SAMPLES, N_FRAMES), dtype=np.complex64)
    channel_data.real = rng.standard_normal((N_RX, N_SAMPLES, N_FRAMES), dtype=np.float32)
    channel_data.imag = rng.standard_normal((N_RX, N_SAMPLES, N_FRAMES), dtype=np.float32)

    rx_coords_m = np.zeros((N_RX, 3), dtype=np.float32)
    rx_coords_m[:, 0] = (np.arange(N_RX, dtype=np.float32) - (N_RX - 1) / 2) * PITCH_M

    depths_m = np.linspace(5e-3, 50e-3, N_VOXELS, dtype=np.float32)
    scan_coords_m = np.zeros((N_VOXELS, 3), dtype=np.float32)
    scan_coords_m[:, 2] = depths_m

    return {
        "channel_data": channel_data,
        "rx_coords_m": rx_coords_m,
        "scan_coords_m": scan_coords_m,
        "tx_wave_arrivals_s": (depths_m / SOUND_SPEED_M_S).astype(np.float32),
    }


def _beamform(inputs: dict, out=None):
    return kernel.beamform(
        inputs["channel_data"],
        inputs["rx_coords_m"],
        inputs["scan_coords_m"],
        inputs["tx_wave_arrivals_s"],
        out,
        rx_start_s=0.0,
        sampling_freq_hz=SAMPLING_FREQ_HZ,
        f_number=F_NUMBER,
        sound_speed_m_s=SOUND_SPEED_M_S,
        modulation_freq_hz=MODULATION_FREQ_HZ,
        tukey_alpha=0.5,
    )


def test_device_namespace_matches_backend():
    """The device-array library mach reports is the one for the active backend."""
    device_xp = _backend.device_namespace()
    if device_xp is None:
        pytest.skip("no device-array library installed")

    expected = "cupy" if _backend.BACKEND == "cuda" else "dpnp"
    assert device_xp.__name__ == expected


@requires_dpnp
def test_dpnp_arrays_are_not_staged():
    """dpnp input is used in place, so mach does not warn about host transfers."""
    import dpnp

    inputs = _make_inputs()
    device_inputs = {key: dpnp.asarray(value) for key, value in inputs.items()}

    with warnings.catch_warnings():
        # A staged array would raise here, since the staging warning mentions latency
        warnings.simplefilter("error", UserWarning)
        result = _beamform(device_inputs)

    assert result.shape == (N_VOXELS, N_FRAMES)
    assert np.isfinite(dpnp.asnumpy(result)).all()


@requires_dpnp
def test_dpnp_matches_numpy():
    """Beamforming dpnp arrays gives the same image as beamforming NumPy ones."""
    import dpnp

    inputs = _make_inputs()
    expected = np.asarray(_beamform(inputs))

    device_inputs = {key: dpnp.asarray(value) for key, value in inputs.items()}
    result = dpnp.asnumpy(_beamform(device_inputs))

    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)


@requires_dpnp
@pytest.mark.skipif(not kernel.SUPPORTS_COMPOUND, reason="backend has no compound entry point")
def test_dpnp_compound_matches_numpy():
    """The whole-sequence path also works on dpnp arrays."""
    import dpnp

    n_transmits = 3
    inputs = _make_inputs()
    channel_data = np.repeat(inputs["channel_data"][None], n_transmits, axis=0)
    tx_arrivals = np.repeat(inputs["tx_wave_arrivals_s"][None], n_transmits, axis=0)

    def run(xp_channel, xp_rx, xp_scan, xp_tx):
        return kernel.beamform_compound(
            xp_channel,
            xp_rx,
            xp_scan,
            xp_tx,
            rx_start_s=0.0,
            sampling_freq_hz=SAMPLING_FREQ_HZ,
            f_number=F_NUMBER,
            sound_speed_m_s=SOUND_SPEED_M_S,
            modulation_freq_hz=MODULATION_FREQ_HZ,
            tukey_alpha=0.5,
        )

    expected = np.asarray(run(channel_data, inputs["rx_coords_m"], inputs["scan_coords_m"], tx_arrivals))
    result = dpnp.asnumpy(
        run(
            dpnp.asarray(channel_data),
            dpnp.asarray(inputs["rx_coords_m"]),
            dpnp.asarray(inputs["scan_coords_m"]),
            dpnp.asarray(tx_arrivals),
        )
    )

    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)
