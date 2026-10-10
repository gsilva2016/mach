"""Tests for the whole-sequence compounding entry point.

`mach.kernel.beamform_compound` keeps the accumulator in device memory across
transmits instead of returning to the host between them, so these tests pin it to
the per-transmit loop it replaces.
"""

import numpy as np
import pytest

import mach.experimental
from mach import kernel

# Exercises whichever backend is installed, so no NVIDIA GPU is implied
pytestmark = pytest.mark.no_cuda

N_TRANSMITS = 5
N_RX = 16
N_SAMPLES = 256
N_FRAMES = 2
N_VOXELS = 64
PITCH_M = 0.3e-3
SAMPLING_FREQ_HZ = 40e6
SOUND_SPEED_M_S = 1540.0
MODULATION_FREQ_HZ = 5e6
F_NUMBER = 1.5
TUKEY_ALPHA = 0.5


def _make_sequence(dtype: np.dtype) -> dict:
    """Synthetic multi-transmit dataset for a linear probe."""
    rng = np.random.default_rng(0)
    shape = (N_TRANSMITS, N_RX, N_SAMPLES, N_FRAMES)

    if np.issubdtype(dtype, np.complexfloating):
        # Fill in place to avoid a complex128 intermediate
        channel_data = np.empty(shape, dtype=dtype)
        channel_data.real = rng.standard_normal(shape, dtype=np.float32)
        channel_data.imag = rng.standard_normal(shape, dtype=np.float32)
    else:
        channel_data = rng.standard_normal(shape, dtype=np.float32)

    rx_coords_m = np.zeros((N_RX, 3), dtype=np.float32)
    rx_coords_m[:, 0] = (np.arange(N_RX, dtype=np.float32) - (N_RX - 1) / 2) * PITCH_M

    depths_m = np.linspace(5e-3, 50e-3, N_VOXELS, dtype=np.float32)
    scan_coords_m = np.zeros((N_VOXELS, 3), dtype=np.float32)
    scan_coords_m[:, 2] = depths_m

    # A different steering angle per transmit, so the transmits are not interchangeable
    angles_rad = np.linspace(-0.2, 0.2, N_TRANSMITS, dtype=np.float32)
    tx_wave_arrivals_s = np.outer(np.cos(angles_rad), depths_m).astype(np.float32) / SOUND_SPEED_M_S

    return {
        "channel_data": channel_data,
        "rx_coords_m": rx_coords_m,
        "scan_coords_m": scan_coords_m,
        "tx_wave_arrivals_s": tx_wave_arrivals_s,
    }


def _beamform_per_transmit(data: dict) -> np.ndarray:
    """The loop `beamform_compound` replaces: one call per transmit, summed on the host."""
    total = None
    for transmit_idx in range(N_TRANSMITS):
        single = kernel.beamform(
            data["channel_data"][transmit_idx],
            data["rx_coords_m"],
            data["scan_coords_m"],
            data["tx_wave_arrivals_s"][transmit_idx],
            rx_start_s=0.0,
            sampling_freq_hz=SAMPLING_FREQ_HZ,
            f_number=F_NUMBER,
            sound_speed_m_s=SOUND_SPEED_M_S,
            modulation_freq_hz=MODULATION_FREQ_HZ,
            tukey_alpha=TUKEY_ALPHA,
        )
        total = np.asarray(single) if total is None else total + np.asarray(single)
    assert total is not None
    return total


@pytest.mark.skipif(not kernel.SUPPORTS_COMPOUND, reason="backend has no compound entry point")
@pytest.mark.parametrize("dtype", [np.complex64, np.float32], ids=["iq", "rf"])
def test_compound_matches_per_transmit_loop(dtype):
    """Compounding on the device gives the same image as summing on the host."""
    data = _make_sequence(np.dtype(dtype))
    expected = _beamform_per_transmit(data)

    result = kernel.beamform_compound(
        data["channel_data"],
        data["rx_coords_m"],
        data["scan_coords_m"],
        data["tx_wave_arrivals_s"],
        rx_start_s=0.0,
        sampling_freq_hz=SAMPLING_FREQ_HZ,
        f_number=F_NUMBER,
        sound_speed_m_s=SOUND_SPEED_M_S,
        modulation_freq_hz=MODULATION_FREQ_HZ,
        tukey_alpha=TUKEY_ALPHA,
    )

    assert result.shape == (N_VOXELS, N_FRAMES)
    assert result.dtype == dtype
    # Looser than exact: the two paths sum the transmits in a different order
    np.testing.assert_allclose(np.asarray(result), expected, rtol=1e-4, atol=1e-5)


@pytest.mark.skipif(not kernel.SUPPORTS_COMPOUND, reason="backend has no compound entry point")
def test_compound_writes_into_provided_out():
    """A host `out` array is overwritten, not accumulated into, as with `beamform`."""
    data = _make_sequence(np.dtype(np.complex64))
    out = np.full((N_VOXELS, N_FRAMES), 7.0 + 3.0j, dtype=np.complex64)

    result = kernel.beamform_compound(
        data["channel_data"],
        data["rx_coords_m"],
        data["scan_coords_m"],
        data["tx_wave_arrivals_s"],
        out,
        rx_start_s=0.0,
        sampling_freq_hz=SAMPLING_FREQ_HZ,
        f_number=F_NUMBER,
        sound_speed_m_s=SOUND_SPEED_M_S,
        modulation_freq_hz=MODULATION_FREQ_HZ,
        tukey_alpha=TUKEY_ALPHA,
    )

    assert result is out
    np.testing.assert_allclose(out, _beamform_per_transmit(data), rtol=1e-4, atol=1e-5)


def test_experimental_beamform_matches_per_transmit_loop():
    """mach.experimental.beamform agrees with the loop whichever path it takes."""
    data = _make_sequence(np.dtype(np.complex64))

    result = mach.experimental.beamform(
        data["channel_data"],
        data["rx_coords_m"],
        data["scan_coords_m"],
        data["tx_wave_arrivals_s"],
        rx_start_s=0.0,
        sampling_freq_hz=SAMPLING_FREQ_HZ,
        f_number=F_NUMBER,
        sound_speed_m_s=SOUND_SPEED_M_S,
        modulation_freq_hz=MODULATION_FREQ_HZ,
        tukey_alpha=TUKEY_ALPHA,
    )

    np.testing.assert_allclose(np.asarray(result), _beamform_per_transmit(data), rtol=1e-4, atol=1e-5)
