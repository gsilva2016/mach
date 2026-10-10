"""Backend-agnostic beamforming benchmarks.

These run against whichever backend mach was built with (CUDA or SYCL) using NumPy
inputs, so they need no device-array library. The benchmarks under tests/compare/
cannot serve this purpose: they require CuPy and an NVIDIA GPU.

Sweeping the problem size separates the fixed per-call cost (device allocation,
kernel launch, synchronization, Python wrapper) from the part that scales with
data volume (host<->device staging and kernel execution).
"""

import numpy as np
import pytest

from mach.kernel import SUPPORTS_COMPOUND, beamform, beamform_compound, nb_beamform

# Exercises whichever backend is installed, so no NVIDIA GPU is implied
pytestmark = pytest.mark.no_cuda

N_RX = 128
N_SAMPLES = 1024
PITCH_M = 0.3e-3
SAMPLING_FREQ_HZ = 40e6
SOUND_SPEED_M_S = 1540.0
MODULATION_FREQ_HZ = 5e6
F_NUMBER = 1.5
TUKEY_ALPHA = 0.5


def _make_inputs(n_voxels: int, n_frames: int) -> dict:
    """Synthetic I/Q dataset for a linear probe imaging a column of scan points."""
    rng = np.random.default_rng(0)

    # Fill in place to avoid a complex128 intermediate for the larger cases
    channel_data = np.empty((N_RX, N_SAMPLES, n_frames), dtype=np.complex64)
    channel_data.real = rng.standard_normal((N_RX, N_SAMPLES, n_frames), dtype=np.float32)
    channel_data.imag = rng.standard_normal((N_RX, N_SAMPLES, n_frames), dtype=np.float32)

    rx_coords_m = np.zeros((N_RX, 3), dtype=np.float32)
    rx_coords_m[:, 0] = (np.arange(N_RX, dtype=np.float32) - (N_RX - 1) / 2) * PITCH_M

    depths_m = np.linspace(5e-3, 50e-3, n_voxels, dtype=np.float32)
    scan_coords_m = np.zeros((n_voxels, 3), dtype=np.float32)
    scan_coords_m[:, 2] = depths_m

    return {
        "channel_data": channel_data,
        "rx_coords_m": rx_coords_m,
        "scan_coords_m": scan_coords_m,
        "tx_wave_arrivals_s": (depths_m / SOUND_SPEED_M_S).astype(np.float32),
        "out": np.zeros((n_voxels, n_frames), dtype=np.complex64),
    }


def _beamform_nanobind(inputs: dict) -> None:
    """Call straight into the compiled module, skipping the Python wrapper."""
    nb_beamform(
        channel_data=inputs["channel_data"],
        rx_coords_m=inputs["rx_coords_m"],
        scan_coords_m=inputs["scan_coords_m"],
        tx_wave_arrivals_s=inputs["tx_wave_arrivals_s"],
        out=inputs["out"],
        f_number=F_NUMBER,
        rx_start_s=0.0,
        sampling_freq_hz=SAMPLING_FREQ_HZ,
        sound_speed_m_s=SOUND_SPEED_M_S,
        modulation_freq_hz=MODULATION_FREQ_HZ,
        tukey_alpha=TUKEY_ALPHA,
    )


def _beamform_wrapper(inputs: dict) -> None:
    beamform(
        inputs["channel_data"],
        inputs["rx_coords_m"],
        inputs["scan_coords_m"],
        inputs["tx_wave_arrivals_s"],
        inputs["out"],
        rx_start_s=0.0,
        sampling_freq_hz=SAMPLING_FREQ_HZ,
        f_number=F_NUMBER,
        sound_speed_m_s=SOUND_SPEED_M_S,
        modulation_freq_hz=MODULATION_FREQ_HZ,
        tukey_alpha=TUKEY_ALPHA,
    )


# A host `out` array is staged and zeroed on the device every call, so repeated
# rounds overwrite rather than accumulate and need no reset between them.
@pytest.mark.benchmark(group="scaling_voxels", min_rounds=3, warmup=True, warmup_iterations=1)
@pytest.mark.parametrize("n_voxels", [4_096, 16_384, 65_536, 262_144])
def test_scaling_voxels(benchmark, n_voxels):
    """Cost against voxel count at a fixed frame count."""
    inputs = _make_inputs(n_voxels, n_frames=1)
    benchmark(_beamform_nanobind, inputs)
    assert np.isfinite(inputs["out"]).all()


@pytest.mark.benchmark(group="scaling_frames", min_rounds=3, warmup=True, warmup_iterations=1)
@pytest.mark.parametrize("n_frames", [1, 4, 16])
def test_scaling_frames(benchmark, n_frames):
    """Cost against frame count at a fixed voxel count."""
    inputs = _make_inputs(16_384, n_frames=n_frames)
    benchmark(_beamform_nanobind, inputs)
    assert np.isfinite(inputs["out"]).all()


@pytest.mark.benchmark(group="call_overhead", min_rounds=5, warmup=True, warmup_iterations=1)
@pytest.mark.parametrize("entrypoint", ["nanobind", "wrapper"])
def test_call_overhead(benchmark, entrypoint):
    """Python wrapper cost, at a size small enough that the kernel is negligible."""
    inputs = _make_inputs(1_024, n_frames=1)
    runner = _beamform_nanobind if entrypoint == "nanobind" else _beamform_wrapper
    benchmark(runner, inputs)
    assert np.isfinite(inputs["out"]).all()


N_TRANSMITS = 16


def _make_sequence_inputs(n_voxels: int) -> dict:
    """The same dataset as `_make_inputs`, replicated across transmits."""
    single = _make_inputs(n_voxels, n_frames=1)
    return {
        "channel_data": np.broadcast_to(single["channel_data"], (N_TRANSMITS, *single["channel_data"].shape)).copy(),
        "rx_coords_m": single["rx_coords_m"],
        "scan_coords_m": single["scan_coords_m"],
        "tx_wave_arrivals_s": np.broadcast_to(single["tx_wave_arrivals_s"], (N_TRANSMITS, n_voxels)).copy(),
        "out": single["out"],
    }


def _compound_per_transmit(inputs: dict) -> None:
    """One `beamform` call per transmit, summed on the host."""
    total = None
    for transmit_idx in range(N_TRANSMITS):
        single = beamform(
            inputs["channel_data"][transmit_idx],
            inputs["rx_coords_m"],
            inputs["scan_coords_m"],
            inputs["tx_wave_arrivals_s"][transmit_idx],
            rx_start_s=0.0,
            sampling_freq_hz=SAMPLING_FREQ_HZ,
            f_number=F_NUMBER,
            sound_speed_m_s=SOUND_SPEED_M_S,
            modulation_freq_hz=MODULATION_FREQ_HZ,
            tukey_alpha=TUKEY_ALPHA,
        )
        total = single if total is None else total + single
    inputs["out"][:] = total


def _compound_whole_sequence(inputs: dict) -> None:
    """One call for the whole sequence, accumulating on the device."""
    beamform_compound(
        inputs["channel_data"],
        inputs["rx_coords_m"],
        inputs["scan_coords_m"],
        inputs["tx_wave_arrivals_s"],
        inputs["out"],
        rx_start_s=0.0,
        sampling_freq_hz=SAMPLING_FREQ_HZ,
        f_number=F_NUMBER,
        sound_speed_m_s=SOUND_SPEED_M_S,
        modulation_freq_hz=MODULATION_FREQ_HZ,
        tukey_alpha=TUKEY_ALPHA,
    )


# The throughput case mach is built for: a whole plane-wave sequence per frame.
# "loop" is the per-transmit baseline, "sequence" the single-call path; the gap
# between them is the per-transmit staging and synchronization that was removed.
@pytest.mark.benchmark(group="compounding", min_rounds=3, warmup=True, warmup_iterations=1)
@pytest.mark.parametrize("strategy", ["loop", "sequence"])
@pytest.mark.parametrize("n_voxels", [16_384, 65_536])
def test_compounding(benchmark, strategy, n_voxels):
    """Cost of compounding a transmit sequence, per strategy."""
    if strategy == "sequence" and not SUPPORTS_COMPOUND:
        pytest.skip("backend has no compound entry point")

    inputs = _make_sequence_inputs(n_voxels)
    runner = _compound_per_transmit if strategy == "loop" else _compound_whole_sequence
    benchmark(runner, inputs)
    assert np.isfinite(inputs["out"]).all()
