"""The specific function arguments / names (API) are experimental and may change."""

from typing import cast

from jaxtyping import Num, Real

from mach import _backend, kernel
from mach._array_api import Array, DLPackDevice, array_namespace

__all__ = ["DLPackDevice", "beamform"]


def beamform(
    channel_data: Num[Array, "n_transmits n_rx n_samples n_frames"],
    rx_coords_m: Real[Array, "n_rx xyz=3"],
    scan_coords_m: Real[Array, "n_points xyz=3"],
    tx_wave_arrivals_s: Real[Array, "n_transmits n_points"],
    out: Num[Array, "n_points n_frames"] | None = None,
    *,
    rx_start_s: float,
    sampling_freq_hz: float,
    f_number: float,
    sound_speed_m_s: float,
    modulation_freq_hz: float | None = None,
    tukey_alpha: float = 0.5,
) -> Num[Array, "n_points n_frames"]:
    """Wrapper around kernel.beamform that includes coherent compounding.

    The implementation takes some shortcuts for quick prototyping.

    Args:
        channel_data: like kernel.beamform channel_data, but with an extra first dimension for transmits
        tx_wave_arrivals_s: like kernel.beamform tx_wave_arrivals_s, but with an extra first dimension for transmits
    See kernel.beamform for other argument descriptions.

    Returns:
        beamformed+compounded data with shape (n_points, n_frames)
    """

    xp_data = array_namespace(channel_data)
    n_points = scan_coords_m.shape[0]
    n_transmits, _, _, n_frames = channel_data.shape
    is_complex = (channel_data.dtype == xp_data.complex64) or (channel_data.dtype == xp_data.complex128)
    output_dtype = xp_data.complex64 if is_complex else xp_data.float32

    # Allocate output array if not provided
    if out is None:
        out = xp_data.zeros((n_points, n_frames), dtype=output_dtype)
    else:
        # Validate output array dtype matches input
        expected_dtype = xp_data.complex64 if is_complex else xp_data.float32
        if out.dtype != expected_dtype:
            raise ValueError(f"Output array dtype {out.dtype} doesn't match expected {expected_dtype}")

    # Backends with a whole-sequence entry point compound the transmits without
    # returning to Python or to host memory in between.
    if kernel.SUPPORTS_COMPOUND:
        return kernel.beamform_compound(
            channel_data,
            rx_coords_m,
            scan_coords_m,
            tx_wave_arrivals_s,
            out=out,
            rx_start_s=rx_start_s,
            sampling_freq_hz=sampling_freq_hz,
            f_number=f_number,
            sound_speed_m_s=sound_speed_m_s,
            modulation_freq_hz=modulation_freq_hz,
            tukey_alpha=tukey_alpha,
        )

    # TODO: clean this up
    # The kernel accumulates into an output array that is already on the device, so
    # an output array that is not gets replaced by a device-side or host-side accumulator.
    out_orig = out
    on_device = out_orig.__dlpack_device__()[0] == _backend.BACKEND_DEVICE
    accumulate_on_host = False
    device_xp = None
    if not on_device:
        device_xp = _backend.device_namespace()
        if device_xp is not None:
            out = device_xp.zeros((n_points, n_frames), dtype=output_dtype)
        else:
            # Without a device-array library, beamform each transmit into its own
            # array and sum those on the host.
            accumulate_on_host = True
            out = xp_data.zeros((n_points, n_frames), dtype=output_dtype)

    # Compounding is: simply summing the beamform data across transmits
    for transmit_idx in range(n_transmits):
        # Extract single-transmit data
        single_channel_data = channel_data[transmit_idx]

        # Call single-transmit beamform
        transmit_out = kernel.beamform(
            single_channel_data,
            rx_coords_m,
            scan_coords_m,
            tx_wave_arrivals_s[transmit_idx],
            out=None if accumulate_on_host else out,
            rx_start_s=rx_start_s,
            sampling_freq_hz=sampling_freq_hz,
            f_number=f_number,
            sound_speed_m_s=sound_speed_m_s,
            modulation_freq_hz=modulation_freq_hz,
            tukey_alpha=tukey_alpha,
        )
        if accumulate_on_host:
            out = out + transmit_out

    # Move the data back to the dedicated output array
    if not on_device:
        if not accumulate_on_host:
            assert device_xp is not None, "expected a device-array library for the device accumulator"
            # cupy and dpnp both spell host transfer the same way
            out = device_xp.asnumpy(out)

        out_orig[:] = out
        return out_orig

    return cast(Array, out)
