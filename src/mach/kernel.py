"""Python bindings and wrapper for the compiled CUDA/SYCL kernel."""

from typing import Any, cast

from array_api_compat import is_writeable_array
from jaxtyping import Num, Real

# _backend resolves to the nanobind module of whichever backend was compiled
from mach import _backend
from mach._array_api import Array, array_namespace
from mach._backend import InterpolationType
from mach._backend import beamform as nb_beamform
from mach._check import ensure_contiguous, is_contiguous

#: Name of the active backend, either "cuda" or "sycl".
__backend__: str = _backend.BACKEND
#: NVCC version string when the CUDA backend is active, otherwise None.
__nvcc_version__: str | None = getattr(_backend.impl, "__nvcc_version__", None)
#: DPC++ version string when the SYCL backend is active, otherwise None.
__sycl_version__: str | None = getattr(_backend.impl, "__sycl_version__", None)

#: Whole-sequence entry point, or None on backends that only beamform one transmit.
nb_beamform_compound = getattr(_backend.impl, "beamform_compound", None)
#: Whether `beamform_compound` is available on the active backend.
SUPPORTS_COMPOUND: bool = nb_beamform_compound is not None

#: Whether input arrays may carry a SYCL queue that mach's queue is not ordered against.
_SYNC_DEVICE_QUEUES: bool = _backend.SHARES_DPCTL_CONTEXT


def _sync_input_queues(*arrays: Any) -> None:
    """Wait for pending dpnp/dpctl work on the inputs.

    mach beamforms on its own in-order queue. Sharing dpctl's context makes its
    arrays readable without a copy, but says nothing about ordering, so the work
    that filled them has to finish before the kernel starts.
    """
    if not _SYNC_DEVICE_QUEUES:
        return
    for array in arrays:
        queue = getattr(array, "sycl_queue", None)
        if queue is not None:
            queue.wait()

# Export the InterpolationType enum for public use
__all__ = ["SUPPORTS_COMPOUND", "InterpolationType", "beamform", "beamform_compound"]


def beamform(  # noqa: C901
    channel_data: Num[Array, "n_rx n_samples n_frames"],
    rx_coords_m: Real[Array, "n_rx xyz=3"],
    scan_coords_m: Real[Array, "n_scan xyz=3"],
    tx_wave_arrivals_s: Real[Array, " n_scan"],
    out: Num[Array, "n_scan n_frames"] | None = None,
    *,
    rx_start_s: float,
    sampling_freq_hz: float,
    f_number: float,
    sound_speed_m_s: float,
    modulation_freq_hz: float | None = None,
    tukey_alpha: float = 0.5,
    interp_type: InterpolationType = InterpolationType.Linear,
) -> Array:
    """GPU ultrasound beamforming with automatic GPU/CPU dispatch.

    This function implements delay-and-sum beamforming with the following features:
    - Dynamic aperture growth based on F-number
    - Tukey apodization with adjustable taper width
    - Support for both RF and IQ data
    - Multi-frame processing
    - Automatic array protocol detection and GPU/CPU dispatch
    - Mixed CPU/GPU array handling with automatic memory management

    For theoretical background on delay-and-sum beamforming, see:
    Perrot et al., "So you think you can DAS? A viewpoint on delay-and-sum beamforming"
    https://www.biomecardio.com/publis/ultrasonics21.pdf

    Runs on the compiled backend selected by `mach._backend`: CUDA on NVIDIA GPUs,
    or SYCL/DPC++ on Intel GPUs and CPUs.

    `beamform` is a wrapper around `nb_beamform`, a nanobind-generated
    Python/C++ function. `beamform` adds more helpful error-messages than `nb_beamform`,
    but it adds about 0.1ms overhead (on AMD Ryzen Threadripper Pro).
    If your inputs are properly shaped/typed, or you are okay reading nanobind's
    slightly-confusing type-check error messages, you can use `nb_beamform` directly.

    Args:
        channel_data:
            RF/IQ data with shape (n_rx, n_samples, n_frames).
            For I/Q data: use complex64 dtype.
            For RF data: use float32 dtype.
            Note: this layout order improves memory-access patterns for the GPU kernel.
        rx_coords_m:
            Receive element positions with shape (n_rx, 3) where each row is [x, y, z] in meters.
            Each element represents the physical location of a transducer element on the probe.
        scan_coords_m:
            Scan grid point coordinates with shape (n_scan, 3) where each row is [x, y, z] in meters.
            These are the spatial locations where beamformed values will be computed.
            Note: n_scan = number of points_m in the imaging grid where you want beamformed output.
        tx_wave_arrivals_s:
            Transmit wave arrival times with shape (n_scan,) in seconds.
            This represents the time when the transmitted acoustic wave arrives at each
            scan grid point. For different transmit types:
            - Plane wave: arrivals computed from wave direction and grid positions
            - Focused/diverging wave: arrivals computed from focal point and grid positions

            Use `mach.wavefront.plane() / sound_speed_m_s` or
            `mach.wavefront.spherical() / sound_speed_m_s` to compute these values.
        out:
            Optional output array with shape (n_scan, nframes).
            Must match input type: complex64 for I/Q, float32 for RF.
        rx_start_s:
            Receive start time offset in seconds. This corresponds to t0 in the literature
            (biomecardio.com/publis/ultrasonics21.pdf) - the time when the 0th sample
            was recorded relative to the transmit event. When rx_start_s=0, the wave
            is assumed to pass through the coordinate origin_m at t=0.
        sampling_freq_hz:
            Sampling frequency in Hz.
        f_number:
            F-number for aperture calculations. Controls the size of the receive aperture
            based on depth. Typical values range from 1.0 to 3.0.
        sound_speed_m_s:
            Speed of sound in m/s. Typical value for soft tissue is ~1540 m/s.
        modulation_freq_hz:
            Center frequency in Hz (only used for I/Q data; ignored for RF data).
            For I/Q data: required parameter, set to 0 if no demodulation was used.
            For RF data: automatically defaults to 0.0 if not provided.
        tukey_alpha:
            Tukey window alpha parameter for apodization. Range [0, 1]:
            - 0.0: no apodization (rectangular window)
            - 0.5: moderate apodization (default)
            - 1.0: maximum apodization (Hann window)
        interp_type:
            Interpolation method for sensor data sampling. Options:
            - InterpolationType.NearestNeighbor: Nearest neighbor (fastest, lower quality)
            - InterpolationType.Linear: Linear interpolation (default, good balance)
            - InterpolationType.Quadratic: Quadratic interpolation (higher quality, slower)

    Returns:
        Beamformed data with shape (n_scan, nframes).
        Will be out if provided, otherwise a new array will be created.
        Output dtype matches input dtype (complex64 or float32).

    Notes:
        - All spatial coordinates should be in meters.
        - All time values should be in seconds.
        - All frequencies should be in Hz.
        - For optimal performance, use contiguous arrays with appropriate dtypes.
        - If the input arrays are not contiguous, the function automatically handles memory layout conversion.
        - Arrays can be on different devices (CPU/GPU); automatic copying will be performed with performance warnings.

    Examples:
        Basic plane wave beamforming:

        >>> import numpy as np
        >>> from mach import beamform, wavefront
        >>>
        >>> # Set up geometry
        >>> rx_positions = np.array([[0, 0, 0], [1e-3, 0, 0]])  # 2 elements, 1mm spacing
        >>> scan_points = np.array([[0, 0, 10e-3], [0, 0, 20e-3]])  # 2 depths: 10mm, 20mm
        >>>
        >>> # Compute transmit arrivals for 0° plane wave
        >>> arrivals_dist = wavefront.plane(
        ...     origin_m=np.array([0, 0, 0]),
        ...     points_m=scan_points,
        ...     direction=np.array([0, 0, 1])  # +z direction
        ... )
        >>> tx_arrivals = arrivals_dist / 1540  # Convert to time (assuming 1540 m/s)
        >>>
        >>> # Beamform (assuming you have channel_data)
        >>> result = beamform(
        ...     channel_data=channel_data,  # shape: (2, n_samples, n_frames)
        ...     rx_coords_m=rx_positions,
        ...     scan_coords_m=scan_points,
        ...     tx_wave_arrivals_s=tx_arrivals,
        ...     rx_start_s=0.0,
        ...     sampling_freq_hz=40e6,
        ...     f_number=1.5,
        ...     sound_speed_m_s=1540
        ... )
        >>>
        >>> # Use nearest neighbor interpolation for faster processing
        >>> from mach.kernel import InterpolationType
        >>> result_fast = beamform(
        ...     channel_data=channel_data,
        ...     rx_coords_m=rx_positions,
        ...     scan_coords_m=scan_points,
        ...     tx_wave_arrivals_s=tx_arrivals,
        ...     rx_start_s=0.0,
        ...     sampling_freq_hz=40e6,
        ...     f_number=1.5,
        ...     sound_speed_m_s=1540,
        ...     interp_type=InterpolationType.NearestNeighbor
        ... )
        >>>
        >>> # Use quadratic interpolation for highest quality
        >>> result_hq = beamform(
        ...     channel_data=channel_data,
        ...     rx_coords_m=rx_positions,
        ...     scan_coords_m=scan_points,
        ...     tx_wave_arrivals_s=tx_arrivals,
        ...     rx_start_s=0.0,
        ...     sampling_freq_hz=40e6,
        ...     f_number=1.5,
        ...     sound_speed_m_s=1540,
        ...     interp_type=InterpolationType.Quadratic
        ... )
    """

    # Check input type and dtype before trying to convert dtypes
    # shape should be checked in the kernel
    if not isinstance(channel_data, cast(type, Num[Array, "..."])):
        channel_data_type = type(channel_data)
        channel_data_dtype = getattr(channel_data, "dtype", None)
        raise TypeError(
            f"channel_data must be array with dtype=numeric, got type={channel_data_type}, dtype={channel_data_dtype}"
        )
    if not isinstance(rx_coords_m, cast(type, Real[Array, "..."])):
        rx_coords_m_type = type(rx_coords_m)
        rx_coords_m_dtype = getattr(rx_coords_m_type, "dtype", None)
        raise TypeError(
            f"rx_coords_m must be array with dtype=Real, got type={rx_coords_m_type}, dtype={rx_coords_m_dtype}"
        )
    if not isinstance(scan_coords_m, cast(type, Real[Array, "..."])):
        scan_coords_m_type = type(scan_coords_m)
        scan_coords_m_dtype = getattr(scan_coords_m_type, "dtype", None)
        raise TypeError(
            f"scan_coords_m must be array with dtype=Real, got type={scan_coords_m_type}, dtype={scan_coords_m_dtype}"
        )
    if not isinstance(tx_wave_arrivals_s, cast(type, Real[Array, "..."])):
        tx_wave_arrivals_s_type = type(tx_wave_arrivals_s)
        tx_wave_arrivals_s_dtype = getattr(tx_wave_arrivals_s_type, "dtype", None)
        raise TypeError(
            f"tx_wave_arrivals_s must be array with dtype=Real, got type={tx_wave_arrivals_s_type}, dtype={tx_wave_arrivals_s_dtype}"
        )
    if (out is not None) and (not isinstance(out, cast(type, Num[Array, "..."]))):
        out_type = type(out)
        out_dtype = getattr(out_type, "dtype", None)
        raise TypeError(f"out must be array with dtype=numeric, got type={out_type}, dtype={out_dtype}")

    # Get array namespaces
    xp_data = array_namespace(channel_data)
    xp_coords = array_namespace(rx_coords_m)
    xp_grid = array_namespace(scan_coords_m)
    xp_idt = array_namespace(tx_wave_arrivals_s)

    # Ensure arrays have required DLPack methods
    for arr, name in [
        (channel_data, "channel_data"),
        (rx_coords_m, "rx_coords_m"),
        (scan_coords_m, "scan_coords_m"),
        (tx_wave_arrivals_s, "tx_wave_arrivals_s"),
    ]:
        if not hasattr(arr, "__dlpack_device__"):
            raise TypeError(f"Array '{name}' does not support DLPack protocol")

    if out is not None and not hasattr(out, "__dlpack_device__"):
        raise TypeError("Array 'out' does not support DLPack protocol")

    # Ensure float32 dtype for coordinate arrays
    rx_coords_m = xp_coords.astype(rx_coords_m, xp_coords.float32, copy=False)
    scan_coords_m = xp_grid.astype(scan_coords_m, xp_grid.float32, copy=False)
    tx_wave_arrivals_s = xp_idt.astype(tx_wave_arrivals_s, xp_idt.float32, copy=False)

    nframes = channel_data.shape[2]
    n_scan = scan_coords_m.shape[0]

    # Determine data type and prepare sensor data
    is_complex = (channel_data.dtype == xp_data.complex64) or (channel_data.dtype == xp_data.complex128)
    if is_complex:
        channel_data = xp_data.astype(channel_data, xp_data.complex64, copy=False)
        output_dtype = xp_data.complex64
        if modulation_freq_hz is None:
            raise ValueError(
                "modulation_freq_hz is required for complex phase-correction. set it to 0 if no demodulation was used."
            )
    else:
        channel_data = xp_data.astype(channel_data, xp_data.float32, copy=False)
        output_dtype = xp_data.float32
        if modulation_freq_hz is None:
            modulation_freq_hz = 0.0

    # Check for contiguous arrays in libraries that support it
    channel_data = ensure_contiguous(channel_data)
    rx_coords_m = ensure_contiguous(rx_coords_m)
    scan_coords_m = ensure_contiguous(scan_coords_m)
    tx_wave_arrivals_s = ensure_contiguous(tx_wave_arrivals_s)

    if out is None:
        out = xp_data.zeros((n_scan, nframes), dtype=output_dtype)
    else:
        # Validate output array dtype matches input
        expected_dtype = xp_data.complex64 if is_complex else xp_data.float32
        if out.dtype != expected_dtype:
            raise ValueError(f"Output array dtype {out.dtype} doesn't match expected {expected_dtype}")

    if not is_writeable_array(out):
        raise ValueError("Output array `out` is not writable. Try using cupy or pytorch for the output array.")
    if not is_contiguous(cast(Array, out)):
        raise ValueError("Output array `out` must be contiguous.")

    assert modulation_freq_hz is not None

    _sync_input_queues(channel_data, rx_coords_m, scan_coords_m, tx_wave_arrivals_s, out)

    # Nanobind's generated stub types these args as NumPy `ArrayLike`,
    # but we intentionally pass broader DLPack-compatible arrays (NumPy/CuPy/JAX/custom),
    # so we ignore the type error from nanobind's generated stub.
    nb_beamform(  # ty: ignore[no-matching-overload]
        channel_data=channel_data,
        rx_coords_m=rx_coords_m,
        scan_coords_m=scan_coords_m,
        tx_wave_arrivals_s=tx_wave_arrivals_s,
        out=out,
        f_number=f_number,
        rx_start_s=rx_start_s,
        sampling_freq_hz=sampling_freq_hz,
        sound_speed_m_s=sound_speed_m_s,
        modulation_freq_hz=modulation_freq_hz,
        tukey_alpha=tukey_alpha,
        interp_type=interp_type,
    )

    return cast(Array, out)


def beamform_compound(
    channel_data: Num[Array, "n_transmits n_rx n_samples n_frames"],
    rx_coords_m: Real[Array, "n_rx xyz=3"],
    scan_coords_m: Real[Array, "n_scan xyz=3"],
    tx_wave_arrivals_s: Real[Array, "n_transmits n_scan"],
    out: Num[Array, "n_scan n_frames"] | None = None,
    *,
    rx_start_s: float,
    sampling_freq_hz: float,
    f_number: float,
    sound_speed_m_s: float,
    modulation_freq_hz: float | None = None,
    tukey_alpha: float = 0.5,
    interp_type: InterpolationType = InterpolationType.Linear,
) -> Array:
    """Beamform a whole transmit sequence and coherently compound it in a single call.

    Equivalent to calling `beamform` once per transmit and summing the results, but
    the probe geometry, the scan grid and the arrival times cross the bus once for
    the sequence instead of once per transmit, the accumulator stays in device
    memory throughout, and there is one host/device synchronization instead of one
    per transmit. Only `channel_data` is transferred per transmit.

    Not every backend provides this; check `SUPPORTS_COMPOUND` before calling.

    Args:
        channel_data:
            RF/IQ data with shape (n_transmits, n_rx, n_samples, n_frames), i.e. the
            shape `beamform` takes with a leading transmit axis.
        tx_wave_arrivals_s:
            Transmit wave arrival times with shape (n_transmits, n_scan), in seconds.
        out:
            Optional output array with shape (n_scan, n_frames). An array already on
            the device is accumulated into; a host array is overwritten.

    See `beamform` for the remaining arguments.

    Returns:
        Compounded data with shape (n_scan, n_frames). Will be `out` if provided,
        otherwise a new array.

    Raises:
        NotImplementedError: if the active backend has no compound entry point.
    """
    if nb_beamform_compound is None:
        raise NotImplementedError(
            f"The {__backend__!r} backend has no compound entry point. "
            "Beamform each transmit with mach.kernel.beamform and sum the results."
        )

    xp_data = array_namespace(channel_data)
    xp_coords = array_namespace(rx_coords_m)
    xp_grid = array_namespace(scan_coords_m)
    xp_idt = array_namespace(tx_wave_arrivals_s)

    n_scan = scan_coords_m.shape[0]
    n_frames = channel_data.shape[3]

    is_complex = (channel_data.dtype == xp_data.complex64) or (channel_data.dtype == xp_data.complex128)
    if is_complex:
        channel_data = xp_data.astype(channel_data, xp_data.complex64, copy=False)
        output_dtype = xp_data.complex64
        if modulation_freq_hz is None:
            raise ValueError(
                "modulation_freq_hz is required for complex phase-correction. set it to 0 if no demodulation was used."
            )
    else:
        channel_data = xp_data.astype(channel_data, xp_data.float32, copy=False)
        output_dtype = xp_data.float32
        if modulation_freq_hz is None:
            modulation_freq_hz = 0.0

    channel_data = ensure_contiguous(channel_data)
    rx_coords_m = ensure_contiguous(xp_coords.astype(rx_coords_m, xp_coords.float32, copy=False))
    scan_coords_m = ensure_contiguous(xp_grid.astype(scan_coords_m, xp_grid.float32, copy=False))
    tx_wave_arrivals_s = ensure_contiguous(xp_idt.astype(tx_wave_arrivals_s, xp_idt.float32, copy=False))

    if out is None:
        out = xp_data.zeros((n_scan, n_frames), dtype=output_dtype)
    elif out.dtype != output_dtype:
        raise ValueError(f"Output array dtype {out.dtype} doesn't match expected {output_dtype}")

    if not is_writeable_array(out):
        raise ValueError("Output array `out` is not writable. Try using cupy or pytorch for the output array.")
    if not is_contiguous(cast(Array, out)):
        raise ValueError("Output array `out` must be contiguous.")

    _sync_input_queues(channel_data, rx_coords_m, scan_coords_m, tx_wave_arrivals_s, out)

    nb_beamform_compound(
        channel_data=channel_data,
        rx_coords_m=rx_coords_m,
        scan_coords_m=scan_coords_m,
        tx_wave_arrivals_s=tx_wave_arrivals_s,
        out=out,
        f_number=f_number,
        rx_start_s=rx_start_s,
        sampling_freq_hz=sampling_freq_hz,
        sound_speed_m_s=sound_speed_m_s,
        modulation_freq_hz=modulation_freq_hz,
        tukey_alpha=tukey_alpha,
        interp_type=interp_type,
    )

    return cast(Array, out)
