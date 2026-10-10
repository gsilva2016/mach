/*
 * SYCL / Intel oneAPI DPC++ port of `kernel.cu`.
 *
 * Builds the `mach._sycl_impl` extension module, which exposes the same
 * `beamform` overloads and `InterpolationType` enum as `mach._cuda_impl`.
 *
 * Mapping notes relative to the CUDA kernel:
 * - SYCL's slowest-varying nd_range dimension is 0 and the fastest-varying is
 *   the last one, which is the opposite of CUDA. Dimension 2 therefore carries
 *   CUDA's `x` index, dimension 1 carries `y`, and dimension 0 carries `z`.
 * - `__shared__` becomes a `sycl::local_accessor`, `__syncthreads()` becomes
 *   `sycl::group_barrier`, and `atomicAdd` becomes `sycl::atomic_ref`.
 * - Device memory is managed with USM (`sycl::malloc_device`) instead of thrust.
 */

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cctype>
#include <climits>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include "sycl_math.hpp"

// Normally supplied by CMake via target_compile_definitions
#ifndef MACH_SYCL_COMPILER_ID
#define MACH_SYCL_COMPILER_ID "unknown"
#endif
#ifndef MACH_SYCL_COMPILER_VERSION
#define MACH_SYCL_COMPILER_VERSION "unknown"
#endif
#ifndef MACH_SYCL_TARGETS_STR
#define MACH_SYCL_TARGETS_STR "spir64"
#endif

namespace nb = nanobind;
using namespace nb::literals;

using mach::Float2;
using mach::Float3;

/**
 * @brief Interpolation types for sensor data sampling
 */
enum class InterpolationType {
    NearestNeighbor = 0,  ///< Use nearest neighbor (no interpolation)
    Linear = 1,           ///< Use linear interpolation (default)
    Quadratic = 2         ///< Use quadratic interpolation
};

#ifdef MACH_SYCL_PROFILE
#include <chrono>

/** @brief Get the current timestamp in milliseconds */
double get_timestamp_ms() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration<double, std::milli>(duration).count();
}

/** @brief Struct to track section timing */
struct SectionTimer {
    const char* name;
    double start_time;
    SectionTimer(const char* section_name) : name(section_name) {
        start_time = get_timestamp_ms();
        printf("TIMER_START: %s (%.3f ms)\n", name, start_time);
    }
    ~SectionTimer() {
        double end_time = get_timestamp_ms();
        double elapsed = end_time - start_time;
        printf("TIMER_END: %s (%.3f ms, elapsed: %.3f ms)\n", name, end_time, elapsed);
    }
};

#define TIME_SECTION(name) SectionTimer section_timer(name)
#define TIME_FUNCTION() SectionTimer function_timer(__func__)

#else
// No-op implementations when profiling is disabled
#define TIME_SECTION(name)
#define TIME_FUNCTION()
#endif

#ifndef PI
#define PI 3.14159265358979323846f
#endif

// Work-group and batching configuration (mirrors the CUDA kernel)
#define DEFAULT_RECEIVE_ELEMENTS_BATCH_SIZE 140  // 8960 elements / 64 batches; might be able to increase to 150
#define DEFAULT_NUM_VOXELS_PER_BLOCK 8  // Tuned for large-scale beamforming with many frames
#define VOXELS_RECEIVE_ELEMENTS_BATCH_SIZE (DEFAULT_NUM_VOXELS_PER_BLOCK * DEFAULT_RECEIVE_ELEMENTS_BATCH_SIZE)
// Results in 140 * 8 * sizeof(Float2) = 8960 bytes of work-group local memory

// Some extra-tuning parameters for small-frame-count beamforming (i.e. not ensembles)
// Keep VOXELS_RECEIVE_ELEMENTS_BATCH_SIZE constant, and adjust voxels_per_block and receive_elements_batch_size
#define MAX_FRAME_THREADS_PER_BLOCK 32
#define MIN_THREADS_PER_BLOCK 32

// Custom debug assertion macro that can be toggled on/off with:
// icpx -fsycl -DMACH_SYCL_DEBUG kernel_sycl.cpp
#ifdef MACH_SYCL_DEBUG
    #include <cassert>
    #define DEBUG_ASSERT(condition) assert(condition)
#else
    #define DEBUG_ASSERT(condition) ((void)0)
#endif

// ---------------------------------------------------------------------------
// SYCL queue, error handling and USM helpers
// ---------------------------------------------------------------------------

namespace {

std::mutex g_async_error_mutex;
std::string g_async_error;

/**
 * @brief Capture asynchronous SYCL errors so they can be surfaced to Python.
 *
 * Without a handler the SYCL runtime terminates the process on an async error,
 * which would take the Python interpreter down with it.
 */
void async_exception_handler(sycl::exception_list exceptions) {
    for (const std::exception_ptr& exception : exceptions) {
        try {
            std::rethrow_exception(exception);
        } catch (const std::exception& error) {
            std::lock_guard<std::mutex> lock(g_async_error_mutex);
            if (g_async_error.empty()) {
                g_async_error = error.what();
            }
        }
    }
}

/**
 * @brief Emit a Python warning, propagating it when filters escalate it to an error.
 *
 * PyErr_WarnEx leaves a Python exception set when it returns -1 (e.g. under
 * `-W error`). Carrying on from there makes the next unrelated CPython call
 * fail with an opaque SystemError.
 */
void warn_or_throw(PyObject* category, const char* message) {
    if (PyErr_WarnEx(category, message, 1) < 0) {
        throw nb::python_error();
    }
}

/** @brief Re-raise any asynchronous SYCL error captured since the last check. */
void throw_pending_async_error() {
    std::string message;
    {
        std::lock_guard<std::mutex> lock(g_async_error_mutex);
        message.swap(g_async_error);
    }
    if (!message.empty()) {
        throw std::runtime_error("Asynchronous SYCL error: " + message);
    }
}

/**
 * @brief Pick the device to beamform on.
 *
 * Prefers a GPU (Level Zero on Intel hardware) and falls back to whatever the
 * default selector offers, which is usually the OpenCL CPU runtime. Set
 * MACH_SYCL_DEVICE to "gpu", "cpu" or "default" to override.
 */
sycl::device select_device() {
    const char* requested = std::getenv("MACH_SYCL_DEVICE");
    if (requested != nullptr && requested[0] != '\0') {
        std::string preference(requested);
        std::transform(preference.begin(), preference.end(), preference.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (preference == "gpu") {
            return sycl::device(sycl::gpu_selector_v);
        }
        if (preference == "cpu") {
            return sycl::device(sycl::cpu_selector_v);
        }
        if (preference == "default") {
            return sycl::device(sycl::default_selector_v);
        }
        throw std::runtime_error("MACH_SYCL_DEVICE must be one of 'gpu', 'cpu' or 'default', got '" + preference + "'");
    }

    try {
        return sycl::device(sycl::gpu_selector_v);
    } catch (const sycl::exception&) {
        return sycl::device(sycl::default_selector_v);
    }
}

/**
 * @brief The properties every mach queue is built with.
 *
 * In-order so that staging copies are visible to the kernel without explicit
 * event dependencies, matching the CUDA code's reliance on the default stream.
 * beamform_compound relies on this to pipeline transmits.
 */
sycl::property_list queue_properties() {
    return sycl::property_list{
        sycl::property::queue::in_order()
#ifdef MACH_SYCL_PROFILE
        ,
        sycl::property::queue::enable_profiling()
#endif
    };
}

/**
 * @brief Storage for the process-wide beamforming queue.
 *
 * Held in an optional rather than a plain static because the queue can be
 * rebuilt on another context, see replace_queue().
 */
std::optional<sycl::queue>& queue_storage() {
    static std::optional<sycl::queue> queue;
    return queue;
}

/** @brief The process-wide beamforming queue, created on first use. */
sycl::queue& get_queue() {
    std::optional<sycl::queue>& storage = queue_storage();
    if (!storage.has_value()) {
        storage.emplace(select_device(), async_exception_handler, queue_properties());
    }
    return *storage;
}

/** @brief Queue of the interop library whose context mach adopted, if any. */
std::optional<sycl::queue>& foreign_queue_storage() {
    static std::optional<sycl::queue> queue;
    return queue;
}

/**
 * @brief Wait for pending work on the interop library's queue.
 *
 * Sharing dpctl's context makes its arrays readable in place but says nothing
 * about ordering: mach runs a queue of its own, so whatever produced an input
 * array has to have finished before the kernel starts. Only called when an
 * input actually came from that library.
 */
void wait_for_foreign_queue() {
    std::optional<sycl::queue>& storage = foreign_queue_storage();
    if (storage.has_value()) {
        storage->wait();
    }
}

/**
 * @brief A device allocation that is kept between calls and only ever grows.
 *
 * Beamforming stages the same handful of arrays on every call, usually at the
 * same sizes, so allocating and freeing them per call puts the USM allocator on
 * the critical path. Holding onto the allocation trades idle device memory for
 * throughput.
 */
class StagingBuffer {
public:
    StagingBuffer() = default;
    StagingBuffer(const StagingBuffer&) = delete;
    StagingBuffer& operator=(const StagingBuffer&) = delete;

    ~StagingBuffer() { release(); }

    /** @brief A device pointer with room for at least `bytes`. */
    void* acquire(std::size_t bytes, sycl::queue& queue) {
        const sycl::context context = queue.get_context();
        if (pointer_ != nullptr && bytes <= capacity_ && context_.has_value() && *context_ == context) {
            return pointer_;
        }
        release();
        void* pointer = sycl::malloc_device(bytes, queue);
        if (pointer == nullptr) {
            throw std::runtime_error("Failed to allocate " + std::to_string(bytes) +
                                     " bytes of SYCL device memory on '" +
                                     queue.get_device().get_info<sycl::info::device::name>() + "'");
        }
        pointer_ = pointer;
        capacity_ = bytes;
        context_ = context;
        return pointer_;
    }

    void release() {
        if (pointer_ != nullptr && context_.has_value()) {
            sycl::free(pointer_, *context_);
        }
        pointer_ = nullptr;
        capacity_ = 0;
        context_.reset();
    }

private:
    void* pointer_ = nullptr;
    std::size_t capacity_ = 0;
    // Keyed on the context rather than the queue: USM allocations belong to a
    // context, and holding one by value keeps it alive until the free above,
    // whatever order the statics in this file are torn down in.
    std::optional<sycl::context> context_;
};

/** @brief One reusable staging slot per array the kernel reads or writes. */
struct StagingBuffers {
    StagingBuffer channel_data;
    StagingBuffer rx_coords_m;
    StagingBuffer scan_coords_m;
    StagingBuffer tx_arrivals_s;
    StagingBuffer out;

    void release_all() {
        channel_data.release();
        rx_coords_m.release();
        scan_coords_m.release();
        tx_arrivals_s.release();
        out.release();
    }
};

/**
 * @brief Drains the queue on the way out of a scope, including while unwinding.
 *
 * Beamforming enqueues copies that read host memory owned by the caller's
 * nanobind handles, and those handles are released as soon as the call frame
 * goes away. Anything that throws in between would otherwise leave the device
 * reading freed memory. Uses wait() rather than wait_and_throw(), since an
 * asynchronous error must not escape a destructor.
 */
struct QueueDrainGuard {
    sycl::queue& queue;

    ~QueueDrainGuard() {
        try {
            queue.wait();
        } catch (...) {  // NOLINT(bugprone-empty-catch)
        }
    }
};

/**
 * @brief The process-wide staging pool.
 *
 * nanobind holds the GIL for the whole of every bound call, so the pool is only
 * ever touched by one thread at a time.
 */
StagingBuffers& get_staging_buffers() {
    static StagingBuffers buffers;
    return buffers;
}

/**
 * @brief Rebuild the beamforming queue on another context.
 *
 * The pooled allocations belong to the context they were made in, so they are
 * handed back before the switch.
 */
void replace_queue(const sycl::context& context, const sycl::device& device) {
    std::optional<sycl::queue>& storage = queue_storage();
    if (storage.has_value()) {
        storage->wait();
    }
    get_staging_buffers().release_all();
    storage.emplace(context, device, async_exception_handler, queue_properties());
}

}  // namespace

/**
 * @brief Template function for nearest neighbor interpolation with bounds checking
 * @tparam DataType Either float or Float2
 * @param channel_data Pointer to sensor data
 * @param sample_idx Floating point sample index
 * @param receive_element_idx Index of receive element
 * @param frame_idx Index of frame
 * @param n_samples Number of samples per element
 * @param n_frames Number of frames
 * @param[out] is_valid Whether the sample is within bounds
 * @return Interpolated sensor sample (undefined if is_valid is false)
 */
template <typename DataType>
inline DataType interpolate_nearest(
    const DataType* const __restrict__ channel_data,
    float sample_idx,
    uint32_t receive_element_idx,
    uint32_t frame_idx,
    uint32_t n_samples,
    uint32_t n_frames,
    bool& is_valid
) {
    // For nearest neighbor, check if rounded sample is in bounds
    if ((sample_idx < -0.5f) || (sample_idx > (n_samples - 0.5f))) {
        is_valid = false;
        return DataType{};  // Return default-constructed value (won't be used)
    }

    const unsigned int sample_idx_round = mach::float_to_uint_rn(sample_idx);  // Round to nearest
    DEBUG_ASSERT(sample_idx_round < n_samples);  // Verify sample index is in bounds
    const uint32_t channel_data_idx = receive_element_idx * n_samples * n_frames +
                                      sample_idx_round * n_frames +
                                      frame_idx;
    DEBUG_ASSERT(channel_data_idx < static_cast<uint64_t>(n_samples) * n_frames * (receive_element_idx + 1));  // Verify channel data index is in bounds

    is_valid = true;
    return channel_data[channel_data_idx];
}

/**
 * @brief Template function for linear interpolation with bounds checking
 * @tparam DataType Either float or Float2
 * @param channel_data Pointer to sensor data
 * @param sample_idx Floating point sample index
 * @param receive_element_idx Index of receive element
 * @param frame_idx Index of frame
 * @param n_samples Number of samples per element
 * @param n_frames Number of frames
 * @param[out] is_valid Whether the sample is within bounds
 * @return Interpolated sensor sample (undefined if is_valid is false)
 */
template <typename DataType>
inline DataType interpolate_linear(
    const DataType* const __restrict__ channel_data,
    float sample_idx,
    uint32_t receive_element_idx,
    uint32_t frame_idx,
    uint32_t n_samples,
    uint32_t n_frames,
    bool& is_valid
) {
    // For linear interpolation, check if floor/ceil samples are in bounds
    if ((sample_idx < 0.0f) || (sample_idx > (n_samples - 1))) {
        is_valid = false;
        return DataType{};  // Return default-constructed value (won't be used)
    }

    const unsigned int sample_idx_floor = mach::float_to_uint_rd(sample_idx);
    const unsigned int sample_idx_ceil = mach::float_to_uint_ru(sample_idx);
    const float lerp_alpha = sample_idx - (float)sample_idx_floor;

    DEBUG_ASSERT(sample_idx_floor < n_samples);  // Verify floor sample index is in bounds
    DEBUG_ASSERT(sample_idx_ceil < n_samples);   // Verify ceil sample index is in bounds

    const uint32_t channel_data_idx_floor = receive_element_idx * n_samples * n_frames +
                                            sample_idx_floor * n_frames +
                                            frame_idx;
    const uint32_t channel_data_idx_ceil = receive_element_idx * n_samples * n_frames +
                                           sample_idx_ceil * n_frames +
                                           frame_idx;

    DEBUG_ASSERT(channel_data_idx_floor < static_cast<uint64_t>(n_samples) * n_frames * (receive_element_idx + 1));  // Verify floor channel data index is in bounds
    DEBUG_ASSERT(channel_data_idx_ceil < static_cast<uint64_t>(n_samples) * n_frames * (receive_element_idx + 1));   // Verify ceil channel data index is in bounds

    is_valid = true;
    return mach::lerp(channel_data[channel_data_idx_floor], channel_data[channel_data_idx_ceil], lerp_alpha);
}

/**
 * @brief Template function for quadratic interpolation with bounds checking
 * @tparam DataType Either float or Float2
 * @param channel_data Pointer to sensor data
 * @param sample_idx Floating point sample index
 * @param receive_element_idx Index of receive element
 * @param frame_idx Index of frame
 * @param n_samples Number of samples per element
 * @param n_frames Number of frames
 * @param[out] is_valid Whether the sample is within bounds
 * @return Interpolated sensor sample (undefined if is_valid is false)
 */
template <typename DataType>
inline DataType interpolate_quadratic(
    const DataType* const __restrict__ channel_data,
    float sample_idx,
    uint32_t receive_element_idx,
    uint32_t frame_idx,
    uint32_t n_samples,
    uint32_t n_frames,
    bool& is_valid
) {
    // For quadratic interpolation, we need 3 points centered around sample_idx
    // Check if all 3 points are in bounds
    if ((sample_idx < 1.0f) || (sample_idx > (n_samples - 2.0f))) {
        is_valid = false;
        return DataType{};  // Return default-constructed value (won't be used)
    }

    const unsigned int sample_idx_center = mach::float_to_uint_rn(sample_idx);  // Round to nearest for better symmetry

    // Indices for the 3 points: (center-1), center, (center+1)
    const unsigned int idx_neg1 = sample_idx_center - 1;  // Left point (x=-1)
    const unsigned int idx_0 = sample_idx_center;         // Center point (x=0)
    const unsigned int idx_1 = sample_idx_center + 1;     // Right point (x=1)

    DEBUG_ASSERT(idx_neg1 < n_samples);  // Verify left point is in bounds
    DEBUG_ASSERT(idx_0 < n_samples);     // Verify center point is in bounds
    DEBUG_ASSERT(idx_1 < n_samples);     // Verify right point is in bounds

    // Calculate channel data indices
    const uint32_t base_idx = receive_element_idx * n_samples * n_frames + frame_idx;
    const uint32_t channel_data_idx_neg1 = base_idx + idx_neg1 * n_frames;
    const uint32_t channel_data_idx_0 = base_idx + idx_0 * n_frames;
    const uint32_t channel_data_idx_1 = base_idx + idx_1 * n_frames;

    DEBUG_ASSERT(channel_data_idx_neg1 < static_cast<uint64_t>(n_samples) * n_frames * (receive_element_idx + 1));
    DEBUG_ASSERT(channel_data_idx_0 < static_cast<uint64_t>(n_samples) * n_frames * (receive_element_idx + 1));
    DEBUG_ASSERT(channel_data_idx_1 < static_cast<uint64_t>(n_samples) * n_frames * (receive_element_idx + 1));

    // Calculate Lagrange basis weights using actual sample_idx
    // For points at (center-1), center, (center+1), evaluating at sample_idx
    const float x = sample_idx - (float)sample_idx_center;  // x relative to center point

    // Lagrange basis polynomials:
    // L₋₁(x) = x(x-1)/2     (for point at center-1)
    // L₀(x) = (1-x)(1+x)    (for point at center)
    // L₁(x) = x(x+1)/2      (for point at center+1)
    const float w_neg1 = 0.5f * x * (x - 1.0f);      // Weight for left point (x=-1)
    const float w_0 = (1.0f - x) * (1.0f + x);       // Weight for center point (x=0)
    const float w_1 = 0.5f * x * (x + 1.0f);         // Weight for right point (x=1)

    // Get the 3 data points
    const DataType data_neg1 = channel_data[channel_data_idx_neg1];  // Left point (x=-1)
    const DataType data_0 = channel_data[channel_data_idx_0];        // Center point (x=0)
    const DataType data_1 = channel_data[channel_data_idx_1];        // Right point (x=1)

    // Compute weighted sum using Lagrange basis
    is_valid = true;
    return w_neg1 * data_neg1 + w_0 * data_0 + w_1 * data_1;
}

/**
 * @brief Unified template function for interpolation dispatch with bounds checking
 * @tparam DataType Either float or Float2
 * @tparam interpType Interpolation type (compile-time constant)
 */
template <typename DataType, InterpolationType interpType>
inline DataType interpolate_sample(
    const DataType* const __restrict__ channel_data,
    float sample_idx,
    uint32_t receive_element_idx,
    uint32_t frame_idx,
    uint32_t n_samples,
    uint32_t n_frames,
    bool& is_valid
) {
    if constexpr (interpType == InterpolationType::NearestNeighbor) {
        return interpolate_nearest<DataType>(channel_data, sample_idx, receive_element_idx, frame_idx, n_samples, n_frames, is_valid);
    } else if constexpr (interpType == InterpolationType::Linear) {
        return interpolate_linear<DataType>(channel_data, sample_idx, receive_element_idx, frame_idx, n_samples, n_frames, is_valid);
    } else if constexpr (interpType == InterpolationType::Quadratic) {
        return interpolate_quadratic<DataType>(channel_data, sample_idx, receive_element_idx, frame_idx, n_samples, n_frames, is_valid);
    }
}

/**
 * Calculate the number of voxels to process per work-group based on frame count.
 * For small frame counts (< 4), we increase voxels_per_block to maintain at least MIN_THREADS_PER_BLOCK work-items.
 * For larger frame counts, we use DEFAULT_NUM_VOXELS_PER_BLOCK for optimal performance.
 *
 * @param frames_per_block Number of frames to process in this work-group
 * @return Number of voxels to process per work-group
 */
static inline int calculate_voxels_per_block(int frames_per_block) {
    // For small frame counts, increase voxels to maintain minimum work-item count
    // For larger frame counts, use default voxels for optimal performance
    DEBUG_ASSERT(frames_per_block > 0);
    return std::max((MIN_THREADS_PER_BLOCK + frames_per_block - 1) / frames_per_block, DEFAULT_NUM_VOXELS_PER_BLOCK);
}

/**
 * Calculate the number of receive elements to process per batch based on voxels per work-group.
 * This maintains constant local memory usage by scaling inversely with voxels_per_block.
 *
 * @param voxels_per_block Number of voxels being processed in this work-group
 * @return Number of receive elements to process per batch
 */
static inline int calculate_receive_elements_batch_size(int voxels_per_block) {
    // Scale receive elements inversely with voxels to maintain constant local memory usage
    DEBUG_ASSERT(voxels_per_block > 0);
    return VOXELS_RECEIVE_ELEMENTS_BATCH_SIZE / voxels_per_block;
}

/**
 * @brief Tukey window apodization function
 *
 * @param r_norm: float, the normalized distance from the center of the aperture
 * @param alpha: float, the alpha parameter for the Tukey window
 *   - Range [0, 1]:
 *   - 0.0: no apodization (rectangular window)
 *   - 0.5: moderate apodization
 *   - 1.0: maximum apodization (Hann window)
 *
 * @remark: this is technically a half-window, because we only need
 * the positive half of the window for r_norm ∈ [0, 1].
 */
static inline float tukey_apod_weight(float r_norm, float alpha) {
    DEBUG_ASSERT(alpha >= 0.0f && alpha <= 1.0f);
    if ((r_norm < 0.0f) || (r_norm > 1.0f)) {
        return 0.0f;
    }
    if (r_norm <= (1.0f - alpha)) {
        // flat region
        return 1.0f;
    }
    // positive-taper region
    // use the mirror of the negative-taper region
    // https://en.wikipedia.org/wiki/Window_function#Tukey_window
    // r_norm_mirror corresponds to n/2 in the negative-taper region
    const float r_norm_mirror = 1.0f - r_norm;
    const float weight = 0.5f - 0.5f * sycl::cos(PI * r_norm_mirror / alpha);
    return weight;
}

/**
 * @brief Calculate the transmit+receive delay and apodization weight for a single element position
 * @param rx_coord_m: Float3, the position of the receive element (m)
 * @param voxel_xyz: Float3, the position of the voxel (m)
 * @param aperture_radius_squared: float, the square of the aperture radius (m^2)
 * @param aperture_radius: float, the aperture radius (m)
 * @param voxel_tx_delay_s: float, the transmit delay time (seconds, includes rx_start_s offset)
 * @param inv_sound_speed_m_s: float, the inverse of the speed of sound in medium (seconds/meters)
 * @param tukey_alpha: float, the alpha parameter for the Tukey window
 * @tparam UseApodization: bool, whether to use apodization
 * @return Float2: (physical_tau_s, apod_weight)
 *         physical_tau_s: total physical wave-travel time in seconds (for phase correction)
 *              note: slightly different definition from biomecardio.com/publis/ultrasonics21.pdf
 *              here: tau = (d_TX + d_RX) / c, (does not include t0)
 *         apod_weight: apodization weight
 */
template <bool UseApodization>
static inline Float2 calculateTxRxDelayAndApodization(
    const Float3 rx_coord_m,
    const Float3 voxel_xyz,
    const float aperture_radius_squared,
    const float aperture_radius,
    const float voxel_tx_delay_s,
    const float inv_sound_speed_m_s,
    float tukey_alpha
) {
    // Compute relative position and distance
    const float dx = rx_coord_m.x - voxel_xyz.x;
    const float dy = rx_coord_m.y - voxel_xyz.y;
    const float dz = rx_coord_m.z - voxel_xyz.z;
    const float horizontal_distance_squared = dx * dx + dy * dy;

    // Skip computation if outside aperture based on F-number (early out)
    if (horizontal_distance_squared > aperture_radius_squared) {
        return mach::make_float2(-1.0f, 0.0f);  // Return invalid marker for elements outside aperture
    }

    // Calculate physical distances and wave-travel times
    const float rx_distance = sycl::sqrt(horizontal_distance_squared + dz * dz);
    const float rx_delay_s = rx_distance * inv_sound_speed_m_s;  // rx_distance / speed_of_sound

    // For phase calculation: physical wave-travel time in seconds (tau = tx_time + rx_time)
    const float physical_tau_s = voxel_tx_delay_s + rx_delay_s;

    if constexpr (!UseApodization) {
        return mach::make_float2(physical_tau_s, 1.0f);
    }

    // Calculate apodization weight
    const float horizontal_distance = sycl::sqrt(horizontal_distance_squared);  // Radial distance
    // Look up the apodization weight using the Tukey window function
    const float weight = tukey_apod_weight(horizontal_distance / aperture_radius, tukey_alpha);

    return mach::make_float2(physical_tau_s, weight);
}

/**
 * @brief Everything the beamforming kernel needs, bundled so the six
 *        template instantiations can share one launch path.
 *
 * @tparam DataType Either float (for RF data) or Float2 (for I/Q data)
 */
template <typename DataType>
struct BeamformArgs {
    const DataType* channel_data;       ///< [n_receive_elements][n_samples][n_frames]
    const Float3* rx_coords_m;          ///< [n_receive_elements] (x,y,z in meters)
    const Float3* output_voxels_xyz;    ///< [n_output_voxels] (x,y,z in meters)
    const float* tx_arrival_delays;     ///< [n_output_voxels] (seconds)
    DataType* beamformed;               ///< [n_output_voxels][n_frames]
    uint32_t n_frames;
    uint32_t n_receive_elements;
    uint32_t n_samples;
    float inv_sound_speed_m_s;
    float modulation_freq_hz;
    float sampling_freq_hz;
    float f_number;
    float tukey_alpha;
    float rx_start_s;
    uint64_t n_output_voxels;
    uint32_t receive_elements_batch_size;
};

/** @brief Kernel name, so the module exports stable symbols instead of lambda mangling. */
template <typename DataType, bool UseApodization, InterpolationType interpType>
class BeamformKernel;

/**
 * @brief SYCL kernel for delay-and-sum beamforming.
 *
 * Each work-item processes specific frames and receive elements for a particular voxel.
 *
 * Work-item organization (SYCL dimension 2 is the fastest-varying):
 * - Work-group range: (1, voxels, frames)
 * - Group range: (receive_element_batches, voxel_batches_y, voxel_batches_x)
 * - Each work-group processes VOXELS_RECEIVE_ELEMENTS_BATCH_SIZE voxels * receive elements.
 *   By default, this is 8 voxels * 140 receive elements = 1120, although this scales
 *   to increase work-item count if needed.
 *
 * Memory access patterns:
 * - Work-group local memory for delay/apodization tables (reused across frames)
 * - Coalesced memory access to sensor data
 * - Atomic operations for output accumulation across receive element batches
 *
 * @tparam DataType Either float (for RF data) or Float2 (for I/Q data)
 * @tparam UseApodization Whether to apply Tukey window apodization
 * @tparam interpType Interpolation method for sensor data sampling
 */
template <typename DataType, bool UseApodization, InterpolationType interpType>
sycl::event submit_beamform_kernel(
    sycl::queue& queue,
    const BeamformArgs<DataType>& args,
    sycl::range<3> global_range,
    sycl::range<3> local_range
) {
    // Ensure DataType is one of the supported types for ultrasound beamforming
    static_assert(std::is_same_v<DataType, float> || std::is_same_v<DataType, Float2>,
                  "DataType must be float (for RF data) or Float2 (for I/Q data). "
                  "Other types like double/double2 or half/half2 would require kernel modifications.");
    constexpr bool is_complex = std::is_same_v<DataType, Float2>;

    using AtomicFloatRef = sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                            sycl::access::address_space::global_space>;

    return queue.submit([&](sycl::handler& cgh) {
        // Work-group local memory - organized as a single flat array
        // Indexing dimensions: [voxel_tid][receive_element_idx_in_batch]
        sycl::local_accessor<Float2, 1> voxel_tau_and_apod_weights(
            sycl::range<1>(VOXELS_RECEIVE_ELEMENTS_BATCH_SIZE), cgh);

        cgh.parallel_for<BeamformKernel<DataType, UseApodization, interpType>>(
            sycl::nd_range<3>(global_range, local_range),
            [=](sycl::nd_item<3> item) {
                // local_id(2) is the fastest-varying dimension, matching CUDA's threadIdx.x
                const unsigned int frame_tid = static_cast<unsigned int>(item.get_local_id(2));
                const unsigned int voxel_tid = static_cast<unsigned int>(item.get_local_id(1));
                const unsigned int num_frame_threads = static_cast<unsigned int>(item.get_local_range(2));
                const unsigned int num_voxels_per_block = static_cast<unsigned int>(item.get_local_range(1));

                const uint32_t receive_element_block_start_idx =
                    static_cast<uint32_t>(item.get_group(0)) * args.receive_elements_batch_size;
                const uint32_t receive_elements_in_batch =
                    sycl::min(args.receive_elements_batch_size,
                              args.n_receive_elements - receive_element_block_start_idx);

                // Groups 2 and 1 jointly index the voxel batch, so that very large voxel
                // counts do not need an unbounded range in a single dimension.
                const uint32_t voxel_batch_idx = static_cast<uint32_t>(item.get_group(2)) +
                                                 static_cast<uint32_t>(item.get_group(1)) *
                                                     static_cast<uint32_t>(item.get_group_range(2));
                // voxel_batch_idx * num_voxels_per_block may overflow uint32_t, so we use uint64_t for voxel_idx
                const uint64_t voxel_idx =
                    static_cast<uint64_t>(voxel_batch_idx) * num_voxels_per_block + voxel_tid;

                // Pre-calculate group variables for performance
                // Note: modulation_freq_hz is only used for I/Q data (Float2), ignored for RF data (float)
                const float modulation_freq_rad = 2.0f * PI * args.modulation_freq_hz;

                // Every work-item in the group must reach the barrier below, so out-of-range
                // voxels skip the body rather than returning early.
                const bool voxel_in_range = voxel_idx < args.n_output_voxels;

                if (voxel_in_range) {
                    // Load grid point and tx delay value for this voxel
                    const Float3 voxel_xyz = args.output_voxels_xyz[voxel_idx];
                    const float voxel_tx_delay_s = args.tx_arrival_delays[voxel_idx];
                    const float aperture_radius = voxel_xyz.z / (2.0f * args.f_number);
                    const float aperture_radius_squared = aperture_radius * aperture_radius;

                    // For the delay calculation phase, we use the frame-work-items to parallelize over receive elements
                    // Pre-compute delays and weights, which are shared across frames, but are different for each voxel
                    for (unsigned int receive_element_idx_in_batch = frame_tid;
                         receive_element_idx_in_batch < receive_elements_in_batch;
                         receive_element_idx_in_batch += num_frame_threads) {
                        const uint32_t receive_element_idx =
                            receive_element_block_start_idx + receive_element_idx_in_batch;
                        const Float3 rx_coord_m = args.rx_coords_m[receive_element_idx];

                        const Float2 tau_and_weight = calculateTxRxDelayAndApodization<UseApodization>(
                            rx_coord_m,
                            voxel_xyz,
                            aperture_radius_squared,
                            aperture_radius,
                            voxel_tx_delay_s,
                            args.inv_sound_speed_m_s,
                            args.tukey_alpha
                        );

                        const uint32_t shared_mem_idx =
                            voxel_tid * receive_elements_in_batch + receive_element_idx_in_batch;
                        DEBUG_ASSERT(shared_mem_idx < VOXELS_RECEIVE_ELEMENTS_BATCH_SIZE);  // Verify local memory access is in bounds

                        voxel_tau_and_apod_weights[shared_mem_idx] = tau_and_weight;

                        DEBUG_ASSERT(receive_element_idx < args.n_receive_elements);  // Verify element index is valid
                        DEBUG_ASSERT(receive_element_idx_in_batch < receive_elements_in_batch);  // Verify batch index is valid
                    }
                }
                sycl::group_barrier(item.get_group());

                if (!voxel_in_range) {
                    return;
                }

                // Strided-loop over frames for coalesced memory access into channel_data
                for (uint32_t frame_idx = frame_tid; frame_idx < args.n_frames; frame_idx += num_frame_threads) {
                    DataType frame_sum{};

                    // Every work-item (managing 1 voxel, for 1 frame per iteration)
                    // Needs to sum over all receive elements in the batch
                    for (uint32_t receive_element_idx_in_batch = 0;
                         receive_element_idx_in_batch < receive_elements_in_batch;
                         receive_element_idx_in_batch++) {
                        const uint32_t receive_element_idx =
                            receive_element_block_start_idx + receive_element_idx_in_batch;
                        const uint32_t shared_mem_idx =
                            voxel_tid * receive_elements_in_batch + receive_element_idx_in_batch;
                        const Float2 tau_and_weight = voxel_tau_and_apod_weights[shared_mem_idx];
                        const float physical_tau_s = tau_and_weight.x;  // Physical wave-travel time in seconds (for phase)
                        const float apod_weight = tau_and_weight.y;     // Apodization weight

                        // Skip if element is outside aperture (marked with physical_tau_s = -1.0f) or apodization weight is 0
                        if ((physical_tau_s < 0.0f) || (apod_weight == 0.0f)) continue;

                        // Receive-sample-index (float, before interpolation)
                        const float sample_idx = (physical_tau_s - args.rx_start_s) * args.sampling_freq_hz;

                        // Use template-based interpolation dispatch with unified bounds checking
                        bool is_valid;
                        DataType sensor_sample = interpolate_sample<DataType, interpType>(
                            args.channel_data, sample_idx, receive_element_idx, frame_idx,
                            args.n_samples, args.n_frames, is_valid
                        );

                        // Skip if sample is outside bounds
                        if (!is_valid) continue;

                        if constexpr (UseApodization) {
                            sensor_sample *= apod_weight;
                        }

                        // Process and accumulate the sample data
                        if constexpr (is_complex) {
                            if (args.modulation_freq_hz != 0.0f) {
                                // Phase-shift I/Q data using physical wave-travel time (tau)
                                const float phi = modulation_freq_rad * physical_tau_s;
                                float cos_phi;
                                const float sin_phi = mach::sincos(phi, cos_phi);
                                const float shifted_sample_real =
                                    sycl::fma(sensor_sample.x, cos_phi, sycl::fma(sensor_sample.y, -sin_phi, 0.0f));
                                const float shifted_sample_imag =
                                    sycl::fma(sensor_sample.y, cos_phi, sycl::fma(sensor_sample.x, sin_phi, 0.0f));
                                frame_sum += mach::make_float2(shifted_sample_real, shifted_sample_imag);
                            } else {
                                // Special case for modulation_freq_hz = 0
                                frame_sum += sensor_sample;
                            }
                        } else {
                            // Real data - just accumulate (modulation_freq_hz is ignored for RF data)
                            frame_sum += sensor_sample;
                        }
                    }

                    const uint64_t beamformed_idx = voxel_idx * static_cast<uint64_t>(args.n_frames) +
                                                    static_cast<uint64_t>(frame_idx);
                    DEBUG_ASSERT(beamformed_idx < args.n_output_voxels * static_cast<uint64_t>(args.n_frames));  // Verify beamformed output index is valid

                    // Accumulate atomically, because receive-element batches share an output element
                    if constexpr (is_complex) {
                        AtomicFloatRef(args.beamformed[beamformed_idx].x).fetch_add(frame_sum.x);
                        AtomicFloatRef(args.beamformed[beamformed_idx].y).fetch_add(frame_sum.y);
                    } else {
                        AtomicFloatRef(args.beamformed[beamformed_idx]).fetch_add(frame_sum);
                    }
                }
            });
    });
}

/** @brief Dispatch over the runtime interpolation type to the compile-time specialization. */
template <typename DataType, bool UseApodization>
static sycl::event submit_with_interpolation(
    sycl::queue& queue,
    InterpolationType interp_type,
    const BeamformArgs<DataType>& args,
    sycl::range<3> global_range,
    sycl::range<3> local_range
) {
    switch (interp_type) {
        case InterpolationType::NearestNeighbor:
            return submit_beamform_kernel<DataType, UseApodization, InterpolationType::NearestNeighbor>(
                queue, args, global_range, local_range);
        case InterpolationType::Linear:
            return submit_beamform_kernel<DataType, UseApodization, InterpolationType::Linear>(
                queue, args, global_range, local_range);
        case InterpolationType::Quadratic:
            return submit_beamform_kernel<DataType, UseApodization, InterpolationType::Quadratic>(
                queue, args, global_range, local_range);
    }
    throw std::runtime_error("Error: unknown interpolation type " +
                             std::to_string(static_cast<int>(interp_type)));
}

/**
 * @brief Beamforming function template wrapper that calls the appropriate kernel based on the data type.
 *
 * This function sets up the SYCL nd-range and submits the beamforming kernel to perform
 * delay-and-sum beamforming. It handles both float (RF data) and Float2 (I/Q data) variants.
 *
 * @tparam DataType Either float (for RF data) or Float2 (for I/Q data)
 * @param queue The SYCL queue to submit to
 * @param d_channel_data Device pointer to sensor data [n_receive_elements, n_samples, n_frames]
 * @param d_rx_coords_m Device pointer to receive element positions [n_receive_elements, 3]
 * @param d_scan_coords_m Device pointer to output voxel positions [n_output_voxels, 3]
 * @param d_tx_arrivals_s Device pointer to transmit delays [n_output_voxels]
 * @param d_out Device pointer to output beamformed data [n_output_voxels, n_frames]
 * @param n_receive_elements Number of receive elements
 * @param n_samples Number of time samples
 * @param n_output_voxels Number of output voxels
 * @param n_frames Number of frames to beamform
 * @param f_number F-number for aperture growth control
 * @param rx_start_s Receive start time offset (seconds, corresponds to t0 in biomecardio.com/publis/ultrasonics21.pdf)
 * @param sampling_freq_hz Sampling frequency of channel_data (Hz)
 * @param sound_speed_m_s Speed of sound in medium (meters/second)
 * @param modulation_freq_hz Modulation frequency (Hz)
 * @param tukey_alpha Tukey window alpha for apodization (0=no apodization, 1=full apodization)
 * @param interp_type Interpolation method for sensor data sampling
 *
 * @return The kernel's event. The caller decides when to synchronize, so that a
 *         sequence of transmits costs one host/device round trip rather than one
 *         per transmit. The queue is in-order, so work enqueued afterwards (a
 *         copy of the result, or the next transmit) is already ordered correctly.
 */
template <typename DataType>
sycl::event _beamform_impl(
    sycl::queue& queue,
    const DataType* d_channel_data,
    const Float3* d_rx_coords_m,
    const Float3* d_scan_coords_m,
    const float* d_tx_arrivals_s,
    DataType* d_out,
    uint32_t n_receive_elements,
    uint32_t n_samples,
    uint64_t n_output_voxels,
    uint32_t n_frames,
    float f_number,
    float rx_start_s,
    float sampling_freq_hz,
    float sound_speed_m_s,
    float modulation_freq_hz,
    float tukey_alpha,
    InterpolationType interp_type
) {
    TIME_FUNCTION();

    // Check for potential overflow in sensor data indexing
    const uint64_t n_channel_data = static_cast<uint64_t>(n_receive_elements) * static_cast<uint64_t>(n_samples) * static_cast<uint64_t>(n_frames);
    if (n_channel_data > UINT32_MAX) {
        throw std::runtime_error("Error: Sensor data array size exceeds 32-bit indexing limit. Maximum size is " +
                                 std::to_string(UINT32_MAX) + " elements, but requested size is " +
                                 std::to_string(n_channel_data) + " elements.");
    }

    if (tukey_alpha < 0.0f || tukey_alpha > 1.0f) {
        throw std::runtime_error("Error: tukey_alpha must be in range [0, 1], but got " +
                                 std::to_string(tukey_alpha));
    }
    const bool apod_flag = tukey_alpha > 0.0f;
    const float inv_sound_speed_m_s = 1.0f / sound_speed_m_s;

    // An nd_range may not have a zero extent in any dimension
    if (n_output_voxels == 0 || n_frames == 0 || n_receive_elements == 0) {
        return sycl::event();
    }

    const sycl::device device = queue.get_device();

    // Calculate work-group dimensions
    const int frames_per_block = std::min<int>(static_cast<int>(n_frames), MAX_FRAME_THREADS_PER_BLOCK);
    const int voxels_per_block = calculate_voxels_per_block(frames_per_block);
    const std::size_t work_group_size = static_cast<std::size_t>(frames_per_block) * voxels_per_block;
    const std::size_t max_work_group_size = device.get_info<sycl::info::device::max_work_group_size>();
    if (work_group_size > max_work_group_size) {
        throw std::runtime_error("Error: Required work-group size (" + std::to_string(work_group_size) +
                                 ") exceeds the device limit (" + std::to_string(max_work_group_size) +
                                 "). Reducing MAX_FRAME_THREADS_PER_BLOCK or DEFAULT_NUM_VOXELS_PER_BLOCK is required.");
    }

    const int receive_elements_batch_size = calculate_receive_elements_batch_size(voxels_per_block);

#ifdef MACH_SYCL_DEBUG
    std::cout << "Work-group dimensions: " << frames_per_block << " (frames) x " << voxels_per_block
              << " (voxels) = " << work_group_size << " work-items per group" << std::endl;
#endif

    // Calculate group count - each work-group processes voxels_per_block voxels
    if (n_output_voxels > INT_MAX) {
        throw std::runtime_error("Error: Number of voxels (" + std::to_string(n_output_voxels) +
                                 ") exceeds the maximum integer value (" + std::to_string(INT_MAX) + ").");
    }
    const int num_blocks = static_cast<int>((n_output_voxels + voxels_per_block - 1) / voxels_per_block);
    const int max_blocks_per_dim = (1 << 16) - 32;  // Keep the CUDA kernel's 2D voxel-batch decomposition

    // x&y dimensions: voxel-batches, z dimension: receive-element-batches
    const int grid_x = std::min(max_blocks_per_dim, num_blocks);
    const int grid_y = (num_blocks + grid_x - 1) / grid_x;
    const int grid_z = static_cast<int>((n_receive_elements + receive_elements_batch_size - 1) /
                                        receive_elements_batch_size);

    // SYCL orders nd_range dimensions slowest-to-fastest, the reverse of CUDA's dim3
    const sycl::range<3> local_range(1, static_cast<std::size_t>(voxels_per_block),
                                     static_cast<std::size_t>(frames_per_block));
    const sycl::range<3> global_range(static_cast<std::size_t>(grid_z),
                                      static_cast<std::size_t>(grid_y) * voxels_per_block,
                                      static_cast<std::size_t>(grid_x) * frames_per_block);

#ifdef MACH_SYCL_PROFILE
    std::cout << "Group dimensions: " << grid_x << " x " << grid_y << " x " << grid_z << " = "
              << static_cast<int64_t>(grid_x) * grid_y * grid_z << " work-groups, each handling "
              << voxels_per_block << " voxels x " << receive_elements_batch_size
              << " receive_elements" << std::endl;
#endif

    // Check if our local memory allocation will fit
    static constexpr std::size_t local_mem_size = VOXELS_RECEIVE_ELEMENTS_BATCH_SIZE * sizeof(Float2);
    const std::size_t max_local_mem = device.get_info<sycl::info::device::local_mem_size>();

#ifdef MACH_SYCL_PROFILE
    std::cout << "Local memory per work-group: " << local_mem_size / 1024.0f << " KB" << std::endl;
    std::cout << "Maximum local memory available: " << max_local_mem / 1024.0f << " KB" << std::endl;
#endif

    if (local_mem_size > max_local_mem) {
        throw std::runtime_error("Error: Work-group local memory (" + std::to_string(local_mem_size) +
                                 " bytes) exceeds device limit (" + std::to_string(max_local_mem) +
                                 " bytes). Reducing DEFAULT_NUM_VOXELS_PER_BLOCK or DEFAULT_RECEIVE_ELEMENTS_BATCH_SIZE is required.");
    }

    BeamformArgs<DataType> args{};
    args.channel_data = d_channel_data;
    args.rx_coords_m = d_rx_coords_m;
    args.output_voxels_xyz = d_scan_coords_m;
    args.tx_arrival_delays = d_tx_arrivals_s;
    args.beamformed = d_out;
    args.n_frames = n_frames;
    args.n_receive_elements = n_receive_elements;
    args.n_samples = n_samples;
    args.inv_sound_speed_m_s = inv_sound_speed_m_s;
    args.modulation_freq_hz = modulation_freq_hz;
    args.sampling_freq_hz = sampling_freq_hz;
    args.f_number = f_number;
    args.tukey_alpha = tukey_alpha;
    args.rx_start_s = rx_start_s;
    args.n_output_voxels = n_output_voxels;
    args.receive_elements_batch_size = static_cast<uint32_t>(receive_elements_batch_size);

    sycl::event kernel_event;
    {
        TIME_SECTION("kernel_submit");

        // We use template compile-time specialization to handle the different cases
        // Dispatch based on apodization and interpolation type
        if (apod_flag) {
            kernel_event = submit_with_interpolation<DataType, true>(queue, interp_type, args, global_range, local_range);
        } else {
            kernel_event = submit_with_interpolation<DataType, false>(queue, interp_type, args, global_range, local_range);
        }
    }

#ifdef MACH_SYCL_PROFILE
    kernel_event.wait_and_throw();
    const uint64_t start_ns = kernel_event.get_profiling_info<sycl::info::event_profiling::command_start>();
    const uint64_t end_ns = kernel_event.get_profiling_info<sycl::info::event_profiling::command_end>();
    std::cout << "Kernel execution time: " << (end_ns - start_ns) / 1.0e6 << " ms" << std::endl;
#endif

    return kernel_event;
}

/**
 * @brief Helper function to convert ndarray shape to string representation
 * @tparam ArrayType Any nanobind ndarray type
 * @param array The ndarray to get shape from
 * @return String representation of shape like "[dim0, dim1, dim2]"
 */
template <typename ArrayType>
std::string shape_to_string(const ArrayType& array) {
    std::string result = "[";
    for (size_t i = 0; i < array.ndim(); ++i) {
        if (i > 0) result += ", ";
        result += std::to_string(array.shape(i));
    }
    result += "]";
    return result;
}

/**
 * @brief Check the dimensions of the input arrays
 */
template <typename SensorArrayType, typename CoordArrayType, typename TransmitArrayType, typename OutputArrayType>
void check_dimensions(
    const SensorArrayType& channel_data,
    const CoordArrayType& rx_coords_m,
    const CoordArrayType& scan_coords_m,
    const TransmitArrayType& tx_wave_arrivals_s,
    const OutputArrayType& out,
    size_t n_receive_elements,
    size_t n_samples,
    size_t n_output_voxels,
    size_t n_frames
) {
    // Validate dimensions
    if ((n_receive_elements != channel_data.shape(0)) || (n_receive_elements != rx_coords_m.shape(0))) {
        std::string error_msg = "Dimension mismatch in receive elements:\n";
        error_msg += "  Expected n_receive_elements: " + std::to_string(n_receive_elements) + "\n";
        error_msg += "  channel_data.shape: " + shape_to_string(channel_data) + "\n";
        error_msg += "  rx_coords_m.shape: " + shape_to_string(rx_coords_m) + "\n";
        error_msg += "→ channel_data.shape[0] and rx_coords_m.shape[0] must both equal n_receive_elements";
        throw std::runtime_error(error_msg);
    }
    if ((n_output_voxels != tx_wave_arrivals_s.shape(0)) || (n_output_voxels != scan_coords_m.shape(0)) || (n_output_voxels != out.shape(0))) {
        std::string error_msg = "Dimension mismatch in output voxels:\n";
        error_msg += "  Expected n_output_voxels: " + std::to_string(n_output_voxels) + "\n";
        error_msg += "  scan_coords_m.shape: " + shape_to_string(scan_coords_m) + "\n";
        error_msg += "  tx_wave_arrivals_s.shape: " + shape_to_string(tx_wave_arrivals_s) + "\n";
        error_msg += "  out.shape: " + shape_to_string(out) + "\n";
        error_msg += "→ scan_coords_m.shape[0], tx_wave_arrivals_s.shape[0], and out.shape[0] must all equal n_output_voxels";
        throw std::runtime_error(error_msg);
    }
    if (n_frames != channel_data.shape(2) || n_frames != out.shape(1)) {
        std::string error_msg = "Dimension mismatch in frames:\n";
        error_msg += "  Expected n_frames: " + std::to_string(n_frames) + "\n";
        error_msg += "  channel_data.shape: " + shape_to_string(channel_data) + "\n";
        error_msg += "  out.shape: " + shape_to_string(out) + "\n";
        error_msg += "→ channel_data.shape[2] and out.shape[1] must both equal n_frames";
        throw std::runtime_error(error_msg);
    }
}

/** @brief Whether an array's memory came from a SYCL library rather than the host. */
template <typename ArrayType>
bool is_oneapi_array(const ArrayType& array) {
    return array.device_type() == nb::device::oneapi::value;
}

/**
 * @brief Decide whether an array's pointer can be handed to the kernel as-is.
 *
 * A pointer is usable directly when it is a USM allocation belonging to our
 * context, or when it is ordinary host memory and the device can read host
 * allocations (true for the OpenCL CPU runtime). Everything else is staged
 * through a temporary device allocation.
 *
 * @param allow_host_pointer Pass false for the output array. The kernel
 *        accumulates into its output, and a staged buffer starts zeroed, so
 *        always staging a host output keeps the overwrite semantics that the
 *        CUDA backend has for CPU arrays.
 * @throws std::runtime_error if the array lives on an unsupported device, or is
 *         a oneAPI array from a different SYCL context.
 */
template <typename ArrayType>
bool can_use_pointer_directly(const ArrayType& array, const char* name, sycl::queue& queue,
                              bool allow_host_pointer = true) {
    const uint32_t device_type = array.device_type();

    if (device_type == nb::device::oneapi::value) {
        if (sycl::get_pointer_type(array.data(), queue.get_context()) == sycl::usm::alloc::unknown) {
            throw std::runtime_error(
                std::string("Array '") + name + "' is a oneAPI array whose memory does not belong to mach's "
                "SYCL context. Allocate it with a library that shares mach's context, or pass a NumPy array "
                "and let mach stage the copy.");
        }
        return true;
    }

    if (device_type == nb::device::cpu::value) {
        // Devices with usm_system_allocations (e.g. the OpenCL CPU runtime) can
        // dereference plain malloc'd memory, so no staging copy is needed.
        return allow_host_pointer && queue.get_device().has(sycl::aspect::usm_system_allocations);
    }

    throw std::runtime_error(
        std::string("Found input array '") + name + "' on device type " + std::to_string(device_type) +
        ". The SYCL backend supports CPU arrays and oneAPI (SYCL USM) arrays only.");
}

/**
 * @brief Main beamforming function that processes ultrasound data on a SYCL device
 *
 * This function automatically detects the device location of input arrays and handles
 * host<->device copying as needed.
 *
 * This function implements delay-and-sum beamforming with the following features:
 * - Dynamic aperture growth based on F-number
 * - Tukey apodization with adjustable taper width
 * - Support for both RF and IQ data
 * - Multi-frame processing
 * - Configurable interpolation (nearest neighbor, linear or quadratic)
 *
 * @tparam DataType Either float (for RF data) or std::complex<float> (for I/Q data)
 * @param channel_data Input sensor data (I/Q or RF) [n_receive_elements, n_samples, n_frames]
 * @param rx_coords_m Receive element positions [n_receive_elements, 3] (in meters)
 * @param scan_coords_m Output voxel positions [n_output_voxels, 3] (in meters)
 * @param tx_wave_arrivals_s Transmit delays for each voxel [n_output_voxels] (in seconds)
 * @param out Output beamformed data [n_output_voxels, n_frames]
 * @param f_number F-number for aperture growth control
 * @param rx_start_s Receive start time offset (seconds, corresponds to t0 in biomecardio.com/publis/ultrasonics21.pdf)
 * @param sampling_freq_hz Sampling frequency (Hz)
 * @param sound_speed_m_s Speed of sound in medium (meters/second)
 * @param modulation_freq_hz Modulation frequency (Hz)
 * @param tukey_alpha Tukey window alpha for apodization (0=no apodization, 1=full apodization)
 * @param interp_type Interpolation method for sensor data sampling
 */
template <typename DataType>
void beamform(
    nb::ndarray<const DataType, nb::ndim<3>, nb::c_contig> channel_data,
    nb::ndarray<const float, nb::shape<-1, 3>, nb::c_contig> rx_coords_m,
    nb::ndarray<const float, nb::shape<-1, 3>, nb::c_contig> scan_coords_m,
    nb::ndarray<const float, nb::ndim<1>, nb::c_contig> tx_wave_arrivals_s,
    nb::ndarray<DataType, nb::ndim<2>, nb::c_contig> out,
    float f_number,
    float rx_start_s,
    float sampling_freq_hz,
    float sound_speed_m_s,
    float modulation_freq_hz,
    float tukey_alpha,
    InterpolationType interp_type
) {
    TIME_FUNCTION();

    static_assert(std::is_same_v<DataType, float> || std::is_same_v<DataType, std::complex<float>>,
                  "DataType must be float (for RF data) or std::complex<float> (for I/Q data). "
                  "Other types like double/double2 or half/half2 would require kernel modifications.");
    // Float2 is the device-side spelling of std::complex<float>
    using KernelType = std::conditional_t<std::is_same_v<DataType, std::complex<float>>, Float2, float>;

    // Extract dimensions from arrays
    const size_t n_receive_elements = rx_coords_m.shape(0);
    const size_t n_samples = channel_data.shape(1);
    const size_t n_output_voxels = scan_coords_m.shape(0);
    const size_t n_frames = out.shape(1);

    check_dimensions(channel_data, rx_coords_m, scan_coords_m, tx_wave_arrivals_s, out,
                     n_receive_elements, n_samples, n_output_voxels, n_frames);

    sycl::queue& queue = get_queue();
    const QueueDrainGuard drain_guard{queue};

    const bool direct_channel_data = can_use_pointer_directly(channel_data, "channel_data", queue);
    const bool direct_rx_coords = can_use_pointer_directly(rx_coords_m, "rx_coords_m", queue);
    const bool direct_scan_coords = can_use_pointer_directly(scan_coords_m, "scan_coords_m", queue);
    const bool direct_tx_arrivals = can_use_pointer_directly(tx_wave_arrivals_s, "tx_wave_arrivals_s", queue);
    const bool direct_out = can_use_pointer_directly(out, "out", queue, /* allow_host_pointer = */ false);

    if (is_oneapi_array(channel_data) || is_oneapi_array(rx_coords_m) || is_oneapi_array(scan_coords_m) ||
        is_oneapi_array(tx_wave_arrivals_s) || is_oneapi_array(out)) {
        wait_for_foreign_queue();
    }

    const int staged_count = static_cast<int>(!direct_channel_data) + static_cast<int>(!direct_rx_coords) +
                             static_cast<int>(!direct_scan_coords) + static_cast<int>(!direct_tx_arrivals) +
                             static_cast<int>(!direct_out);

    if (staged_count > 0) {
        // Use PyErr_WarnEx directly instead of Python warnings module
        // This is more reliable across different binding libraries.
        // warn_or_throw respects Python's warning filter configuration (e.g. -W error).
        const std::string warning_msg =
            "Found " + std::to_string(staged_count) + " input array(s) on CPU. " +
            "This will add latency due to CPU<->GPU memory transfers. " +
            "For optimal performance with SYCL beamforming, use arrays backed by SYCL USM memory.";
        warn_or_throw(PyExc_UserWarning, warning_msg.c_str());
    }

    // Pooled device memory for the arrays that need staging. The buffers outlive
    // the call and are reused by the next one.
    KernelType* d_staged_channel_data = nullptr;
    Float3* d_staged_rx_coords_m = nullptr;
    Float3* d_staged_scan_coords_m = nullptr;
    float* d_staged_tx_arrivals_s = nullptr;
    KernelType* d_staged_out = nullptr;

    {
        TIME_SECTION("copy_cpu_arrays_to_device");
        StagingBuffers& staging = get_staging_buffers();

        if (!direct_channel_data) {
            d_staged_channel_data = static_cast<KernelType*>(staging.channel_data.acquire(channel_data.nbytes(), queue));
            queue.memcpy(d_staged_channel_data, channel_data.data(), channel_data.nbytes());
        }
        if (!direct_rx_coords) {
            d_staged_rx_coords_m = static_cast<Float3*>(staging.rx_coords_m.acquire(rx_coords_m.nbytes(), queue));
            queue.memcpy(d_staged_rx_coords_m, rx_coords_m.data(), rx_coords_m.nbytes());
        }
        if (!direct_scan_coords) {
            d_staged_scan_coords_m = static_cast<Float3*>(staging.scan_coords_m.acquire(scan_coords_m.nbytes(), queue));
            queue.memcpy(d_staged_scan_coords_m, scan_coords_m.data(), scan_coords_m.nbytes());
        }
        if (!direct_tx_arrivals) {
            d_staged_tx_arrivals_s = static_cast<float*>(staging.tx_arrivals_s.acquire(tx_wave_arrivals_s.nbytes(), queue));
            queue.memcpy(d_staged_tx_arrivals_s, tx_wave_arrivals_s.data(), tx_wave_arrivals_s.nbytes());
        }
        if (!direct_out) {
            // Pooled, so possibly dirty from an earlier call: zero it explicitly.
            d_staged_out = static_cast<KernelType*>(staging.out.acquire(out.nbytes(), queue));
            queue.memset(d_staged_out, 0, out.nbytes());
        }
    }

    {
        TIME_SECTION("call_beamform_impl");

        // If the array was staged into device memory, use that pointer;
        // otherwise the nanobind array data is already usable by the kernel.
        const KernelType* d_channel_data = d_staged_channel_data
            ? d_staged_channel_data
            : reinterpret_cast<const KernelType*>(channel_data.data());
        const Float3* d_rx_coords_m = d_staged_rx_coords_m
            ? d_staged_rx_coords_m
            : reinterpret_cast<const Float3*>(rx_coords_m.data());
        const Float3* d_scan_coords_m = d_staged_scan_coords_m
            ? d_staged_scan_coords_m
            : reinterpret_cast<const Float3*>(scan_coords_m.data());
        const float* d_tx_arrivals_s = d_staged_tx_arrivals_s
            ? d_staged_tx_arrivals_s
            : tx_wave_arrivals_s.data();
        KernelType* d_out = d_staged_out
            ? d_staged_out
            : reinterpret_cast<KernelType*>(out.data());

        _beamform_impl<KernelType>(
            queue,
            d_channel_data,
            d_rx_coords_m,
            d_scan_coords_m,
            d_tx_arrivals_s,
            d_out,
            static_cast<uint32_t>(n_receive_elements),
            static_cast<uint32_t>(n_samples),
            static_cast<uint64_t>(n_output_voxels),
            static_cast<uint32_t>(n_frames),
            f_number,
            rx_start_s,
            sampling_freq_hz,
            sound_speed_m_s,
            modulation_freq_hz,
            tukey_alpha,
            interp_type
        );
    }

    // Copy results back to the host array if we staged it
    if (d_staged_out) {
        queue.memcpy(out.data(), d_staged_out, out.nbytes());
    }

    // The single synchronization point of the call: the queue is in-order, so
    // waiting here covers the uploads, the kernel and the read-back. The host
    // arrays the copies read from are owned by the caller's nanobind handles,
    // which outlive this function.
    {
        TIME_SECTION("synchronize");
        queue.wait_and_throw();
        throw_pending_async_error();
    }
}

/**
 * @brief Beamform every transmit of a sequence and coherently compound them.
 *
 * Equivalent to calling beamform() once per transmit and summing the results,
 * but the probe geometry, the scan grid and the transmit arrival times are
 * uploaded once for the whole sequence instead of once per transmit, and the
 * accumulator stays in device memory until the last transmit has been summed
 * into it. The kernel already accumulates into its output, so compounding is
 * simply a matter of not clearing it in between.
 *
 * The queue is in-order, so a single scratch buffer can serve every transmit's
 * channel data: the upload for transmit N+1 is ordered after the kernel reading
 * transmit N. That costs the overlap a double buffer would give, but removes the
 * per-transmit host/device round trip, which is the larger of the two effects
 * for the transmit counts used in plane-wave compounding.
 *
 * @tparam DataType Either float (for RF data) or std::complex<float> (for I/Q data)
 * @param channel_data Input sensor data [n_transmits, n_receive_elements, n_samples, n_frames]
 * @param rx_coords_m Receive element positions [n_receive_elements, 3] (in meters)
 * @param scan_coords_m Output voxel positions [n_output_voxels, 3] (in meters)
 * @param tx_wave_arrivals_s Transmit delays per transmit and voxel [n_transmits, n_output_voxels]
 * @param out Compounded output [n_output_voxels, n_frames]
 * @param f_number F-number for aperture growth control
 * @param rx_start_s Receive start time offset (seconds)
 * @param sampling_freq_hz Sampling frequency (Hz)
 * @param sound_speed_m_s Speed of sound in medium (meters/second)
 * @param modulation_freq_hz Modulation frequency (Hz)
 * @param tukey_alpha Tukey window alpha for apodization (0=no apodization, 1=full apodization)
 * @param interp_type Interpolation method for sensor data sampling
 */
template <typename DataType>
void beamform_compound(
    nb::ndarray<const DataType, nb::ndim<4>, nb::c_contig> channel_data,
    nb::ndarray<const float, nb::shape<-1, 3>, nb::c_contig> rx_coords_m,
    nb::ndarray<const float, nb::shape<-1, 3>, nb::c_contig> scan_coords_m,
    nb::ndarray<const float, nb::ndim<2>, nb::c_contig> tx_wave_arrivals_s,
    nb::ndarray<DataType, nb::ndim<2>, nb::c_contig> out,
    float f_number,
    float rx_start_s,
    float sampling_freq_hz,
    float sound_speed_m_s,
    float modulation_freq_hz,
    float tukey_alpha,
    InterpolationType interp_type
) {
    TIME_FUNCTION();

    static_assert(std::is_same_v<DataType, float> || std::is_same_v<DataType, std::complex<float>>,
                  "DataType must be float (for RF data) or std::complex<float> (for I/Q data).");
    using KernelType = std::conditional_t<std::is_same_v<DataType, std::complex<float>>, Float2, float>;

    const size_t n_transmits = channel_data.shape(0);
    const size_t n_receive_elements = rx_coords_m.shape(0);
    const size_t n_samples = channel_data.shape(2);
    const size_t n_output_voxels = scan_coords_m.shape(0);
    const size_t n_frames = out.shape(1);

    if (channel_data.shape(1) != n_receive_elements) {
        throw std::runtime_error(
            "Dimension mismatch in receive elements:\n"
            "  channel_data.shape: " + shape_to_string(channel_data) + "\n" +
            "  rx_coords_m.shape: " + shape_to_string(rx_coords_m) + "\n" +
            "→ channel_data.shape[1] must equal rx_coords_m.shape[0]");
    }
    if (tx_wave_arrivals_s.shape(0) != n_transmits) {
        throw std::runtime_error(
            "Dimension mismatch in transmits:\n"
            "  channel_data.shape: " + shape_to_string(channel_data) + "\n" +
            "  tx_wave_arrivals_s.shape: " + shape_to_string(tx_wave_arrivals_s) + "\n" +
            "→ channel_data.shape[0] and tx_wave_arrivals_s.shape[0] must both equal n_transmits");
    }
    if (tx_wave_arrivals_s.shape(1) != n_output_voxels || out.shape(0) != n_output_voxels) {
        throw std::runtime_error(
            "Dimension mismatch in output voxels:\n"
            "  scan_coords_m.shape: " + shape_to_string(scan_coords_m) + "\n" +
            "  tx_wave_arrivals_s.shape: " + shape_to_string(tx_wave_arrivals_s) + "\n" +
            "  out.shape: " + shape_to_string(out) + "\n" +
            "→ tx_wave_arrivals_s.shape[1] and out.shape[0] must both equal scan_coords_m.shape[0]");
    }
    if (channel_data.shape(3) != n_frames) {
        throw std::runtime_error(
            "Dimension mismatch in frames:\n"
            "  channel_data.shape: " + shape_to_string(channel_data) + "\n" +
            "  out.shape: " + shape_to_string(out) + "\n" +
            "→ channel_data.shape[3] and out.shape[1] must both equal n_frames");
    }

    sycl::queue& queue = get_queue();
    const QueueDrainGuard drain_guard{queue};

    const bool direct_channel_data = can_use_pointer_directly(channel_data, "channel_data", queue);
    const bool direct_rx_coords = can_use_pointer_directly(rx_coords_m, "rx_coords_m", queue);
    const bool direct_scan_coords = can_use_pointer_directly(scan_coords_m, "scan_coords_m", queue);
    const bool direct_tx_arrivals = can_use_pointer_directly(tx_wave_arrivals_s, "tx_wave_arrivals_s", queue);
    const bool direct_out = can_use_pointer_directly(out, "out", queue, /* allow_host_pointer = */ false);

    if (is_oneapi_array(channel_data) || is_oneapi_array(rx_coords_m) || is_oneapi_array(scan_coords_m) ||
        is_oneapi_array(tx_wave_arrivals_s) || is_oneapi_array(out)) {
        wait_for_foreign_queue();
    }

    StagingBuffers& staging = get_staging_buffers();

    // Everything that does not vary with the transmit index is uploaded once.
    const Float3* d_rx_coords_m = reinterpret_cast<const Float3*>(rx_coords_m.data());
    const Float3* d_scan_coords_m = reinterpret_cast<const Float3*>(scan_coords_m.data());
    const float* d_tx_arrivals_s = tx_wave_arrivals_s.data();
    KernelType* d_out = reinterpret_cast<KernelType*>(out.data());

    {
        TIME_SECTION("copy_invariants_to_device");

        if (!direct_rx_coords) {
            Float3* staged = static_cast<Float3*>(staging.rx_coords_m.acquire(rx_coords_m.nbytes(), queue));
            queue.memcpy(staged, rx_coords_m.data(), rx_coords_m.nbytes());
            d_rx_coords_m = staged;
        }
        if (!direct_scan_coords) {
            Float3* staged = static_cast<Float3*>(staging.scan_coords_m.acquire(scan_coords_m.nbytes(), queue));
            queue.memcpy(staged, scan_coords_m.data(), scan_coords_m.nbytes());
            d_scan_coords_m = staged;
        }
        if (!direct_tx_arrivals) {
            float* staged = static_cast<float*>(staging.tx_arrivals_s.acquire(tx_wave_arrivals_s.nbytes(), queue));
            queue.memcpy(staged, tx_wave_arrivals_s.data(), tx_wave_arrivals_s.nbytes());
            d_tx_arrivals_s = staged;
        }
        if (!direct_out) {
            KernelType* staged = static_cast<KernelType*>(staging.out.acquire(out.nbytes(), queue));
            queue.memset(staged, 0, out.nbytes());
            d_out = staged;
        }
    }

    const size_t transmit_elements = n_receive_elements * n_samples * n_frames;
    const size_t transmit_bytes = transmit_elements * sizeof(KernelType);

    KernelType* d_channel_scratch = nullptr;
    if (!direct_channel_data && n_transmits > 0) {
        d_channel_scratch = static_cast<KernelType*>(staging.channel_data.acquire(transmit_bytes, queue));
    }

    {
        TIME_SECTION("compound_transmits");

        for (size_t transmit_idx = 0; transmit_idx < n_transmits; ++transmit_idx) {
            const KernelType* d_channel_data;
            if (direct_channel_data) {
                d_channel_data = reinterpret_cast<const KernelType*>(channel_data.data()) +
                                 transmit_idx * transmit_elements;
            } else {
                queue.memcpy(d_channel_scratch, channel_data.data() + transmit_idx * transmit_elements,
                             transmit_bytes);
                d_channel_data = d_channel_scratch;
            }

            _beamform_impl<KernelType>(
                queue,
                d_channel_data,
                d_rx_coords_m,
                d_scan_coords_m,
                d_tx_arrivals_s + transmit_idx * n_output_voxels,
                d_out,
                static_cast<uint32_t>(n_receive_elements),
                static_cast<uint32_t>(n_samples),
                static_cast<uint64_t>(n_output_voxels),
                static_cast<uint32_t>(n_frames),
                f_number,
                rx_start_s,
                sampling_freq_hz,
                sound_speed_m_s,
                modulation_freq_hz,
                tukey_alpha,
                interp_type
            );
        }
    }

    if (!direct_out) {
        queue.memcpy(out.data(), d_out, out.nbytes());
    }

    {
        TIME_SECTION("synchronize");
        queue.wait_and_throw();
        throw_pending_async_error();
    }
}

/**
 * @brief Translate SYCL's exception hierarchy into plain runtime errors for Python.
 *
 * nanobind maps std::runtime_error onto RuntimeError; an escaping sycl::exception
 * would otherwise surface as an opaque "unknown exception".
 */
template <typename DataType>
void beamform_checked(
    nb::ndarray<const DataType, nb::ndim<3>, nb::c_contig> channel_data,
    nb::ndarray<const float, nb::shape<-1, 3>, nb::c_contig> rx_coords_m,
    nb::ndarray<const float, nb::shape<-1, 3>, nb::c_contig> scan_coords_m,
    nb::ndarray<const float, nb::ndim<1>, nb::c_contig> tx_wave_arrivals_s,
    nb::ndarray<DataType, nb::ndim<2>, nb::c_contig> out,
    float f_number,
    float rx_start_s,
    float sampling_freq_hz,
    float sound_speed_m_s,
    float modulation_freq_hz,
    float tukey_alpha,
    InterpolationType interp_type
) {
    try {
        beamform<DataType>(channel_data, rx_coords_m, scan_coords_m, tx_wave_arrivals_s, out,
                           f_number, rx_start_s, sampling_freq_hz, sound_speed_m_s,
                           modulation_freq_hz, tukey_alpha, interp_type);
    } catch (const sycl::exception& error) {
        throw std::runtime_error(std::string("SYCL error: ") + error.what());
    }
}

/** @copydoc beamform_checked */
template <typename DataType>
void beamform_compound_checked(
    nb::ndarray<const DataType, nb::ndim<4>, nb::c_contig> channel_data,
    nb::ndarray<const float, nb::shape<-1, 3>, nb::c_contig> rx_coords_m,
    nb::ndarray<const float, nb::shape<-1, 3>, nb::c_contig> scan_coords_m,
    nb::ndarray<const float, nb::ndim<2>, nb::c_contig> tx_wave_arrivals_s,
    nb::ndarray<DataType, nb::ndim<2>, nb::c_contig> out,
    float f_number,
    float rx_start_s,
    float sampling_freq_hz,
    float sound_speed_m_s,
    float modulation_freq_hz,
    float tukey_alpha,
    InterpolationType interp_type
) {
    try {
        beamform_compound<DataType>(channel_data, rx_coords_m, scan_coords_m, tx_wave_arrivals_s, out,
                                    f_number, rx_start_s, sampling_freq_hz, sound_speed_m_s,
                                    modulation_freq_hz, tukey_alpha, interp_type);
    } catch (const sycl::exception& error) {
        throw std::runtime_error(std::string("SYCL error: ") + error.what());
    }
}

/**
 * @brief Hand back the device memory the staging pool is holding.
 *
 * The pool grows to the largest problem it has seen and keeps that memory for
 * the lifetime of the process. Releasing it is only worth doing when something
 * else needs the device, since the next beamform call reallocates.
 */
static void free_cached_memory() {
    get_staging_buffers().release_all();
}

/**
 * @brief Move mach onto another library's SYCL context, so its arrays work in place.
 *
 * A USM pointer is only meaningful inside the context it was allocated in, so
 * zero-copy interop with dpnp means mach has to run in dpnp's context rather
 * than one of its own. mach still builds its own in-order queue on that context
 * instead of borrowing the caller's, which is typically out-of-order: the
 * staging and compounding paths depend on in-order semantics.
 *
 * Because the two libraries then use different queues on a shared context,
 * neither one's work is automatically ordered against the other's. The Python
 * wrapper waits on the owning queue of any foreign array before launching.
 *
 * @param queue_ref Address of a dpctl DPCTLSyclQueueRef, as returned by
 *        `dpctl.SyclQueue.addressof_ref()` (reachable from a dpnp array as
 *        `array.sycl_queue`). dpctl's C API hands out a `sycl::queue*` behind
 *        that opaque handle.
 */
static void adopt_sycl_queue(std::uintptr_t queue_ref) {
    if (queue_ref == 0) {
        throw std::runtime_error("adopt_sycl_queue() was given a null queue reference.");
    }
    const sycl::queue* foreign_queue = reinterpret_cast<const sycl::queue*>(queue_ref);
    try {
        replace_queue(foreign_queue->get_context(), foreign_queue->get_device());
        // sycl::queue is a reference-counted handle, so this keeps dpctl's queue
        // alive for the synchronization in wait_for_foreign_queue()
        foreign_queue_storage() = *foreign_queue;
    } catch (const sycl::exception& error) {
        throw std::runtime_error(std::string("SYCL error while adopting a foreign queue: ") + error.what());
    }
}

/** @brief Name of the device beamforming currently runs on, or "" if there is none. */
static std::string current_device_name() {
    try {
        return get_queue().get_device().get_info<sycl::info::device::name>();
    } catch (const std::exception&) {
        return std::string();
    }
}

/**
 * @brief Resolve the device beamforming will run on, warning if it is unusable.
 *
 * @return The device name, or "" when no device could be selected.
 */
static std::string checkSyclDevice() {
    std::string name;
    std::string warning_msg;

    try {
        const sycl::device device = select_device();
        name = device.get_info<sycl::info::device::name>();
        if (!device.has(sycl::aspect::usm_device_allocations)) {
            warning_msg = "[mach] SYCL device '" + name +
                          "' does not support USM device allocations. Beamforming will fail on this device.";
        }
    } catch (const std::exception& error) {
        warning_msg = std::string("[mach] No usable SYCL device was found (") + error.what() +
                      "). Install the Intel oneAPI runtime (Level Zero or OpenCL) and make sure a device is "
                      "visible to `sycl-ls`.";
    }

    // Warn outside the catch, so an escalated warning propagates instead of
    // being mistaken for a device-selection failure
    if (!warning_msg.empty()) {
        warn_or_throw(PyExc_RuntimeWarning, warning_msg.c_str());
    }
    return name;
}

NB_MODULE(_sycl_impl, m) {
    m.doc() = "SYCL-accelerated ultrasound beamforming with nanobind";

    // Expose essential build-time version information
    m.attr("__sycl_version__") = MACH_SYCL_COMPILER_VERSION;
    m.attr("__sycl_compiler__") = MACH_SYCL_COMPILER_ID;
    m.attr("__sycl_targets__") = MACH_SYCL_TARGETS_STR;

    // Resolve the target device at import time (warns if unusable)
    m.attr("__sycl_device__") = checkSyclDevice();

    // Export InterpolationType enum to Python
    nb::enum_<InterpolationType>(m, "InterpolationType")
        .value("NearestNeighbor", InterpolationType::NearestNeighbor, "Use nearest neighbor interpolation (fastest)")
        .value("Linear", InterpolationType::Linear, "Use linear interpolation (default, good balance)")
        .value("Quadratic", InterpolationType::Quadratic, "Use quadratic interpolation (higher quality)")
        .export_values();

    // Overloaded beamform functions - nanobind automatically handles dispatch based on argument types
    m.def("beamform", &beamform_checked<std::complex<float>>,
        "channel_data"_a.noconvert(),
        "rx_coords_m"_a.noconvert(),
        "scan_coords_m"_a.noconvert(),
        "tx_wave_arrivals_s"_a.noconvert(),
        "out"_a.noconvert(),
        "f_number"_a,
        "rx_start_s"_a,
        "sampling_freq_hz"_a,
        "sound_speed_m_s"_a,
        "modulation_freq_hz"_a,
        "tukey_alpha"_a = 0.5f,
        "interp_type"_a = InterpolationType::Linear);

    m.def("beamform", &beamform_checked<float>,
        "channel_data"_a.noconvert(),
        "rx_coords_m"_a.noconvert(),
        "scan_coords_m"_a.noconvert(),
        "tx_wave_arrivals_s"_a.noconvert(),
        "out"_a.noconvert(),
        "f_number"_a,
        "rx_start_s"_a,
        "sampling_freq_hz"_a,
        "sound_speed_m_s"_a,
        "modulation_freq_hz"_a = 0.0f,
        "tukey_alpha"_a = 0.5f,
        "interp_type"_a = InterpolationType::Linear);

    // Whole-sequence entry point: one call beamforms and compounds every transmit
    // without returning to Python or to host memory in between.
    m.def("beamform_compound", &beamform_compound_checked<std::complex<float>>,
        "channel_data"_a.noconvert(),
        "rx_coords_m"_a.noconvert(),
        "scan_coords_m"_a.noconvert(),
        "tx_wave_arrivals_s"_a.noconvert(),
        "out"_a.noconvert(),
        "f_number"_a,
        "rx_start_s"_a,
        "sampling_freq_hz"_a,
        "sound_speed_m_s"_a,
        "modulation_freq_hz"_a,
        "tukey_alpha"_a = 0.5f,
        "interp_type"_a = InterpolationType::Linear);

    m.def("beamform_compound", &beamform_compound_checked<float>,
        "channel_data"_a.noconvert(),
        "rx_coords_m"_a.noconvert(),
        "scan_coords_m"_a.noconvert(),
        "tx_wave_arrivals_s"_a.noconvert(),
        "out"_a.noconvert(),
        "f_number"_a,
        "rx_start_s"_a,
        "sampling_freq_hz"_a,
        "sound_speed_m_s"_a,
        "modulation_freq_hz"_a = 0.0f,
        "tukey_alpha"_a = 0.5f,
        "interp_type"_a = InterpolationType::Linear);

    m.def("free_cached_memory", &free_cached_memory,
        "Release the device memory mach's staging pool is holding on to.");

    m.def("adopt_sycl_queue", &adopt_sycl_queue,
        "queue_ref"_a,
        "Rebuild mach's queue on the context of the given dpctl.SyclQueue.addressof_ref(), "
        "so that USM arrays from that library can be beamformed without a copy.");

    m.def("current_device_name", &current_device_name,
        "Name of the SYCL device beamforming currently runs on.");
}
