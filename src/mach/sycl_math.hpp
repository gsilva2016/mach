#pragma once

/*
 * Minimal vector types and math shims for the SYCL beamforming kernel.
 *
 * This replaces the CUDA-only `helper_math.h` / `vector_types.h` used by
 * `kernel.cu`. The types deliberately do not use `sycl::vec`: `sycl::float3`
 * occupies 16 bytes (it is padded out to a `float4`), so it cannot alias the
 * tightly packed (N, 3) float32 arrays that callers hand us.
 */

#include <sycl/sycl.hpp>

#include <complex>
#include <cstddef>

namespace mach {

/** @brief Packed 3-component vector, layout-compatible with CUDA's `float3`. */
struct Float3 {
    float x;
    float y;
    float z;
};

/** @brief 2-component vector, layout-compatible with CUDA's `float2` and `std::complex<float>`. */
struct alignas(8) Float2 {
    float x;
    float y;
};

static_assert(sizeof(Float3) == 3 * sizeof(float), "Float3 must be tightly packed to alias (N, 3) float32 arrays");
static_assert(sizeof(Float2) == sizeof(std::complex<float>), "Float2 must alias std::complex<float>");

inline Float2 make_float2(float x, float y) {
    return Float2{x, y};
}

inline Float2 operator+(Float2 a, Float2 b) {
    return Float2{a.x + b.x, a.y + b.y};
}

inline Float2 operator-(Float2 a, Float2 b) {
    return Float2{a.x - b.x, a.y - b.y};
}

inline Float2 operator*(Float2 a, float s) {
    return Float2{a.x * s, a.y * s};
}

inline Float2 operator*(float s, Float2 a) {
    return Float2{a.x * s, a.y * s};
}

inline Float2& operator+=(Float2& a, Float2 b) {
    a.x += b.x;
    a.y += b.y;
    return a;
}

inline Float2& operator*=(Float2& a, float s) {
    a.x *= s;
    a.y *= s;
    return a;
}

inline float lerp(float a, float b, float t) {
    return a + t * (b - a);
}

inline Float2 lerp(Float2 a, Float2 b, float t) {
    return a + (b - a) * t;
}

/** @brief Simultaneous sine and cosine, replacing CUDA's `__sincosf`. */
inline float sincos(float x, float& cos_out) {
    return sycl::sincos(x, sycl::address_space_cast<sycl::access::address_space::generic_space,
                                                    sycl::access::decorated::no>(&cos_out));
}

/** @brief Round-to-nearest float to unsigned conversion, replacing CUDA's `__float2uint_rn`. */
inline unsigned int float_to_uint_rn(float x) {
    return static_cast<unsigned int>(sycl::rint(x));
}

/** @brief Round-down float to unsigned conversion, replacing CUDA's `__float2uint_rd`. */
inline unsigned int float_to_uint_rd(float x) {
    return static_cast<unsigned int>(sycl::floor(x));
}

/** @brief Round-up float to unsigned conversion, replacing CUDA's `__float2uint_ru`. */
inline unsigned int float_to_uint_ru(float x) {
    return static_cast<unsigned int>(sycl::ceil(x));
}

}  // namespace mach
