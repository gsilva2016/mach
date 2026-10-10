cmake_minimum_required(VERSION 3.18...3.27)
# Override the project name to "mach" so we can import mach, even though the wheel is called "mach-beamform"
set(SKBUILD_PROJECT_NAME "mach")

# Backend selection. This has to happen before project(), because it decides
# which languages and which C++ compiler CMake configures.
#   -Ccmake.define.MACH_BUILD_CUDA=OFF -Ccmake.define.MACH_BUILD_SYCL=ON
option(MACH_BUILD_CUDA "Build the CUDA backend (mach._cuda_impl)" ON)
option(MACH_BUILD_SYCL "Build the SYCL/DPC++ backend (mach._sycl_impl)" OFF)

if(NOT MACH_BUILD_CUDA AND NOT MACH_BUILD_SYCL)
    message(FATAL_ERROR "At least one of MACH_BUILD_CUDA or MACH_BUILD_SYCL must be ON.")
endif()

# A CMake build tree has exactly one CXX compiler, but nvcc needs a host compiler
# it supports (g++) while SYCL device code needs icpx. So the two backends are
# configured separately, each into its own build directory.
if(MACH_BUILD_CUDA AND MACH_BUILD_SYCL)
    message(FATAL_ERROR
        "MACH_BUILD_CUDA and MACH_BUILD_SYCL cannot be built in the same build tree, because nvcc "
        "requires a host compiler such as g++ while SYCL device code requires icpx.\n"
        "Build them one at a time, into separate build directories:\n"
        "  pip install --no-build-isolation -ve . -Cbuild-dir=build/cuda\n"
        "  pip install --no-build-isolation -ve . -Cbuild-dir=build/sycl "
        "-Ccmake.define.MACH_BUILD_CUDA=OFF -Ccmake.define.MACH_BUILD_SYCL=ON")
endif()

if(NOT DEFINED CMAKE_CXX_COMPILER)
    if(MACH_BUILD_SYCL)
        # SYCL device code needs a SYCL compiler; oneAPI's is icpx
        find_program(MACH_SYCL_CXX_COMPILER NAMES icpx
                     DOC "Intel oneAPI DPC++ compiler used to build the SYCL backend")
        if(NOT MACH_SYCL_CXX_COMPILER)
            message(FATAL_ERROR
                "MACH_BUILD_SYCL=ON but 'icpx' was not found on PATH. Source the oneAPI environment "
                "first (e.g. 'source /opt/intel/oneapi/setvars.sh'), or pass "
                "-Ccmake.define.CMAKE_CXX_COMPILER=/path/to/icpx.")
        endif()
        set(CMAKE_CXX_COMPILER "${MACH_SYCL_CXX_COMPILER}")
    else()
        # Prefer g++ compiler for C++ code
        set(CMAKE_CXX_COMPILER g++)
    endif()
endif()

set(MACH_LANGUAGES CXX)
if(MACH_BUILD_CUDA)
    list(APPEND MACH_LANGUAGES CUDA)
endif()
project(${SKBUILD_PROJECT_NAME} LANGUAGES ${MACH_LANGUAGES})

# https://github.com/wjakob/nanobind_example/blob/1c2a122/CMakeLists.txt
if (NOT SKBUILD)
  message(WARNING "\
  This CMake file is meant to be executed using 'scikit-build'. Running
  it directly will almost certainly not produce the desired result. If
  you are a user trying to install this package, please use the command
  below, which will install all necessary build dependencies, compile
  the package in an isolated environment, and then install it.
  =====================================================================
    pip install .
  =====================================================================
  If you are a software developer, and this is your own package, then
  it is usually much more efficient to install the build dependencies
  in your environment once and use the following command that avoids
  a costly creation of a new virtual environment at every compilation:
  =====================================================================
    pip install nanobind scikit-build-core[pyproject]
    pip install --no-build-isolation -ve .
  =====================================================================
  You may optionally add -Ceditable.rebuild=true to auto-rebuild when
  the package is imported. Otherwise, you need to re-run the above
  after editing C++ files.")
endif()


# search for Python >= 3.11 including the Development.Module component required by nanobind
find_package(Python 3.11 COMPONENTS Interpreter Development.Module REQUIRED)

# set the build type to Release by default
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# find the nanobind package
find_package(nanobind CONFIG REQUIRED)

# CUDA requires C++11 or newer; SYCL requires C++17
if(MACH_BUILD_SYCL)
    set(CMAKE_CXX_STANDARD 17)
else()
    set(CMAKE_CXX_STANDARD 11)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# nanobind requires GCC 8+
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 8.0)
        message(FATAL_ERROR "g++ version must be at least 8.0 for nanobind!")
    endif()
endif()

# Add vendor directory to include paths
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/${SKBUILD_PROJECT_NAME}/vendor)

# ---------------------------------------------------------------------------
# CUDA backend
# ---------------------------------------------------------------------------
if(MACH_BUILD_CUDA)
    # Set CUDA compilation flags
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS_INIT} ${CMAKE_CUDA_FLAGS}")
    # includes details about register usage and shared memory for each kernel.
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")
    # debug info
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --generate-line-info")
    # Build native SASS for every GPU we deploy on. Shipping a native cubin per
    # arch avoids relying on PTX JIT at load time -- a JIT only happens when no
    # cubin matches the device, and it requires a driver at least as new as the
    # build toolkit.
    set(CMAKE_CUDA_ARCHITECTURES 75 89 90)

    # Check minimum CUDA compiler version
    set(NVCC_MIN_VERSION 11.0)
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS ${NVCC_MIN_VERSION})
        message(FATAL_ERROR "NVCC (CUDA compiler) version must be at least ${NVCC_MIN_VERSION}!")
    endif()

    # Parse NVCC version to get major.minor components
    set(NVCC_VERSION_STR "${CMAKE_CUDA_COMPILER_VERSION}")
    string(REGEX MATCH "^([0-9]+)\\.([0-9]+)" _ "${NVCC_VERSION_STR}")
    set(NVCC_MAJOR ${CMAKE_MATCH_1})
    set(NVCC_MINOR ${CMAKE_MATCH_2})

    message(STATUS "Parsed NVCC version: major=${NVCC_MAJOR}, minor=${NVCC_MINOR}")

    # Validate parsing worked
    if(NOT DEFINED NVCC_MAJOR OR NOT DEFINED NVCC_MINOR)
        message(FATAL_ERROR "Failed to parse NVCC version '${NVCC_VERSION_STR}'. Expected format: X.Y")
    endif()

    # Build the nanobind module with the CUDA kernel
    nanobind_add_module(
      # Name of the extension
      _cuda_impl

      # Build libnanobind statically and merge it into the
      # extension (which itself remains a shared library)
      NB_STATIC

      # Source code
      src/${SKBUILD_PROJECT_NAME}/kernel.cu
    )

    # Pass CUDA version information as compiler definitions
    target_compile_definitions(_cuda_impl PRIVATE
        NVCC_VERSION_STR="${NVCC_VERSION_STR}"
        NVCC_MAJOR=${NVCC_MAJOR}
        NVCC_MINOR=${NVCC_MINOR}
    )

    # install the CUDA module
    install(TARGETS _cuda_impl LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME})
endif()

# ---------------------------------------------------------------------------
# SYCL / Intel oneAPI DPC++ backend
# ---------------------------------------------------------------------------
if(MACH_BUILD_SYCL)
    if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM" AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        message(FATAL_ERROR
            "MACH_BUILD_SYCL requires a SYCL compiler (Intel oneAPI DPC++ 'icpx', or SYCL-enabled Clang), "
            "but this build tree is configured with '${CMAKE_CXX_COMPILER_ID}' at '${CMAKE_CXX_COMPILER}'.\n"
            "If icpx is on PATH, the cause is usually a CMake cache left over from a CUDA build in "
            "'${CMAKE_BINARY_DIR}': CMake cannot change the compiler of an existing build tree. "
            "Configure the SYCL backend into its own build directory:\n"
            "  pip install --no-build-isolation -ve . -Cbuild-dir=build/sycl "
            "-Ccmake.define.MACH_BUILD_CUDA=OFF -Ccmake.define.MACH_BUILD_SYCL=ON\n"
            "Otherwise source the oneAPI environment first ('source /opt/intel/oneapi/setvars.sh').")
    endif()

    # mach hands dpctl a raw sycl::queue pointer for zero-copy dpnp interop, so the
    # compiler used here and the runtime dpnp loads have to be the same oneAPI
    # release. dpnp >= 0.20 requires dpcpp-cpp-rt >= 2026.0, hence the floor.
    set(MACH_SYCL_MIN_VERSION "2026.0" CACHE STRING
        "Minimum Intel oneAPI DPC++ version. Lower it only if you are not using dpnp.")
    if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM"
       AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "${MACH_SYCL_MIN_VERSION}")
        message(FATAL_ERROR
            "mach requires Intel oneAPI DPC++ >= ${MACH_SYCL_MIN_VERSION}, but '${CMAKE_CXX_COMPILER}' "
            "is ${CMAKE_CXX_COMPILER_VERSION}.\n"
            "dpnp's wheels bundle the 2026 SYCL runtime, and mach passes a sycl::queue pointer "
            "across that boundary, so an older compiler here would pair two incompatible runtimes.\n"
            "Source a 2026 oneAPI environment ('source /opt/intel/oneapi/setvars.sh'), or if you do "
            "not need dpnp interop, lower the floor with "
            "-Ccmake.define.MACH_SYCL_MIN_VERSION=${CMAKE_CXX_COMPILER_VERSION}.")
    endif()

    # spir64 is the JIT SPIR-V target: it runs on Intel GPUs through Level Zero
    # and on CPUs through the OpenCL runtime, from a single binary. Set this to
    # e.g. "spir64_gen" plus MACH_SYCL_AOT_FLAGS for ahead-of-time Intel GPU builds.
    set(MACH_SYCL_TARGETS "spir64" CACHE STRING "Value passed to -fsycl-targets")
    set(MACH_SYCL_AOT_FLAGS "" CACHE STRING "Extra backend flags, e.g. -Xs \"-device arc\"")

    separate_arguments(MACH_SYCL_AOT_FLAGS_LIST NATIVE_COMMAND "${MACH_SYCL_AOT_FLAGS}")
    set(MACH_SYCL_FLAGS -fsycl -fsycl-targets=${MACH_SYCL_TARGETS} ${MACH_SYCL_AOT_FLAGS_LIST})

    message(STATUS "Building SYCL backend with ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}, targets: ${MACH_SYCL_TARGETS}")

    nanobind_add_module(
      # Name of the extension
      _sycl_impl

      # Build libnanobind statically and merge it into the
      # extension (which itself remains a shared library)
      NB_STATIC

      # nanobind otherwise compiles C++ sources with -Os, which would also apply
      # to the SYCL device code. The CUDA kernel is unaffected because -Os is
      # scoped to COMPILE_LANGUAGE:CXX.
      NOMINSIZE

      # Source code
      src/${SKBUILD_PROJECT_NAME}/kernel_sycl.cpp
    )

    target_compile_features(_sycl_impl PRIVATE cxx_std_17)
    target_compile_options(_sycl_impl PRIVATE ${MACH_SYCL_FLAGS})
    target_link_options(_sycl_impl PRIVATE ${MACH_SYCL_FLAGS})

    # Pass SYCL toolchain information as compiler definitions
    target_compile_definitions(_sycl_impl PRIVATE
        MACH_SYCL_COMPILER_ID="${CMAKE_CXX_COMPILER_ID}"
        MACH_SYCL_COMPILER_VERSION="${CMAKE_CXX_COMPILER_VERSION}"
        MACH_SYCL_TARGETS_STR="${MACH_SYCL_TARGETS}"
    )

    # install the SYCL module
    install(TARGETS _sycl_impl LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME})
endif()
