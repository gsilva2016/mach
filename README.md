# mach

[![PyPI](https://img.shields.io/pypi/v/mach-beamform.svg)](https://pypi.org/project/mach-beamform/)
[![Python](https://img.shields.io/pypi/pyversions/mach-beamform.svg)](https://pypi.org/project/mach-beamform/)
[![License](https://img.shields.io/github/license/Forest-Neurotech/mach.svg)](https://github.com/Forest-Neurotech/mach/blob/main/LICENSE)
[![Actions status](https://github.com/Forest-Neurotech/mach/actions/workflows/test_gpu.yml/badge.svg)](https://github.com/Forest-Neurotech/mach/actions/)

An ultrafast accelerated ultrasound beamformer for Python users. Developed at [Forest Neurotech](https://forestneurotech.org/).

![Benchmark Results](assets/benchmark-doppler_disk.svg)

_[Benchmark](https://github.com/Forest-Neurotech/mach/blob/main/BENCHMARKS.md): Beamforming PyMUST's [rotating-disk Doppler dataset](https://github.com/creatis-ULTIM/PyMUST/blob/170ba68/examples/rotatingDisk_real.ipynb) at 1.1 trillion points per second ([**6.5**x the speed of sound](https://github.com/Forest-Neurotech/mach/blob/main/BENCHMARKS.md))._

## Highlights

* ⚡ **Ultra-fast beamforming**: ~10x faster than prior state-of-the-art
* 🚀 **GPU-accelerated**: Leverages CUDA for maximum performance on NVIDIA GPUs
* :fire: **New - GPU-accelerated**: SYCL port for maximum performance on Intel architecture
* 🎯 **Optimized for research**: Designed for functional ultrasound imaging (fUSI) and other ultrafast, high-channel-count, or volumetric-ensemble imaging
* 🐍 **Python bindings**: Zero-copy integration with CuPy, and JAX arrays via [nanobind](https://nanobind.readthedocs.io/en/latest/index.html). NumPy support included.
* 🔬 **Validated**: Matches [vbeam](https://github.com/magnusdk/vbeam) and [PyMUST](https://github.com/creatis-ULTIM/PyMUST) [outputs](https://github.com/Forest-Neurotech/mach/tree/812062f/tests/compare)


## Installation

### Install from PyPI (recommended):

```bash
pip install mach-beamform
```

Or: to include all optional dependencies, including to run the examples:
```bash
pip install mach-beamform[all]
```

Wheel prerequisites:
* [Linux](https://github.com/pypa/manylinux)
* CUDA-enabled GPU with driver >= 12.3, [compute-capability >= 7.5](https://developer.nvidia.com/cuda-gpus)

### Build from source

```bash
make compile
```
Build prerequisites:
* Linux
* `make`
* `uv >= 0.9.7`
* `gcc >= 8`
* `nvcc >= 11.0`

### Build the SYCL backend (Intel GPUs and CPUs)

Alongside the CUDA kernel, mach ships a SYCL port of the same beamformer that runs on
Intel GPUs through Level Zero and on CPUs through the OpenCL runtime. [SYCL.md](SYCL.md)
documents how the CUDA and SYCL APIs map onto each other, the optimizations the port
introduced, and how to read the benchmark suite for an ultrasound workload.

```bash
source /opt/intel/oneapi/setvars.sh
make compile-sycl
```
Build prerequisites:
* [Intel oneAPI DPC++/C++ Compiler](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html) (`icpx`)
* `uv >= 0.9.7`

The two backends are configured one at a time, into separate build directories. A CMake build
tree has a single C++ compiler, and nvcc needs a host compiler it supports (`g++`) while SYCL
device code needs `icpx`, so `MACH_BUILD_CUDA=ON` together with `MACH_BUILD_SYCL=ON` is rejected:

```bash
# CUDA backend (the default build)
uv pip install -ve . --no-build-isolation -Cbuild-dir=build/cuda

# SYCL backend
uv pip install -ve . --no-build-isolation -Cbuild-dir=build/sycl \
  -Ccmake.define.MACH_BUILD_CUDA=OFF \
  -Ccmake.define.MACH_BUILD_SYCL=ON
```

Reusing one build directory for both fails, because CMake cannot change the compiler of an
existing build tree.

Run the test suite against the SYCL build with `make test-sycl` / `make benchmark-sycl`. The
plain `make test` and `make benchmark` targets let `uv run` re-sync the project, which rebuilds
it with the default CUDA options and replaces the SYCL module.

If both extension modules are present in an install, mach picks CUDA first. Override with the
`MACH_BACKEND` environment variable (`cuda` or `sycl`), and choose the SYCL device with
`MACH_SYCL_DEVICE` (`gpu`, `cpu` or `default`). `mach.__backend__` reports which one is active.

#### Build and run in Docker

[`Dockerfile.sycl`](Dockerfile.sycl) installs oneAPI DPC++ 2026 and builds the backend, so no
local oneAPI installation is needed. One definition produces both variants; the compiled
extension is identical, and they differ only in whether the Intel GPU driver stack is present.

```bash
# Intel GPU, through Level Zero. Passes /dev/dri through from the host.
docker compose run --rm --build sycl-gpu

# CPU only, through the OpenCL runtime. Runs anywhere, much smaller image.
docker compose run --rm --build sycl-cpu
```

That leaves you in a shell at `/workspace`, with the working tree mounted and `.venv` and
`build/` masked so host artifacts do not collide. Drop `--build` on later runs to reuse the
image.

Both variants preset `MACH_BACKEND=sycl` and a matching `MACH_SYCL_DEVICE`, and the entrypoint
sources `setvars.sh`, so a single command can be run and exited instead:

```bash
docker compose run --rm sycl-gpu sycl-ls
docker compose run --rm sycl-gpu python -c "import mach; print(mach.__backend__)"
docker compose run --rm sycl-gpu make test-sycl
docker compose run --rm sycl-gpu make benchmark-sycl
```

Start with `sycl-ls`: if it lists no GPU, the container reached `/dev/dri` but not a usable
driver, which is a host driver problem rather than a mach one.

Without Compose, build the variant explicitly and pass the device through yourself. The image
already contains a built copy of the project, so nothing needs mounting:

```bash
docker build -f Dockerfile.sycl --build-arg DEVICE=gpu -t mach-beamform-sycl:gpu .
docker run --rm -it --device /dev/dri mach-beamform-sycl:gpu make test-sycl
```

`--build-arg DEVICE=cpu` builds the CPU variant, which needs no `--device` flag, and
`--build-arg INSTALL_SYCL_ARRAY=0` skips dpctl and dpnp if you only want the compiled backend.
Use Compose rather than `docker run -v` if you want to edit the source live: a bind mount over
`/workspace` would hide the `.venv` that was built into the image.

#### Device arrays with dpnp

dpnp is the Intel counterpart to CuPy, with dpctl as the device/queue layer underneath it,
and `make install-sycl-array` installs both. USM memory is only valid inside the SYCL context
that allocated it, so mach joins dpnp's context at import time instead of creating one of its
own; dpnp arrays are then beamformed in place, with no staging copy.
`mach._backend.SHARES_DPCTL_CONTEXT` reports whether that worked (and `DPCTL_ERROR` why not),
and `mach._backend.device_namespace()` returns CuPy or dpnp depending on the active backend.

Use dpnp rather than `dpctl.tensor`: tensor functionality moved into dpnp, and `dpctl.tensor`
was deprecated in dpctl 0.21 and removed in 0.22.

**Both halves must be the same oneAPI release.** Joining the context means handing dpctl a raw
`sycl::queue` pointer, which is only meaningful if mach and dpnp are using the same SYCL
runtime. dpnp >= 0.20 requires `dpcpp-cpp-rt >= 2026.0`, so the SYCL backend requires an
icpx of 2026.0 or newer (`MACH_SYCL_MIN_VERSION` in `CMakeLists.txt`, lower it only if you
are not using dpnp). mach also checks at import whether two different `libsycl` images are
loaded, and declines to share the context rather than risk a crash; `DPCTL_ERROR` then says
so, and beamforming falls back to staging NumPy arrays. `MACH_ALLOW_SYCL_RUNTIME_MISMATCH=1`
overrides the check.

mach still runs its own in-order queue on that shared context, because the compounding path
depends on in-order semantics. Sharing a context says nothing about ordering, so mach waits
for dpnp's queue before launching whenever an input came from it. If you build arrays on a
non-default queue, go through `mach.kernel.beamform` rather than the raw nanobind function,
since the wrapper waits on each input's own queue.

The test suite picks its device-array library the same way, so `make test-sycl` runs the
comparison tests against dpnp where it would use CuPy on an NVIDIA machine.

#### Compounding a whole transmit sequence

The SYCL backend has a second entry point, `mach.kernel.beamform_compound`, that takes
`channel_data` with a leading transmit axis and compounds the sequence in one call. The probe
geometry, the scan grid and the arrival times are uploaded once rather than once per transmit,
the accumulator stays in device memory, and there is one host/device synchronization for the
whole sequence. `mach.experimental.beamform` uses it automatically when the active backend
provides it (`mach.kernel.SUPPORTS_COMPOUND`), and falls back to the per-transmit loop otherwise.

Staging buffers are pooled: they grow to the largest problem seen and are reused, so the USM
allocator stays off the hot path. Call `mach._backend.impl.free_cached_memory()` to hand that
memory back.

To compile ahead-of-time for a specific Intel GPU instead of JIT-ing SPIR-V at load time:

```bash
uv pip install -ve . --no-build-isolation \
  -Ccmake.define.MACH_BUILD_CUDA=OFF \
  -Ccmake.define.MACH_BUILD_SYCL=ON \
  -Ccmake.define.MACH_SYCL_TARGETS=spir64_gen \
  -Ccmake.define.MACH_SYCL_AOT_FLAGS='-Xs "-device pvc"'
```

### Docker Development

Compile and test without installing the CUDA *toolkit* using our Docker development environment.

**Prerequisites:**
* Docker Engine with [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
* CUDA-capable GPU with driver >= 12.3

**Quick start:**

```bash
# Build and start development container
docker compose run --rm dev

# Or use make shortcuts
make docker-build  # Build image (first time: ~2-3 min, rebuilds: ~30s)
make docker-dev    # Run container
```

**Inside the container:**

```bash
make compile  # Compile CUDA extension
make test     # Run tests
```

Your source code is mounted from the host, so you can edit files locally and compile in the container. Build artifacts (`.venv/` and `build/`) are stored in anonymous volumes to avoid permission issues. Dependencies are pre-installed in the image and cached, so rebuilds are fast when only source code changes.

## Examples

Try our [examples](https://forest-neurotech.github.io/mach/examples/):

* [📊 Plane Wave Imaging with PICMUS Dataset](examples/plane_wave_compound.py)
* [🩸 Doppler Imaging](examples/doppler.py)

If you don't have a CUDA-enabled GPU, you can download the notebook from the [docs](https://forest-neurotech.github.io/mach/examples/) and open in Google Colab (select a GPU instance).

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](https://github.com/Forest-Neurotech/mach/blob/812062f/CONTRIBUTING.md) for guidelines.

## Roadmap

### Beta release (v0.Y.0)
- ✅ Single-wave transmissions (plane wave, focused, diverging)
- ✅ Linear interpolation beamforming
- ✅ Allow NumPy/CuPy/JAX/PyTorch inputs through Array API
- ✅ Comprehensive error handling
- ✅ PyPI packaging and distribution
- ✅ Interpolation options: nearest, linear, and quadratic

### Numerically validated, but looking for feedback on API
- ✅ Coherent compounding

See the [project page](https://github.com/orgs/Forest-Neurotech/projects/14) for our up-to-date roadmap.
We welcome [feature requests](https://github.com/Forest-Neurotech/mach/issues)!

## Acknowledgments

mach builds upon the excellent work of the ultrasound imaging community:

- **[vbeam](https://github.com/magnusdk/vbeam)** - For educational examples and validation benchmarks
- **[PyMUST](https://github.com/creatis-ULTIM/PyMUST) / [PICMUS](https://www.creatis.insa-lyon.fr/Challenge/IEEE_IUS_2016/)** - For standardized evaluation datasets
- **Community contributors** - Gev and Qi for CUDA optimization guidance

This package was developed by the [Forest Neurotech](https://forestneurotech.org/) team, a [Focused Research Organization](https://www.convergentresearch.org/about-fros) supported by [Convergent Research](https://www.convergentresearch.org/) and [generous philanthropic funders](https://www.convergentresearch.org/fro-portfolio).

## Citation

If you use mach in your research, you can cite:

```bibtex
@article{mach,
  title = {mach: ultrafast ultrasound beamforming},
  author = {Guan, Charles and Rockhill, Alexander P and Sode, Masashi and Pinton, Gianmarco},
  year = 2026,
  journal = {Journal of Medical Imaging},
  publisher = {Society of Photo-Optical Instrumentation Engineers},
  volume = 13,
  number = 6,
  pages = {062203--062203},
  doi = {10.1117/1.JMI.13.6.062203},
  url = {https://github.com/Forest-Neurotech/mach}
}
```
