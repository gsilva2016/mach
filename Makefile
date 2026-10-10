SHELL := /bin/bash

.PHONY: help
help: ## Displays help information about available make commands
	@if command -v uv &> /dev/null; then \
		uv run --no-project python -c "import re; \
		[[print(f'\033[36m{m[0]:<20}\033[0m {m[1]}') for m in re.findall(r'^([a-zA-Z_-]+):.*?## (.*)$$', open('$(MAKEFILE_LIST)').read(), re.M)] for makefile in ('$(MAKEFILE_LIST)').strip().split()]"; \
	else \
		echo "Available commands include: help, compile, test..."; \
		echo "Run 'make install-uv' to install uv."; \
		echo "Then run 'make help' again to see docs on available commands."; \
	fi

.PHONY: check-system-dep
check-system-dep: ## Checks if system dependencies are installed (nvcc) based on OS
	@if ! command -v nvcc &> /dev/null; then \
		echo "nvcc not found. Please install CUDA and set up your environment."; \
		exit 1; \
	fi
	@if ! command -v uv &> /dev/null; then \
		echo "uv not found.  Please install uv with 'make install-uv'"; \
		exit 1; \
	fi

.PHONY: check-system-dep-sycl
check-system-dep-sycl: ## Checks if system dependencies are installed (icpx >= 2026)
	@if ! command -v icpx &> /dev/null; then \
		echo "icpx not found. Install the Intel oneAPI DPC++/C++ Compiler and run 'source /opt/intel/oneapi/setvars.sh'."; \
		exit 1; \
	fi
	# dpnp bundles the 2026 SYCL runtime and mach passes a sycl::queue pointer across
	# that boundary, so an older compiler here would pair two incompatible runtimes.
	@icpx_year=$$(icpx --version | head -1 | grep -oE '[0-9]{4}\.[0-9]+' | head -1 | cut -d. -f1); \
	if [ -n "$$icpx_year" ] && [ "$$icpx_year" -lt 2026 ]; then \
		echo "icpx is oneAPI $$icpx_year, but mach targets 2026 so that dpnp interop is safe."; \
		echo "Source a 2026 environment, or build without the floor:"; \
		echo "  -Ccmake.define.MACH_SYCL_MIN_VERSION=$$icpx_year.0"; \
		exit 1; \
	fi
	@if ! command -v uv &> /dev/null; then \
		echo "uv not found.  Please install uv with 'make install-uv'"; \
		exit 1; \
	fi

.PHONY: install-uv
install-uv: ## Installs uv package manger
	@if ! command -v uv &> /dev/null; then \
		echo "uv not found. Installing uv..."; \
		curl -LsSf https://astral.sh/uv/0.9.7/install.sh | sh; \
		echo "please follow the uv suggestions to update or restart your shell environment"; \
	fi
	@uv_version=$$(uv --version | awk '{print $$2}'); \
	if [ "$$(printf '%s\n' "0.9.7" "$$uv_version" | sort -V | head -n1)" != "0.9.7" ]; then \
		echo "Updating uv to latest version..."; \
		uv self update 0.9.7; \
	fi

.PHONY: install-python-dep
install-python-dep: ## Installs the Python dependencies
	uv sync

.PHONY: compile
compile: check-system-dep ## Compiles the CUDA extension with nanobind
	@echo "Compiling CUDA extension with nanobind..."
	@echo "If you get an scikit-build-core error, you may need to 'uv cache clean' and 'trash build/'"
	uv pip install scikit-build-core nanobind ninja cmake
	uv sync --group build
	# Not sure if the pip command is also needed
	uv pip install -ve . --no-build-isolation

.PHONY: compile-sycl
compile-sycl: check-system-dep-sycl ## Compiles the SYCL/DPC++ extension with nanobind
	@echo "Compiling SYCL extension with nanobind..."
	@echo "If you get an scikit-build-core error, you may need to 'uv cache clean' and 'trash build/'"
	uv pip install scikit-build-core nanobind ninja cmake
	# --no-install-project because a plain 'uv sync' builds mach with the default
	# CMake options, which turn CUDA on and fail without nvcc. The pip install
	# below is what actually builds the project, with the SYCL options.
	# --inexact so that the sync does not prune packages that are not in the
	# lockfile, which would silently uninstall dpctl/dpnp from 'make install-sycl-array'.
	uv sync --inexact --no-install-project --group build
	# Separate build-dir from the CUDA build: CMake cannot change the compiler
	# of an existing build tree, and this one needs icpx rather than g++.
	uv pip install -ve . --no-build-isolation \
		-Cbuild-dir=build/sycl \
		-Ccmake.define.MACH_BUILD_CUDA=OFF \
		-Ccmake.define.MACH_BUILD_SYCL=ON
	@echo "✓ SYCL backend installed. Use 'make test-sycl' / 'make benchmark-sycl':"
	@echo "  a plain 'uv run' re-syncs the project and rebuilds it with the default"
	@echo "  CUDA options, which replaces the module just installed."

.PHONY: stubgen
stubgen: ## Generates the type-hint stub file for the nanobind-CUDA module
	uv run --group build -m nanobind.stubgen -m mach._cuda_impl -O src/mach/

.PHONY: stubgen-sycl
stubgen-sycl: ## Generates the type-hint stub file for the nanobind-SYCL module
	uv run --group build -m nanobind.stubgen -m mach._sycl_impl -O src/mach/

.PHONY: check
check: ## Checks the code
	@echo "🚀 Checking lock file consistency with 'pyproject.toml'"
	@uv lock --locked
	@echo "🚀 Linting code: Running pre-commit"
	@uv run pre-commit run -a
	@echo "🚀 Type checking: Running ty"
	@uv run ty check src
	@echo "🚀 Checking for obsolete dependencies: Running deptry"
	@uv run deptry .
	@echo "🚀 Checking marimo notebooks"
	@uv run marimo check marimo

.PHONY: test
test: ## Runs Python tests
	@echo "🚀 Running tests"
	uv run --group test --group array --group compare pytest tests -v -s --benchmark-disable --save-output

.PHONY: test-fail
test-fail: ## Runs Python tests that failed, and drop into debugger on failure
	@echo "🚀 Running tests"
	uv run --group test --group array --group compare pytest tests -v -s --benchmark-disable --save-output --pdb --lf

.PHONY: benchmark
benchmark: ## Runs benchmarking comparisons
	@echo "🚀 Running benchmarking comparisons"
	uv run --group test --group array --group compare pytest tests -v -s --benchmark-only --benchmark-histogram --benchmark-autosave --benchmark-save-data

.PHONY: install-sycl-array
install-sycl-array: ## Installs dpctl/dpnp, the device-array libraries for the SYCL backend
	@echo "🚀 Installing dpctl and dpnp"
	uv pip install --group sycl-array
	@echo "🚀 Checking that mach can join dpnp's SYCL context"
	uv run --no-sync python -c \
		"import mach._backend as b; print('shares dpnp SYCL context:', b.SHARES_DPCTL_CONTEXT, '|', b.DPCTL_ERROR or 'ok')"

# The SYCL targets below pass --no-sync on purpose. 'uv run' otherwise re-syncs the
# project, which rebuilds mach with the default CMake options (CUDA on, SYCL off)
# and replaces the module that 'make compile-sycl' installed.
# tests/compare is included: its device arrays now come from whichever library
# matches the active backend (dpnp here), and the vbeam tests skip themselves
# when vbeam/JAX are absent.
.PHONY: test-sycl
test-sycl: ## Runs Python tests against the SYCL backend
	@echo "🚀 Running tests on the SYCL backend"
	MACH_BACKEND=sycl uv run --no-sync pytest tests -v -s --benchmark-disable --save-output

.PHONY: benchmark-sycl
benchmark-sycl: ## Runs benchmarks against the SYCL backend
	@echo "🚀 Benchmarking the SYCL backend"
	MACH_BACKEND=sycl uv run --no-sync pytest tests -v -s --benchmark-only --benchmark-histogram --benchmark-autosave --benchmark-save-data

.PHONY: profile
profile: ## Runs Python test with simple profiling. Recommend using Nsight Compute or Nsight Systems for more detailed profiling.
	@echo "Building with CUDA_PROFILE"
	uv pip install --no-build-isolation -ve . -Ccmake.define.CMAKE_CUDA_FLAGS_INIT="-DCUDA_PROFILE"
	@echo "🚀 Running tests"
	uv run --group profile pyinstrument --timeline -m pytest tests/test_beamform.py -v -s --tile-total-frames 200

.PHONY: docs
docs: ## Build the documentation
	@echo "🚀 Building documentation"
	uv run $(MAKE) -C docs html;

.PHONY: docs-open
docs-open: docs ## Build and open the documentation
	@if [ -n "$$SSH_CONNECTION" ]; then \
		@echo "🚀 Serving documentation"; \
		uv run python -m http.server --directory docs/_build/html 8000; \
	else \
		@echo "🚀 Opening documentation in browser"; \
		uv run python -c "import webbrowser; webbrowser.open_new_tab('file://$(PWD)/docs/_build/html/index.html')"; \
	fi

.PHONY: wheel
wheel: ## Builds a wheel
	uv build

.PHONY: clean
clean: ## Cleans build artifacts
	@echo "Cleaning build artifacts..."
	rm -rf dist/ build/ mach.*.so

.PHONY: docker-build
docker-build: ## Builds the Docker development image
	docker compose build

.PHONY: docker-dev
docker-dev: ## Runs the development container
	docker compose run --rm dev

.DEFAULT_GOAL := help
