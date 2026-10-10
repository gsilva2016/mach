"""Tests for the SYCL runtime compatibility guard in mach._dpctl.

mach joins dpnp's SYCL context by handing dpctl a raw ``sycl::queue`` pointer,
which is only valid if both sides are the same oneAPI runtime. These exercise the
checks that refuse the handoff otherwise, without needing dpnp or a device.
"""

import types

import pytest

from mach import _dpctl

# Pure logic, no device involved
pytestmark = pytest.mark.no_cuda


def test_adopt_context_reports_a_backend_without_the_entry_point():
    """An extension built before the interop support says so, rather than failing oddly."""
    reason = _dpctl.adopt_context(types.SimpleNamespace())
    assert reason is not None
    assert "adopt_sycl_queue" in reason


def test_two_loaded_libsycl_images_block_the_handoff(monkeypatch):
    """Two SYCL runtimes in one process is the case the queue pointer cannot survive."""
    monkeypatch.setattr(
        _dpctl,
        "_loaded_sycl_runtimes",
        lambda: {"/opt/intel/oneapi/lib/libsycl.so.8", "/venv/site-packages/dpcpp/libsycl.so.9"},
    )

    reason = _dpctl._runtime_mismatch(types.SimpleNamespace())

    assert reason is not None
    assert "different SYCL runtimes" in reason


def test_mismatched_oneapi_years_block_the_handoff(monkeypatch):
    """Without procfs, the comparison falls back to the oneAPI release year."""
    monkeypatch.setattr(_dpctl, "_loaded_sycl_runtimes", set)
    monkeypatch.setattr(_dpctl, "version", lambda _name: "2025.2.0")

    reason = _dpctl._runtime_mismatch(types.SimpleNamespace(__sycl_version__="2026.0.0"))

    assert reason is not None
    assert "2025.2.0" in reason
    assert "2026.0.0" in reason


def test_matching_oneapi_years_are_allowed(monkeypatch):
    """Patch releases within the same year are compatible."""
    monkeypatch.setattr(_dpctl, "_loaded_sycl_runtimes", set)
    monkeypatch.setattr(_dpctl, "version", lambda _name: "2026.1.0")

    assert _dpctl._runtime_mismatch(types.SimpleNamespace(__sycl_version__="2026.0.0")) is None


def test_runtime_from_setvars_is_not_second_guessed(monkeypatch):
    """With no pip-installed dpcpp-cpp-rt there is no version to compare against."""

    def not_installed(_name):
        raise _dpctl.PackageNotFoundError(_name)

    monkeypatch.setattr(_dpctl, "_loaded_sycl_runtimes", set)
    monkeypatch.setattr(_dpctl, "version", not_installed)

    assert _dpctl._runtime_mismatch(types.SimpleNamespace(__sycl_version__="2025.2.0")) is None
