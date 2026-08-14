"""Compatibility guard for the retired Python expert pager.

Expert paging now lives in the native ``sytra-engine`` crate, where immutable
byte ranges, RAM/accelerator leases, eviction, and prefetch can be coordinated
without Python or Torch in the decode path.
"""
from __future__ import annotations


class GPUExpertPager:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "GPUExpertPager moved to Sytra's native Rust runtime. Build `sytra-engine` "
            "and create a .sytra-runtime.json expert index instead of using the old "
            "Python/Torch prototype."
        )
