"""torch-inductor-duplicate-index-writeorder-guard: version and package marker."""
__version__ = "0.1.0"

from .core import safe_dup_index_assign, make_safe_dup_index_assign, diagnose  # noqa: F401
