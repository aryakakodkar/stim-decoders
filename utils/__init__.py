"""Utility subpackage init.

Keep this file minimal to avoid importing heavyweight submodules at import time
which can cause circular import problems. Import submodules directly where
they are needed (e.g. `from stimdecoders.utils import codes`).
"""

from .bitops import *

# Do NOT import other submodules here to avoid circular imports. Python's
# `from package import submodule` will import the submodule automatically when
# requested, so direct imports here are unnecessary and may trigger cycles.

__all__ = [
	'bitops',
	'codes',
	'circuits',
	'noise',
]