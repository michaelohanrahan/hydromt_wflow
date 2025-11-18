"""Redirect to the actual hydromt_wflow package in the subdirectory."""

import sys
from pathlib import Path
import importlib.util

# Get the path to the actual package
_package_dir = Path(__file__).parent / "hydromt_wflow"
_actual_init = _package_dir / "__init__.py"

if _actual_init.exists():
    # Load the actual package's __init__.py
    spec = importlib.util.spec_from_file_location("hydromt_wflow._actual", _actual_init)
    if spec and spec.loader:
        _actual_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_actual_module)
        
        # Copy all exports to this module
        for name in getattr(_actual_module, '__all__', []):
            setattr(sys.modules[__name__], name, getattr(_actual_module, name))
        
        # Also copy version
        if hasattr(_actual_module, '__version__'):
            __version__ = _actual_module.__version__
        
        __all__ = getattr(_actual_module, '__all__', [])
