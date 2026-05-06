"""
Module loader with backward compatibility for old model paths.

This module provides a mechanism to resolve module paths when loading
PyTorch models, supporting fallback from old paths (e.g., models.yolo)
to new paths (e.g., detector.models.yolo).

Usage:
    from detector.utils.module_loader import torch_load, register_module_mapping
    
    # Load model with automatic path remapping
    checkpoint = torch_load('model.pt', map_location='cpu')
    
    # Register custom mapping for future similar issues
    register_module_mapping('oldlib', 'newpackage.oldlib')
"""

import importlib
import logging
import sys
import warnings
from typing import Dict, Any, Optional
import torch


# Registry of module path prefix mappings
# Key: old prefix, Value: new prefix
_MODULE_PREFIX_MAPPINGS: Dict[str, str] = {
    'models': 'detector.models'
}


def register_module_mapping(old_prefix: str, new_prefix: str) -> None:
    """
    Register a module path prefix mapping.
    
    Args:
        old_prefix: The old module path prefix (e.g., 'models')
        new_prefix: The new module path prefix (e.g., 'detector.models')
    
    Example:
        register_module_mapping('models', 'detector.models')
        register_module_mapping('mylib', 'newpackage.mylib')
    """
    _MODULE_PREFIX_MAPPINGS[old_prefix] = new_prefix
    # Only initialize the new mapping, not all aliases
    try:
        if old_prefix not in sys.modules:
            new_module = importlib.import_module(new_prefix)
            sys.modules[old_prefix] = new_module
            logging.debug(f"Created alias {old_prefix} -> {new_prefix}")
    except (ModuleNotFoundError, ImportError) as e:
        logging.warning(f"Failed to create alias {old_prefix} -> {new_prefix}: {e}")


def get_module_mappings() -> Dict[str, str]:
    """
    Get a copy of the current module prefix mappings.
    
    Returns:
        Dictionary of old_prefix -> new_prefix mappings
    """
    return _MODULE_PREFIX_MAPPINGS.copy()


def _initialize_module_aliases() -> None:
    """
    Initialize module aliases in sys.modules for backward compatibility.
    
    This creates aliases for old module paths (e.g., 'models') that point
    to the new module implementations (e.g., 'detector.models').
    
    This is called once at module load time and whenever a new mapping
    is registered.
    """
    for old_prefix, new_prefix in _MODULE_PREFIX_MAPPINGS.items():
        if old_prefix in sys.modules:
            # Skip if already exists - could be user-defined or previously patched
            logging.debug(f"Skipping alias {old_prefix} -> {new_prefix}: already exists")
            continue
        
        try:
            # Import the new prefix module
            new_module = importlib.import_module(new_prefix)
            sys.modules[old_prefix] = new_module
            logging.debug(f"Created alias {old_prefix} -> {new_prefix}")
        except (ModuleNotFoundError, ImportError) as e:
            # If the new module doesn't exist yet, skip this mapping
            # This can happen during early initialization
            logging.warning(f"Failed to create alias {old_prefix} -> {new_prefix}: {e}")


# Initialize module aliases at module load time



def torch_load(f, map_location: Optional[Any] = None, **kwargs) -> Any:
    """
    Load a PyTorch model with backward-compatible module resolution.
    
    This wrapper uses a pre-patching approach:
    1. Module aliases are set up at module load time (e.g., 'models' -> 'detector.models')
    2. torch.load() is called normally - pickle finds the patched modules
    3. No cleanup needed as aliases remain for the process lifetime
    
    This approach is compatible with PyTorch 2.0 and all earlier versions
    that don't support the module_resolver parameter.
    
    Args:
        f: File path or file-like object
        map_location: Device to map the tensor to (e.g., 'cpu', 'cuda:0')
        **kwargs: Additional keyword arguments passed to torch.load
    
    Returns:
        The loaded model/checkpoint
    
    Example:
        from detector.utils.module_loader import torch_load
        
        # Load checkpoint with automatic path remapping
        checkpoint = torch_load('model.pt', map_location='cpu')
        
        # Load model only
        model_data = torch_load('best.pt', map_location='cuda:0')
    """

    _initialize_module_aliases()
    return torch.load(f, map_location=map_location, weights_only=False, **kwargs)