"""
AdaptiveCAD core interface stubs for shape definition and UI controls.
This module provides decorators and functions to register adaptive shapes
and define interactive sliders and dropdowns in MCP or GUI contexts.
"""
from typing import Optional

def adaptive_shape(name: str):
    """Decorator to register an adaptive shape under a given name."""
    def decorator(func):
        # Attach metadata for shape registry
        setattr(func, "_adaptive_shape_name", name)
        return func
    return decorator


def slider(name: str, min_value: float, max_value: float, step: float, label: Optional[str] = None):
    """Define a slider control metadata placeholder."""
    # In MCP or GUI frameworks, this would register a slider UI element.
    return None


def dropdown(name: str, options: list[str], label: Optional[str] = None):
    """Define a dropdown control metadata placeholder."""
    # In MCP or GUI frameworks, this would register a dropdown UI element.
    return None
