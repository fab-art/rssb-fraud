"""Utilities module for PharmaScan."""

from pharmascan.utils.normalization import (
    apply_name_normalisation,
    detect_name_clusters,
    fmt_number,
)

__all__ = [
    "fmt_number",
    "detect_name_clusters",
    "apply_name_normalisation",
]
