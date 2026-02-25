import os
import re


def extract_direction_prefix(path: str) -> str:
    """Extract the SOM direction prefix from a file path (e.g., 'centroid_to_som4_sigma0.3_layer13')."""
    m = re.search(r"((?:\w+_)?som[^\s/]+_layer\d{1,2})", os.path.basename(path))
    if not m:
        raise ValueError(f"Cannot parse prefix from {path}")
    return m.group(1)
