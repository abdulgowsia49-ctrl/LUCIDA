"""
Minimal PDS4 label + raw .img reader for Chandrayaan-2 science products.

PDS4 products ship as an XML label (.xml) describing the binary layout,
alongside the raw array file (.img/.qub). This module parses just the
handful of fields needed to load the array correctly -- dimensions, sample
bit depth, byte order, and band count -- rather than a full PDS4 schema
implementation. For anything beyond raw array + basic geometry (full
provenance, calibration history, etc.) use a proper PDS4 toolkit instead.

Reference: Chandrayaan-2 Mission Data Handbook, ISRO (product label schema).
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# PDS4 labels are namespaced; wildcard search below avoids hardcoding a
# specific PDS4 schema version.
_NS_WILDCARD = "{*}"

_DTYPE_MAP = {
    ("UnsignedByte", 1): np.uint8,
    ("UnsignedMSB2", 2): np.dtype(">u2"),
    ("UnsignedLSB2", 2): np.dtype("<u2"),
    ("SignedMSB2", 2): np.dtype(">i2"),
    ("SignedLSB2", 2): np.dtype("<i2"),
    ("IEEE754MSBSingle", 4): np.dtype(">f4"),
    ("IEEE754LSBSingle", 4): np.dtype("<f4"),
}


@dataclass
class PDS4ImageMeta:
    lines: int
    samples: int
    bands: int
    dtype: np.dtype
    data_file: str


def parse_label(label_path: str | Path) -> PDS4ImageMeta:
    """Parse a PDS4 .xml label and return the array shape/dtype needed to
    load the companion .img file."""
    label_path = Path(label_path)
    tree = ET.parse(label_path)
    root = tree.getroot()

    def find(tag_path: list[str], node=None):
        node = node if node is not None else root
        for tag in tag_path:
            node = node.find(f"{_NS_WILDCARD}{tag}")
            if node is None:
                return None
        return node

    file_area = find(["File_Area_Observational"])
    if file_area is None:
        raise ValueError(f"No File_Area_Observational block found in {label_path}")

    file_node = find(["File"], file_area)
    data_file = file_node.find(f"{_NS_WILDCARD}file_name").text.strip()

    array_node = find(["Array_2D_Image"], file_area) or find(["Array_3D_Spectrum"], file_area)
    if array_node is None:
        raise ValueError(f"No supported Array element found in {label_path}")

    element_array = array_node.find(f"{_NS_WILDCARD}Element_Array")
    data_type_str = element_array.find(f"{_NS_WILDCARD}data_type").text.strip()

    axis_dims = {}
    for axis in array_node.findall(f"{_NS_WILDCARD}Axis_Array"):
        name = axis.find(f"{_NS_WILDCARD}axis_name").text.strip().lower()
        elements = int(axis.find(f"{_NS_WILDCARD}elements").text.strip())
        axis_dims[name] = elements

    lines = axis_dims.get("line", axis_dims.get("y", 1))
    samples = axis_dims.get("sample", axis_dims.get("x", 1))
    bands = axis_dims.get("band", 1)

    dtype = None
    for (name, size), np_type in _DTYPE_MAP.items():
        if name == data_type_str:
            dtype = np_type
            break
    if dtype is None:
        raise ValueError(f"Unrecognized PDS4 data_type '{data_type_str}' in {label_path}")

    return PDS4ImageMeta(lines=lines, samples=samples, bands=bands, dtype=dtype, data_file=data_file)


def load_array(label_path: str | Path) -> np.ndarray:
    """
    Load the raw array referenced by a PDS4 label into memory.

    Returns:
        (lines, samples) array if single-band, else (lines, samples, bands).
    """
    label_path = Path(label_path)
    meta = parse_label(label_path)
    img_path = label_path.parent / meta.data_file

    if not img_path.exists():
        raise FileNotFoundError(f"Label references '{meta.data_file}' but it's not next to {label_path}")

    count = meta.lines * meta.samples * meta.bands
    flat = np.fromfile(img_path, dtype=meta.dtype, count=count)

    if flat.size != count:
        raise ValueError(
            f"Expected {count} elements from label geometry, got {flat.size} reading {img_path}"
        )

    if meta.bands > 1:
        return flat.reshape(meta.bands, meta.lines, meta.samples).transpose(1, 2, 0)
    return flat.reshape(meta.lines, meta.samples)


def normalize_dn(array: np.ndarray, dn_max: float | None = None) -> np.ndarray:
    """Convert raw digital numbers to float32 in [0, 1]."""
    dn_max = dn_max or float(np.iinfo(array.dtype).max) if np.issubdtype(array.dtype, np.integer) else float(array.max())
    return np.clip(array.astype(np.float32) / dn_max, 0.0, 1.0)
