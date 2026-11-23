import struct

import pytest

from vli.grouped_projection import residual


def test_grouped_signed_coefficients():
    """Wz+e = [2(3+4)-5+1, 0(3+5)+4-1] with two distinct codes per row."""
    matrix = struct.pack("<6H", 0x4000, 0x4000, 0xbf80, 0, 0x3f80, 0)
    assert residual(matrix, [3, 4, 5], [1, -1]) == ([10, 3], 4)


def test_invalid_source_extent():
    """A BF16 matrix over m×n coordinates requires exactly 2mn bytes."""
    with pytest.raises(ValueError, match="extent"):
        residual(bytes(3), [1, 2], [0])


def test_nonfinite_weight():
    """The distributive identity assumes W ∈ ℝ^(m×n)."""
    with pytest.raises(ValueError, match="Nonfinite trained coefficient"):
        residual(struct.pack("<H", 0x7fc0), [0], [1])
