"""Parallel artifact ranges must cover every byte exactly once."""

from scripts.import_ci_artifact import _byte_ranges


def test_byte_ranges_are_contiguous_for_uneven_sizes() -> None:
    ranges = _byte_ranges(17, 4)

    assert ranges == [(0, 4), (5, 9), (10, 14), (15, 16)]
    assert sum(end - start + 1 for start, end in ranges) == 17
