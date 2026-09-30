"""Reuse fixed input-only calibration in a separate, absolute-target namespace."""
import importlib.util
from pathlib import Path

from . import tail_formula_input_calibration as original

# A separate module namespace keeps the immutable relative artifacts untouched.
# Reuse the same calibration arithmetic rather than copying it or changing it.
spec = importlib.util.spec_from_file_location(__package__ + '._absolute_input_calibration_shared', original.__file__)
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)
shared.STEM = 'tail_formula_absolute_input_calibration'
shared.ROOT = Path('data/research') / shared.STEM
shared.PROTOCOL = Path('config') / (shared.STEM + '_input_protocol.json')
shared.INTENT = Path('config') / (shared.STEM + '_intent.json')

STEM, ROOT, PROTOCOL, INTENT = shared.STEM, shared.ROOT, shared.PROTOCOL, shared.INTENT
INPUTS, META, HEADER, CORE_GATE, ARMS = shared.INPUTS, shared.META, shared.HEADER, shared.CORE_GATE, shared.ARMS
prior = shared.prior
checked = shared.checked
calibration_paths = shared.calibration_paths
calibrate = shared.calibrate
verify_calibration = shared.verify_calibration
