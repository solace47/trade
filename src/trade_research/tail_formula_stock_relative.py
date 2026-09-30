"""Reuse fixed stock-group input/calibration logic in a separate artifact namespace."""
import importlib.util
from pathlib import Path

from . import tail_formula_stock_holdout as original

STEM = 'tail_formula_stock_relative'
ROOT = Path('data/research') / STEM
PROTOCOL = Path('config') / (STEM + '_input_protocol.json')
INTENT = Path('config') / (STEM + '_intent.json')
spec = importlib.util.spec_from_file_location(__package__ + '._stock_relative_shared', original.__file__)
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)
shared.STEM, shared.ROOT, shared.PROTOCOL, shared.INTENT = STEM, ROOT, PROTOCOL, INTENT
shared.LABEL_FIELDS = [*shared.LABEL_FIELDS, 'adverse_return15']

prior, INPUTS, META, HEADER = shared.prior, shared.INPUTS, shared.META, shared.HEADER
ARMS, CORE_GATE, LABEL_FIELDS = shared.ARMS, shared.CORE_GATE, shared.LABEL_FIELDS
checked, group_ids, model_root = shared.checked, shared.group_ids, shared.model_root
project_labels, visible_group_cut, route_flags = shared.project_labels, shared.visible_group_cut, shared.route_flags
calibrate, verify_calibration = shared.calibrate, shared.verify_calibration
