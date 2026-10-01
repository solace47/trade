import json
import pytest
import verify_tail_formula_additive as verify
from trade_research.research_io import sha


def configured(tmp_path,monkeypatch):
    master=tmp_path/'master.json'
    master.write_text(json.dumps({'arms':{'control':{'OLD':'Q/VP20'},'memory':{'NEW':'Q/IV20'}}}))
    fold=tmp_path/'fold.json'
    fold.write_text(json.dumps({'master_protocol_sha256':sha(master),'expected_features':1,'arm':'control'}))
    monkeypatch.setattr(verify,'PROTOCOL',fold)
    report={'protocol_sha256':sha(master),'expressions':{'NEW':'Q/IV20'}}
    return master,report


def test_predeclared_control_can_be_absent_from_memory_definitions(tmp_path,monkeypatch):
    master,report=configured(tmp_path,monkeypatch)
    assert verify.native_definitions(report,{'OLD':'Q/VP20'},master)=={'OLD':'Q/VP20'}
    with pytest.raises(AssertionError):verify.native_definitions(report,{'OLD':'Q/VP20'})


def test_wrong_arm_or_changed_master_cannot_supply_definitions(tmp_path,monkeypatch):
    master,report=configured(tmp_path,monkeypatch)
    with pytest.raises(AssertionError):verify.native_definitions(report,{'NEW':'Q/IV20'},master)
    master.write_text(master.read_text()+'\n')
    with pytest.raises(AssertionError):verify.native_definitions(report,{'OLD':'Q/VP20'},master)
