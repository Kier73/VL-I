"""PBG-008 standard-library integrity and crash-recovery journal tests."""
from pathlib import Path
import json
import multiprocessing
import tempfile

import pytest

from vli.durable_neural_acquisition import (
    ObservationRefused,_atomic_once,_read,_seal,_path,_unpath,_canonical,
)


def test_f64_input_observation_recovers_without_decimal_approximation():
    vector=((101,(-0.0,0.1,1.5)),(102,(0.3333333333333333,-3.5,2.0)))
    projected=_path(vector)
    assert projected[0][1][0]=='-0x0.0p+0'
    assert _unpath(projected)==vector
    assert _path(_unpath(projected))==projected


def test_atomic_once_is_idempotent_and_rejects_conflicting_prepared_output():
    with tempfile.TemporaryDirectory() as d:
        f=Path(d)/'prepared.json'
        a=_seal({'request':'a'*64,'output':'expected'})
        _atomic_once(f,a)
        assert _read(f)==a
        _atomic_once(f,a)
        with pytest.raises(ObservationRefused,match='Conflicting'):
            _atomic_once(f,_seal({'request':'a'*64,'output':'contradictory'}))
        assert _read(f)==a


def test_corruption_and_missing_record_are_refused():
    with tempfile.TemporaryDirectory() as d:
        f=Path(d)/'outbox.json'
        with pytest.raises(ObservationRefused):_read(f)
        _atomic_once(f,_seal({'source':'c'*64}))
        changed=json.loads(f.read_text())
        changed['source']='f'*64
        f.write_bytes(_canonical(changed))
        with pytest.raises(ObservationRefused,match='Missing or invalid'):
            _read(f)


def test_noncanonical_json_is_refused_even_when_decodable():
    with tempfile.TemporaryDirectory() as d:
        f=Path(d)/'request.json'
        f.write_text(json.dumps(_seal({'source':'c'*64}),indent=2))
        with pytest.raises(ObservationRefused):_read(f)


def _race_prepare(path,word,ready,go):
    f=Path(path)
    ready.put(1)
    go.wait(timeout=10)
    try:
        _atomic_once(f,_seal({'value':word}))
        return_word='recorded'
    except ObservationRefused:
        return_word='conflict'
    ready.put(return_word)


def test_two_processes_cannot_overwrite_once_record():
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)/'intent.json'
        ctx=multiprocessing.get_context('spawn')
        ready=ctx.Queue(); go=ctx.Event()
        children=[ctx.Process(target=_race_prepare,args=(str(p),word,ready,go))
                  for word in ('before','after')]
        for c in children:c.start()
        for _ in children:assert ready.get(timeout=12)==1
        go.set()
        results=[ready.get(timeout=12) for _ in children]
        for c in children:c.join(timeout=12)
        assert all(c.exitcode==0 for c in children)
        assert sorted(results)==['conflict','recorded']
        assert _read(p)['value'] in ('before','after')
