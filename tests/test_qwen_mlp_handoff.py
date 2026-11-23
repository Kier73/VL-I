import hashlib
import math
import struct
from types import SimpleNamespace

import pytest

from examples.run_qwen_mlp import project, silu
from examples.form_qwen_mlp import form


def test_silu_negative_tail():
    """SiLU(−1000) rounds to zero in FP64 and SiLU(x)−SiLU(−x)=x."""
    assert silu(-1000.) == 0
    assert math.isclose(silu(2.)-silu(-2.), 2.)


def test_tiled_projection_rejects_reordered_source(tmp_path):
    """Permuting W's rows preserves its extent but changes H(W) and its action."""
    data = struct.pack("<4H", 0x3f80, 0, 0, 0x3f80)
    key = "matrix"
    operand = dict(shape=[2, 2], sha256=hashlib.sha256(data).hexdigest())
    path = tmp_path / key
    path.write_bytes(data)
    assert project(tmp_path, key, operand, [3, 5], [1, -2]) == ([4, 3], 4)
    path.write_bytes(data[4:] + data[:4])
    with pytest.raises(ValueError, match="identity"):
        project(tmp_path, key, operand, [3, 5], [1, -2])


def test_formation_rejects_unbound_bias(tmp_path):
    """The emitted normalization n=(1+w)⊙r/RMS(r) would omit an additive bias b."""
    config = dict(model_type="qwen3_5_text", hidden_act="silu", hidden_size=5120, intermediate_size=17408)
    ground = {"tensors": {"tensors": [{"name": "model.language_model.layers.0.post_attention_layernorm.bias"}]}}
    snapshot = SimpleNamespace(config={"text_config": config}, ground=ground)
    with pytest.raises(ValueError, match="additive bias"):
        form(snapshot, tmp_path)
