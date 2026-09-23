# Copyright 2026 Rupesh Sreeraman. Licensed under the Apache License, Version 2.0.
# Part of an OpenVINO extension of laya (https://github.com/NandhaKishorM/laya).
"""Offline tests for the OpenVINO backend (laya/ov.py).

Converts a tiny from-config BERT DecisionModel, so these run without a checkpoint
download. Skips cleanly when openvino is not installed.

Run:  python3 tests/test_ov.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoConfig, AutoModel  # noqa: E402

from laya.common import DecisionModel, decode_answers, to_internal  # noqa: E402
from laya.ov import MIN_MARKERS, _build_wrapper  # noqa: E402

PASS, FAIL = [], []


def check(name, got, want):
    if got == want:
        PASS.append(name)
    else:
        FAIL.append("%s: got %r, want %r" % (name, got, want))


def check_true(name, cond, detail=""):
    if cond:
        PASS.append(name)
    else:
        FAIL.append("%s %s" % (name, detail))


try:
    import openvino as ov
except ImportError:
    print("openvino not installed; skipping OpenVINO backend tests")
    sys.exit(0)


def _tiny() -> DecisionModel:
    cfg = AutoConfig.for_model(
        "bert", hidden_size=16, num_hidden_layers=1, num_attention_heads=1,
        intermediate_size=32, vocab_size=50,
    )
    m = DecisionModel(AutoModel.from_config(cfg), head_layers=1, n_act=2)
    m.eval()
    return m


def _inputs(B, L, K, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randint(0, 50, (B, L), generator=g, dtype=torch.long),
        torch.ones((B, L), dtype=torch.long),
        torch.randint(0, L, (B, K), generator=g, dtype=torch.long),
        torch.ones((B, K), dtype=torch.long),
        torch.randint(0, 3, (B,), generator=g, dtype=torch.long),
    )


# ---------------------------------------------------------------- convert once, reuse
model = _tiny()
wrapper = _build_wrapper(model)
torch.backends.mha.set_fastpath_enabled(False)
example = _inputs(2, 24, 3)
with torch.no_grad():
    ov_model = ov.convert_model(
        wrapper,
        example_input=example,
        input=[
            ("input_ids", ov.PartialShape([-1, -1]), ov.Type.i64),
            ("attention_mask", ov.PartialShape([-1, -1]), ov.Type.i64),
            ("marker_pos", ov.PartialShape([-1, -1]), ov.Type.i64),
            ("marker_mask", ov.PartialShape([-1, -1]), ov.Type.i64),
            ("qtype", ov.PartialShape([-1]), ov.Type.i64),
        ],
    )
torch.backends.mha.set_fastpath_enabled(True)

for out, nm in zip(ov_model.outputs, ("logits", "act_logits")):
    out.get_tensor().set_names({nm})

check("convert/two outputs", len(ov_model.outputs), 2)
check("convert/five inputs", len(ov_model.inputs), 5)
check_true("convert/batch dim is dynamic", ov_model.inputs[0].get_partial_shape()[0].is_dynamic)
check_true("convert/seq dim is dynamic", ov_model.inputs[0].get_partial_shape()[1].is_dynamic)
check_true("convert/marker dim is dynamic", ov_model.inputs[2].get_partial_shape()[1].is_dynamic)

compiled = ov.Core().compile_model(ov_model, "CPU")


# ---------------------------------------------------------------- numeric parity
# Shapes deliberately differ from the traced 2x24x3: a baked-in dimension would show up here.
for B, L, K in [(2, 24, 3), (1, 24, 2), (3, 40, 5), (1, 12, 2), (4, 33, 4)]:
    t = _inputs(B, L, K, seed=B * 100 + L)
    with torch.no_grad():
        ref = wrapper(*t)
    got = compiled([x.numpy() for x in t])
    tag = "B=%d L=%d K=%d" % (B, L, K)
    check("parity/logits shape %s" % tag, tuple(got[0].shape), (B, K))
    check_true("parity/logits %s" % tag,
               np.allclose(ref[0].numpy(), got[0], atol=1e-4),
               "max %.2e" % np.abs(ref[0].numpy() - got[0]).max())
    check_true("parity/act_logits %s" % tag,
               np.allclose(ref[1].numpy(), got[1], atol=1e-3),
               "max %.2e" % np.abs(ref[1].numpy() - got[1]).max())


# ---------------------------------------------------------------- masked markers
# An unused marker slot must not steal probability from the real options.
t = _inputs(1, 24, 4, seed=7)
mask = t[3].clone()
mask[0, 2:] = 0
masked = (t[0], t[1], t[2], mask, t[4])
got = compiled([x.numpy() for x in masked])
check_true("mask/inactive markers are floored", bool((got[0][0, 2:] <= -1e3).all()),
           "got %r" % (got[0][0, 2:],))


# ---------------------------------------------------------------- MIN_MARKERS padding
# The traced graph bakes in topk(2), so a lone single-option question needs padding.
check("pad/MIN_MARKERS", MIN_MARKERS, 2)
one = _inputs(1, 24, 1, seed=3)
padded = (
    one[0], one[1],
    np.pad(one[2].numpy(), ((0, 0), (0, 1))),
    np.pad(one[3].numpy(), ((0, 0), (0, 1))),
    one[4],
)
got = compiled([x if isinstance(x, np.ndarray) else x.numpy() for x in padded])
check("pad/K=1 padded to 2 runs", tuple(got[0].shape), (1, 2))
# decode_answers slices to k=1, so a single-option choice is always probability 1.0
answers = decode_answers(
    got[0], got[1], ["only"],
    [{"markers": [0]}],
    {"only": {"type": "choice", "instructions": "pick", "criteria": {"a": None}}},
    [1.0, 1.0, 1.0], {},
)
check("pad/single option is certain", answers["only"]["probabilities"], {"a": 1.0})
check("pad/single option confidence", answers["only"]["confidence"], 1.0)


# ---------------------------------------------------------------- decode_answers contract
logits = np.array([[2.0, 1.0, 0.0, -1e4]], dtype=np.float32)
act = np.array([[3.0, 1.0]], dtype=np.float32)
items = [{"markers": [0, 1, 2]}]
qs = {"q": {"type": "score", "instructions": "how much", "criteria": ["lo", "mid", "hi"]}}
a = decode_answers(logits, act, ["q"], items, qs, [1.0, 1.0, 1.0], {})["q"]
check("decode/score legend", a["legend"], {"0": "lo", "1": "mid", "2": "hi"})
check_true("decode/score probabilities sum to 1",
           abs(sum(a["probabilities"].values()) - 1.0) < 1e-3)
check_true("decode/expected score in range", 0.0 <= a["score"] <= 2.0)
check_true("decode/act_probability is a softmax", 0.0 < a["action"]["act_probability"] < 1.0)

# temperature_by_options must win over the per-type default for its bucket
hot = decode_answers(logits, act, ["q"], items, qs, [1.0, 1.0, 1.0], {"score:3-5": 5.0})["q"]
check_true("decode/bucket temperature flattens the distribution",
           hot["probabilities"]["0"] < a["probabilities"]["0"])

check("decode/to_internal normalises a choice list",
      list(to_internal({"type": "choice", "instructions": "i", "criteria": ["x", "y"]})["crit"]),
      ["x", "y"])


# ---------------------------------------------------------------- report
print()
for f in FAIL:
    print("FAIL", f)
print("%d passed, %d failed" % (len(PASS), len(FAIL)))
sys.exit(1 if FAIL else 0)
