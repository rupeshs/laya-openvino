# Copyright 2026 Rupesh Sreeraman. Licensed under the Apache License, Version 2.0.
# Part of an OpenVINO extension of laya (https://github.com/NandhaKishorM/laya).
"""OpenVINO export and inference for laya decision models.

`export()` traces a checkpoint into OpenVINO IR; `OVAgent` runs that IR behind the same
`system_one` API as `Agent`, so it drops straight into `Router` and the shortlist helpers.

    python -m laya.ov export convaiinnovations/laya --out laya-ov
"""
import json
import os
import threading
from typing import Any, Dict, Optional, Union

import numpy as np

from .common import decode_answers, clamp_temperature, prepare_batch

IR_STEM = "openvino_model"

# The traced graph resolves `p.size(-1) >= 2` in DecisionModel.forward at export time, so the
# exported topk(2) needs at least two marker slots. A lone single-option question would collate
# to one; pad the marker axis so that case still runs instead of failing inside the runtime.
MIN_MARKERS = 2


def _build_wrapper(model):
    import torch

    class Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
            # Every input is int64 so the runtime can feed plain integer arrays; bool inputs
            # are the one dtype numpy/OV bindings disagree about most often.
            return self.m(input_ids, attention_mask, marker_pos, marker_mask.bool(), qtype)

    return Wrapper(model).eval()


def export(
    model_id_or_path: str = "convaiinnovations/laya",
    out_dir: str = "laya-ov",
    *,
    subfolder: Optional[str] = None,
    token: Optional[str] = None,
    compress_to_fp16: bool = True,
) -> str:
    """Convert a laya checkpoint to OpenVINO IR in `out_dir`, and return that path.

    The result is self-contained: IR, tokenizer and `rl_agent_config.json`, so `OVAgent`
    needs neither torch nor the original checkpoint.
    """
    import openvino as ov
    import torch

    from .agent import Agent

    agent = Agent(model_id_or_path, device="cpu", token=token, subfolder=subfolder)
    max_len = agent.cfg.get("max_len", 512)

    wrapper = _build_wrapper(agent.model)
    # In eval() under no_grad, TransformerEncoderLayer takes a fused fast path that the tracer
    # cannot see through. Force the decomposed path for the duration of the export.
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        example = (
            torch.zeros((2, max_len), dtype=torch.long),
            torch.ones((2, max_len), dtype=torch.long),
            torch.ones((2, 4), dtype=torch.long),
            torch.ones((2, 4), dtype=torch.long),
            torch.zeros((2,), dtype=torch.long),
        )
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
    finally:
        torch.backends.mha.set_fastpath_enabled(True)

    for out, name in zip(ov_model.outputs, ("logits", "act_logits")):
        out.get_tensor().set_names({name})

    os.makedirs(out_dir, exist_ok=True)
    ov.save_model(ov_model, os.path.join(out_dir, IR_STEM + ".xml"), compress_to_fp16=compress_to_fp16)

    agent.tok.save_pretrained(os.path.join(out_dir, "tokenizer"))
    with open(os.path.join(out_dir, "rl_agent_config.json"), "w") as f:
        json.dump(agent.cfg, f, indent=2)

    return out_dir


class OVAgent:
    """Laya decision model running on OpenVINO. Same `system_one` contract as `Agent`."""

    def __init__(
        self,
        model_dir: str,
        device: str = "CPU",
        *,
        ov_config: Optional[Dict[str, Any]] = None,
    ):
        import openvino as ov
        from transformers import AutoTokenizer

        from .agent import _fix_tokenizer_config

        xml = os.path.join(model_dir, IR_STEM + ".xml")
        if not os.path.exists(xml):
            raise FileNotFoundError(
                f"No OpenVINO IR in {model_dir!r} (expected {IR_STEM}.xml). "
                f"Build one with laya.ov.export(...)."
            )
        cfg_path = os.path.join(model_dir, "rl_agent_config.json")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"'rl_agent_config.json' not found in {model_dir!r}.")

        _fix_tokenizer_config(model_dir)
        with open(cfg_path) as f:
            self.cfg = json.load(f)

        tok_dir = os.path.join(model_dir, "tokenizer")
        self.tok = AutoTokenizer.from_pretrained(tok_dir if os.path.exists(tok_dir) else self.cfg.get("encoder"))

        self.device = device
        self.model_dir = model_dir
        core = ov.Core()
        self._compiled = core.compile_model(xml, device, ov_config or {"PERFORMANCE_HINT": "LATENCY"})
        # Router runs inference outside its lock so concurrent predictions share a checkpoint.
        # An InferRequest is single-threaded, so give each calling thread its own.
        self._local = threading.local()

        self.temperature_raw = self.cfg.get("temperature", [1.0, 1.0, 1.0])
        self.temperature_by_options_raw = self.cfg.get("temperature_by_options", {})
        self.temperature = [clamp_temperature(t) for t in self.temperature_raw]
        self.temperature_by_options = {k: clamp_temperature(v)
                                       for k, v in self.temperature_by_options_raw.items()}

    def _request(self):
        req = getattr(self._local, "req", None)
        if req is None:
            req = self._local.req = self._compiled.create_infer_request()
        return req

    def system_one(self, state: Union[str, dict, list], questions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate typed questions across state in a single, parallel forward pass."""
        ids, items, b = prepare_batch(
            self.tok, state, questions,
            self.cfg.get("max_len", 512), self.cfg.get("head_max_len", 192),
        )

        mpos = b["marker_pos"].numpy().astype(np.int64)
        mmask = b["marker_mask"].numpy().astype(np.int64)
        if mpos.shape[1] < MIN_MARKERS:
            pad = MIN_MARKERS - mpos.shape[1]
            mpos = np.pad(mpos, ((0, 0), (0, pad)))
            mmask = np.pad(mmask, ((0, 0), (0, pad)))

        req = self._request()
        req.infer({
            "input_ids": b["input_ids"].numpy().astype(np.int64),
            "attention_mask": b["attention_mask"].numpy().astype(np.int64),
            "marker_pos": mpos,
            "marker_mask": mmask,
            "qtype": b["qtype"].numpy().astype(np.int64),
        })
        logits = req.get_tensor("logits").data
        act = req.get_tensor("act_logits").data

        answers = decode_answers(
            logits, act, ids, items, questions, self.temperature, self.temperature_by_options,
        )
        return {
            "model": "laya-rl-agent",
            "answers": answers,
            "usage": {"input_tokens": int(b["attention_mask"].sum()), "output_tokens": 0},
        }

    predict = system_one


def load(model_dir: str, device: str = "CPU", **kw) -> OVAgent:
    """Load an exported OpenVINO laya model."""
    return OVAgent(model_dir, device=device, **kw)


def _main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(prog="python -m laya.ov", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("export", help="convert a checkpoint to OpenVINO IR")
    e.add_argument("model", nargs="?", default="convaiinnovations/laya")
    e.add_argument("--out", default="laya-ov")
    e.add_argument("--subfolder", default=None)
    e.add_argument("--fp32", action="store_true", help="keep fp32 weights (default compresses to fp16)")

    args = ap.parse_args(argv)
    path = export(args.model, args.out, subfolder=args.subfolder, compress_to_fp16=not args.fp32)
    print("exported to %s" % path)


if __name__ == "__main__":
    _main()
