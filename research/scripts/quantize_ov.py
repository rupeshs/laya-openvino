# Copyright 2026 Rupesh Sreeraman. Licensed under the Apache License, Version 2.0.
# Part of an OpenVINO extension of laya (https://github.com/NandhaKishorM/laya).
"""Quantize an exported laya OpenVINO IR to int8 or int4 weights.

`laya.ov.export` compresses to fp16, which only shrinks the file. This goes further via NNCF.
int8 is the useful setting; see the benchmark table in the README for why int4 is not.

  python research/scripts/quantize_ov.py laya-ov laya-ov-int8 --mode int8
  python research/scripts/quantize_ov.py laya-ov laya-ov-int4 --mode int4 --group-size 64
"""
import argparse
import builtins
import os
import shutil
import sys
import time
import typing

# NNCF 2.19 puts `numpy.typing.NDArray` in its dispatch registry. Under numpy >= 2.3 that is a
# TypeAliasType rather than a class, so the registry walk raises TypeError instead of missing.
# Resolve such a key to the class it aliases (np.ndarray) so dispatch finds the numpy backend.
_real_issubclass = builtins.issubclass


def _issubclass(a, b):
    try:
        return _real_issubclass(a, b)
    except TypeError:
        origin = typing.get_origin(getattr(b, "__value__", b))
        if origin is None:
            return False
        try:
            return _real_issubclass(a, origin)
        except TypeError:
            return False


builtins.issubclass = _issubclass

import nncf  # noqa: E402
import openvino as ov  # noqa: E402

MODES = {"int8": "INT8_ASYM", "int4": "INT4_ASYM", "int4_sym": "INT4_SYM"}


def quantize(src, dst, mode="int8", group_size=64, ratio=1.0):
    core = ov.Core()
    model = core.read_model(os.path.join(src, "openvino_model.xml"))

    kw = {"mode": getattr(nncf.CompressWeightsMode, MODES[mode])}
    if mode != "int8":
        # act_head/scorer have channel sizes (1028) no useful group size divides, and they are a
        # few thousand weights -- leave them rather than shrink the group for the whole model.
        kw.update(group_size=group_size, ratio=ratio,
                  ignored_scope=nncf.IgnoredScope(patterns=[".*act_head.*", ".*scorer.*"]))

    t0 = time.perf_counter()
    model = nncf.compress_weights(model, **kw)
    os.makedirs(dst, exist_ok=True)
    ov.save_model(model, os.path.join(dst, "openvino_model.xml"))

    # OVAgent needs the config and tokenizer beside the IR for the directory to stay self-contained.
    shutil.copy(os.path.join(src, "rl_agent_config.json"), dst)
    tok = os.path.join(dst, "tokenizer")
    if not os.path.exists(tok):
        shutil.copytree(os.path.join(src, "tokenizer"), tok)

    return time.perf_counter() - t0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src", help="an export dir from `python -m laya.ov export`")
    ap.add_argument("dst")
    ap.add_argument("--mode", default="int8", choices=sorted(MODES))
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--ratio", type=float, default=1.0, help="fraction of layers taken to 4 bit")
    args = ap.parse_args(argv)

    took = quantize(args.src, args.dst, args.mode, args.group_size, args.ratio)
    mb = sum(os.path.getsize(os.path.join(args.dst, f)) for f in os.listdir(args.dst)
             if os.path.isfile(os.path.join(args.dst, f))) / 2 ** 20
    print("wrote %s (%s, %.0f MB) in %.0fs" % (args.dst, args.mode, mb, took))


if __name__ == "__main__":
    sys.exit(main())
