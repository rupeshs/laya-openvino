# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Tests are standalone scripts, not pytest modules.** They call `sys.exit()` at import, so `pytest tests/` fails with `INTERNALERROR`. Run each one directly:

```bash
python tests/test_router.py          # 150 assertions: routing, language detection
python tests/test_criteria.py        # criterion rendering
python tests/test_shortlist.py       # embedding shortlist
python tests/test_decision_model.py  # DecisionModel.forward, tiny from-config encoder
python tests/test_packaging.py       # pyproject/CI version consistency
python tests/test_download.py        # snapshot_download allow_patterns
python tests/test_email.py           # email cleaning
python tests/test_ov.py              # OpenVINO conversion + parity
```

All of the above run offline against tiny from-config models — no checkpoint download. The only test needing real weights is `tests/test_local_e2e.py [model_root]`, which expects `laya/`, `laya-multilingual/` and `laya-typed-decisions/` under `~/laya_models` (override with `LAYA_DEVICE=cpu`).

To run a single case, edit the script — they are linear top-to-bottom assertion lists with a `PASS`/`FAIL` accumulator and a `check(name, got, want)` / `check_true(name, cond)` convention. Match that style when adding tests, and register new test files in both `.github/workflows/ci.yml` and `release.yml`.

Lint, exactly as CI runs it:

```bash
ruff check laya/ --select=E9,F63,F7,F82,F401,F811 --line-length=120
python -m compileall -q laya/ tests/
```

## Architecture

Laya evaluates typed questions over a state in **one forward pass, no generation**. Everything else follows from that.

**The sequence layout** (`common.build_sequence`) is the core trick: `[CLS] <type> instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]`. One `[MASK]` marker precedes each option, and the token positions of those markers are returned alongside the token ids. `DecisionModel.forward` gathers the hidden states *at those marker positions* and scores each with a shared scorer head — so K options cost one pass, not K. Any change to the sequence format must keep markers and `marker_pos` in lockstep, or options get scored against the wrong text.

Options share a single `head_max_len` budget (default 192). A large label set therefore starves each label of tokens; that is the problem `shortlist.py` exists to solve (embed, keep top-k, then one normal pass — it does not add a second decision-model pass).

**`DecisionModel`** (`common.py`) = pretrained encoder + optional transformer head + `scorer` (per-marker logit) + `act_head` (a should-I-act logit fed confidence features: top1, top1−top2, normalized entropy, option count). `qtype` (`choice`/`score`/`noul`) enters as a learned embedding added to every position.

**Backends share one pre/post path.** `common.prepare_batch()` and `common.decode_answers()` are the contract; `Agent` (torch, `agent.py`) and `OVAgent` (OpenVINO, `ov.py`) differ *only* in the forward call between them. Temperature calibration and confidence live in `decode_answers` alone — do not recompute them in a backend, or the backends will drift.

**Calibration has a safety clamp.** Shipped checkpoints contain fitted temperatures, some below 1.0, which *sharpen* logits instead of softening them (the `choice:11+` bucket is 0.1006, turning a 0.24 top probability into a published 0.99). `clamp_temperature` confines them to `[0.5, 5.0]` and warns. Keep raw values available for inspection but never apply an unclamped one.

**`Router`** (`router.py`) picks a checkpoint per request using `lang.py` (exact script detection, plus a best-effort stopword/diacritic heuristic for Latin-script language ID). It holds an LRU of loaded agents behind an `RLock` that guards *lifecycle only* — inference runs outside the lock so concurrent predictions share a checkpoint without serialising. Anything with a `system_one`/`predict` method can be `Router.attach()`ed, which is how `OVAgent` drops in.

All three checkpoints live as subfolders of one Hub repo (`DEFAULT_MODELS`), with standalone mirrors in `STANDALONE_MODELS`. Downloads are `allow_patterns`-restricted so loading one variant does not pull the whole family.

## OpenVINO backend

`python -m laya.ov export convaiinnovations/laya --out laya-ov` traces to IR; `laya.OVAgent("laya-ov")` serves it. Export dirs are self-contained (IR + tokenizer + config) so inference needs no torch. Two non-obvious constraints:

- Export must disable the `TransformerEncoderLayer` fast path (`torch.backends.mha.set_fastpath_enabled(False)`); in eval+no_grad it fuses into an op the converter cannot trace through.
- The trace resolves `p.size(-1) >= 2` in `DecisionModel.forward` statically, baking in `topk(2)`. `OVAgent` pads the marker axis to `MIN_MARKERS` so a lone single-option question still runs.

`compress_to_fp16=True` (the default) is a *storage* change only — CPUs without native fp16 still compute in fp32.

## Invariants enforced by CI

`.github/workflows/security.yml` greps `laya/` and fails the build on:

- `pickle`, `torch.load(`, `joblib.load(`, `yaml.load(` — checkpoints come from the Hub, so a pickle path is remote code execution on `laya.load("someone/their-model")`. Weights load through safetensors only.
- `os.system(`, `subprocess.`, `shell=True`, `eval(`, `exec(` — note `.eval()` is excluded, so `model.eval()` is fine.

`tests/test_packaging.py` keeps `requires-python`, the Python classifiers, and the CI matrix mutually consistent; bumping the floor means editing all three.

## Notes

- `tests/test_download.py` currently fails on Windows over `\` vs `/` path separators. It is a pre-existing platform issue, not a regression.
- ModernBERT's `reference_compile` is force-disabled at load; `torch.compile` is a loss at laya's batch sizes and hangs on some platforms.
