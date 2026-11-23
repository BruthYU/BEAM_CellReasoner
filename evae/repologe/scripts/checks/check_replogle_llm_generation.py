"""Verify genuine condition-only generation and frozen NB decoding after SFT."""

# Support direct CLI execution and shared imports after directory organization.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import _bootstrap


import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, Qwen3_5ForCausalLM, set_seed

from sc_evae.models.factory import load_vae_from_run
from replogle_llm_common import (
    ROOT, DEFAULT_ART, prepare_tokenizer, condition_prefixes, install_fast_delta,
    read_json, save_json, sample_codes,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    set_seed(420)
    install_fast_delta()
    source = read_json(DEFAULT_ART / "sources.json")
    model = Qwen3_5ForCausalLM.from_pretrained(args.model_dir, local_files_only=True,
                                             dtype=torch.float32, attn_implementation="sdpa").to("cuda").eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, padding_side="left")
    conditions = [json.loads(line) for line in (ROOT / source["text_conditions"] / "conditions.jsonl").read_text().splitlines()]
    selected = [row for row in conditions if row["cell_line"] == "HepG2" and
                row["dataset_label"] in ["non-targeting", read_json(DEFAULT_ART / "split.json")["validation_labels"][0]]]
    prompts = condition_prefixes(tokenizer, selected)
    # Include repeated conditions to exercise independent stochastic cell draws.
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        tokens = sample_codes(model, tokenizer, prompts * 4)
    assert tokens.shape == (8, 64)
    vae = load_vae_from_run(str(ROOT / source["vae_run"]), device="cuda")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        expression = vae.decode(vae.quantizer.decode_indices(tokens))
    assert expression.shape == (8, 2000) and torch.isfinite(expression).all() and (expression >= 0).all()
    save_json(args.output, {"status": "passed", "token_shape": list(tokens.shape),
        "token_range": [int(tokens.min()), int(tokens.max())], "expression_shape": list(expression.shape),
        "conditions": [row["dataset_label"] for row in selected], "condition_only_generation": True,
        "target_expression_or_tokens_as_input": False, "nb_depth": "Training prior; no observed target depth",
        "first_generated_tokens": tokens[0].tolist(), "frozen_nb_vae": source["vae_run"]})
    print(json.dumps(read_json(args.output), indent=2), flush=True)


if __name__ == "__main__":
    main()
