"""Shared Qwen loading and observed-depth NB decoding; callers own RNG/batching."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import _bootstrap

import torch
from transformers import Qwen3_5ForCausalLM
from sc_evae.models.factory import load_vae_from_run
from replogle_llm_common import sample_codes


def load_llm(model_dir, device=None):
    model = Qwen3_5ForCausalLM.from_pretrained(model_dir, local_files_only=True,
        dtype=torch.float32, attn_implementation="sdpa")
    return (model.to(device) if device is not None else model).eval()


def load_nb_decoder(vae_run, device=None):
    kwargs = {} if device is None else {"device": device}
    vae = load_vae_from_run(str(vae_run), **kwargs).eval()
    vae.requires_grad_(False)
    return vae


@torch.inference_mode()
def predict_cells(model, tokenizer, vae, prefixes, counts):
    # counts supplies decoder depth only; it never enters the language model.
    codes = sample_codes(model, tokenizer, prefixes)
    latent = vae.quantizer.decode_indices(codes)
    predicted = vae.decode_observed(latent, None, counts.sum(-1, keepdim=True)).float()
    assert torch.isfinite(predicted).all() and (predicted >= 0).all()
    return codes, predicted
