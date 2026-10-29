"""Factories shared by tokenizer training and evaluation entry points."""

from __future__ import annotations

from lib_vqvae.vqvae import VQVAE


def build_vqvae(config: dict) -> VQVAE:
    model = config["model"]
    return VQVAE(
        data_dim=int(model["data_dim"]),
        n=int(model["n"]),
        codebook_dim=int(model["codebook_dim"]),
        n_codebooks=int(model["n_codebooks"]),
        encoder_hidden_dim=list(model["encoder_hidden_dim"]),
        decoder_hidden_dim=list(model["decoder_hidden_dim"]),
        beta=float(model["beta"]),
        gamma=float(model["gamma"]),
        dropout_rate=float(model["dropout_rate"]),
    )
