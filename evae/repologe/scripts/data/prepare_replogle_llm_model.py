"""Export exactly the local model's language weights, without touching the base."""

# Support direct CLI execution and shared imports after directory organization.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import _bootstrap


import json
from pathlib import Path
import shutil

from safetensors.torch import load_file, save_file

from replogle_llm_common import DEFAULT_ART, read_json, save_json, sha


def main():
    source = read_json(DEFAULT_ART / "sources.json")
    model_path = Path(source["model_source"])
    output = DEFAULT_ART / "initial_text_model"
    if (output / "export.json").exists():
        receipt = read_json(output / "export.json")
        assert sha(output / "model.safetensors") == receipt["model_sha256"]
        print("Existing language export verified", flush=True)
        return
    provenance = read_json(model_path / "LOCAL_SOURCE.json")
    for f in provenance["files"]:
        assert sha(model_path / f["file"]) == f["sha256"], f["file"]
    config = read_json(model_path / "config.json")["text_config"]
    config["architectures"] = ["Qwen3_5ForCausalLM"]
    config["use_cache"] = False
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "config.json", config)
    states = load_file(model_path / "model.safetensors-00001-of-00001.safetensors")
    state = {"model." + k.removeprefix("model.language_model."): v
             for k, v in states.items() if k.startswith("model.language_model.")}
    assert "model.embed_tokens.weight" in state
    n_parameters = sum(value.numel() for value in state.values())
    save_file(state, output / "model.safetensors", metadata={"format": "pt"})
    for name in ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja"]:
        shutil.copyfile(model_path / name, output / name)
    save_json(output / "export.json", {"status": "passed", "source_model": str(model_path),
        "source_revision": source["model_revision"], "exported_parameters": n_parameters,
        "mapping": "model.language_model.* -> model.*; tied output uses model.embed_tokens.weight",
        "excluded": ["visual encoder", "MTP auxiliary modules"],
        "model_sha256": sha(output / "model.safetensors"), "frozen_vae": source["vae_run"]})
    print(json.dumps(read_json(output / "export.json"), indent=2), flush=True)


if __name__ == "__main__":
    main()
