def load_into(ckpt, model, prefix, device, dtype=None, remap=None):
    """Just a debugging-friendly hack to apply the weights in a safetensors file to the pytorch module."""
    for key in ckpt.keys():
        model_key = key
        if remap is not None and key in remap:
            model_key = remap[key]
        if model_key.startswith(prefix) and not model_key.startswith("loss."):
            path = model_key[len(prefix) :].split(".")
            obj = model
            for p in path:
                if obj is list:
                    obj = obj[int(p)]
                else:
                    obj = getattr(obj, p, None)
                    if obj is None:
                        print(
                            f"Skipping key '{model_key}' in safetensors file as '{p}' does not exist in python model"
                        )
                        break
            if obj is None:
                continue
            try:
                tensor = ckpt.get_tensor(key).to(device=device)
                if dtype is not None and tensor.dtype != torch.int32:
                    tensor = tensor.to(dtype=dtype)
                obj.requires_grad_(False)
                # print(f"K: {model_key}, O: {obj.shape} T: {tensor.shape}")
                if obj.shape != tensor.shape:
                    print(
                        f"W: shape mismatch for key {model_key}, {obj.shape} != {tensor.shape}"
                    )
                obj.set_(tensor)
            except Exception as e:
                print(f"Failed to load key '{key}' in safetensors file: {e}")
                raise e

import torch
from safetensors import safe_open

MODEL_FOLDER = "/home/lxc/cache/AI-ModelScope/stable-diffusion-3___5-large/text_encoders"

from other_impls import SD3Tokenizer, SDClipModel

prompt = "attractive justin bieber as a god. highly detailed painting by gaston bussiere, craig mullins, j. c. leyendecker 8 k"

with torch.no_grad():
    tokenizer = SD3Tokenizer()

    CLIPL_CONFIG = {
        "hidden_act": "quick_gelu",
        "hidden_size": 768,
        "intermediate_size": 3072,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
    }

    clip_l = SDClipModel(
                    layer="hidden",
                    layer_idx=-2,
                    device="cpu",
                    dtype=torch.float32,
                    layer_norm_hidden_state=False,
                    return_projected_pooled=False,
                    textmodel_json_config=CLIPL_CONFIG)

    with safe_open(
                f"{MODEL_FOLDER}/clip_l.safetensors", framework="pt", device="cpu"
            ) as f:
        load_into(f, clip_l.transformer, "", "cpu", torch.float32)

    
    tokens = tokenizer.tokenize_with_weights(prompt)
    l_out, _ = clip_l.encode_token_weights(tokens["l"])
    
    torch.save(l_out, "tensor2.pt")
        