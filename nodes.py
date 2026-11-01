import json

from .generation import complete
from .runtime import LOCK, MODEL_NAME, MODEL_REVISION, acquire_model


def run_completion(prompt, max_new_tokens, seed, temperature):
    import comfy.model_management as mm
    from comfy.utils import ProgressBar
    bar = ProgressBar(max_new_tokens)
    with LOCK:
        model, tokenizer = acquire_model()
        result = complete(model, tokenizer, prompt, max_new_tokens=max_new_tokens,
                          seed=seed, temperature=temperature,
                          interrupt=mm.throw_exception_if_processing_interrupted,
                          progress=bar.update_absolute)
        info = json.dumps({"model": MODEL_NAME, "revision": MODEL_REVISION,
                           "backend": "native-pytorch-comfy-modelpatcher",
                           "device": str(next(model.parameters()).device),
                           "input_tokens": result.input_tokens,
                           "output_tokens": len(result.token_ids),
                           "finish_reason": result.finish_reason,
                           "seconds": round(result.seconds, 4), "seed": seed}, ensure_ascii=False)
        return result.text, info


class RSTIPONativeCompletion:
    """Low-level baseline exposing the official TIPO completion format."""
    CATEGORY = "RS/TIPO Native"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("completion", "info")
    FUNCTION = "generate"
    DESCRIPTION = "Experimental native TIPO v2.1. Completes the official TIPO format; no chat template."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "prompt": ("STRING", {"multiline": True, "default": "target: <|long|>\ntag: 1girl, garden, "}),
            "max_new_tokens": ("INT", {"default": 128, "min": 1, "max": 1024}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "temperature": ("FLOAT", {"default": 0.5, "min": 0, "max": 2, "step": 0.05,
                                         "tooltip": "0 = greedy baseline. Other values sample using the seed."}),
        }}

    def generate(self, prompt, max_new_tokens, seed, temperature):
        return run_completion(prompt, max_new_tokens, seed, temperature)


class RSTIPONativeExpand:
    CATEGORY = "RS/TIPO Native"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt", "added_tags", "info")
    FUNCTION = "expand"
    DESCRIPTION = ("Experimental TIPO tag expansion. Original text/weights are preserved. "
                   "This baseline does not isolate character ownership or prevent invented attributes.")

    @classmethod
    def INPUT_TYPES(cls):
        base = RSTIPONativeCompletion.INPUT_TYPES()["required"]
        return {"required": {
            "tags": ("STRING", {"multiline": True, "default": "1girl, garden"}),
            "length": (["short", "long", "very_long"], {"default": "long"}),
            **{name: base[name] for name in ("max_new_tokens", "seed", "temperature")},
        }}

    def expand(self, tags, length, max_new_tokens, seed, temperature):
        if not tags.strip():
            raise ValueError("Enter at least one tag before running TIPO.")
        if length not in ("short", "long", "very_long"):
            raise ValueError("Unknown TIPO target length.")
        # This baseline's official tag-only format remains separate from semantic control research.
        # Do not flatten multiple characters/pipes into this field.
        input_prompt = f"target: <|{length}|>\ntag: {tags.rstrip().rstrip(',')}, "
        text, info = run_completion(input_prompt, max_new_tokens, seed, temperature)
        # Only the first tag field belongs in a tag string; preserve all raw output in metadata.
        generated = text.split("<|eos|>", 1)[0].split("\n", 1)[0].strip(" ,")
        meta = json.loads(info)
        meta["raw_completion"] = text
        combined = tags + (", " + generated if generated else "")
        return combined, generated, json.dumps(meta, ensure_ascii=False)
