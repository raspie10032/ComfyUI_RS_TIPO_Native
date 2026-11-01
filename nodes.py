import json

from .generation import complete
from .expansion import expand_owned, expand_pipe
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


def control_inputs():
    base = RSTIPONativeExpand.INPUT_TYPES()["required"]
    return {**{name: base[name] for name in ("length", "max_new_tokens", "seed", "temperature")},
            "common_weight": ("FLOAT", {"default": 0.5, "min": 0., "max": 4., "step": 0.1,
                "tooltip": "Common source-token attention multiplier. 0 blocks it for the character. Not an image prompt weight."}),
            "character_weight": ("FLOAT", {"default": 1.5, "min": 0., "max": 4., "step": 0.1,
                "tooltip": "Character source-token attention multiplier. 1 is neutral. Stronger does not guarantee correct attributes."})}


def run_owned(common, character, length, max_new_tokens, seed, temperature,
              common_weight, character_weight, pipe=None, expand_common=False):
    import comfy.model_management as mm
    from comfy.utils import ProgressBar
    with LOCK:
        model, tokenizer = acquire_model()
        def run(scene, person):
            bar = ProgressBar(max_new_tokens)
            return expand_owned(model, tokenizer, scene, person, length=length,
                max_new_tokens=max_new_tokens, seed=seed, temperature=temperature,
                common_weight=common_weight, character_weight=character_weight,
                interrupt=mm.throw_exception_if_processing_interrupted, progress=bar.update_absolute)
        if pipe is None:
            prompt, report = run(common, character if character.strip() else None)
            return prompt, report["added_tags"], json.dumps(report, ensure_ascii=False)
        result, scene, report = expand_pipe(pipe, run, common_prompt=common, expand_common=expand_common)
        info = {"model": MODEL_NAME, "revision": MODEL_REVISION,
                "backend": "native-pytorch-owned-attention", "seed": seed,
                "device": str(next(model.parameters()).device), "slots": report}
        return result, scene, json.dumps(info, ensure_ascii=False)


class RSTIPONativeControlledExpand:
    CATEGORY = "RS/TIPO Native"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt", "added_tags", "info")
    FUNCTION = "expand"
    DESCRIPTION = ("Experimental token-span attention control. Read common scene and one character with "
                   "separate multipliers. Source text/weights are preserved; invented contradictions remain possible.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "common_prompt": ("STRING", {"multiline": True, "default": "university campus, autumn, afternoon"}),
            "character_prompt": ("STRING", {"multiline": True, "default": "blonde hair, blue eyes, pink cardigan"}),
            **control_inputs()}}

    def expand(self, common_prompt, character_prompt, **settings):
        return run_owned(common_prompt, character_prompt, **settings)


PIPE_TYPES = "RS_MC_BUILD_PIPE,RS_MC_CHARACTER_PIPE,NAI_CONTEXT"


class RSTIPONativePipeExpand:
    CATEGORY = "RS/TIPO Native"
    RETURN_TYPES = (PIPE_TYPES, "STRING", "STRING")
    RETURN_NAMES = ("building_pipe", "common_prompt", "info")
    FUNCTION = "expand"
    DESCRIPTION = ("Expand each active character in an independent TIPO cache with weighted common/own attention. "
        "Preserves masks, positions, LoRAs, text slots and negatives. Put after Character Prompts; "
        "other builder roles may be before or after. Common expansion is optional and may invent appearance tags.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"building_pipe": (PIPE_TYPES,), **control_inputs(),
            "expand_common": ("BOOLEAN", {"default": False,
                "tooltip": "Also expand the scene. Experimental: TIPO may add character attributes to the common caption."})},
            "optional": {"common_prompt": ("STRING", {"multiline": True, "default": "",
                "tooltip": "Empty uses the pipe scene. Draft pipes without a source return this scene as STRING; connect it to the sampler."})}}

    def expand(self, building_pipe, common_prompt="", expand_common=False, **settings):
        return run_owned(common_prompt, "", pipe=building_pipe, expand_common=expand_common, **settings)
