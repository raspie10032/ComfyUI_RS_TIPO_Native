import json

from .generation import complete
from .expansion import expand_owned, expand_pipe, append_completion, finished_completion
from .runtime import LOCK, MODEL_NAME, MODEL_REVISION, acquire_model
from .exclusion import parse_excluded, validate_sources


def exclusion_input():
    return {"excluded_tags": ("STRING", {"multiline": True, "default": "",
        "tooltip": "Comma/newline-separated tags to block during token selection, for all active outputs. Not post-removal. Exact normalized tags; no synonym inference. Source conflicts are errors. / 생성 단계 제외 태그: 쉼표·줄바꿈으로 구분. 원문과 충돌하면 오류."})}


def run_completion(prompt, max_new_tokens, seed, temperature, excluded_tags=""):
    import comfy.model_management as mm
    from comfy.utils import ProgressBar
    bar = ProgressBar(max_new_tokens)
    validate_sources(parse_excluded(excluded_tags), [('source tag field', prompt.rsplit('tag:', 1)[-1])])
    with LOCK:
        model, tokenizer = acquire_model()
        result = complete(model, tokenizer, prompt, max_new_tokens=max_new_tokens,
                          seed=seed, temperature=temperature,
                          excluded_tags=excluded_tags,
                          interrupt=mm.throw_exception_if_processing_interrupted,
                          progress=bar.update_absolute)
        info = json.dumps({"model": MODEL_NAME, "revision": MODEL_REVISION,
                           "backend": "native-pytorch-comfy-modelpatcher",
                           "device": str(next(model.parameters()).device),
                           "input_tokens": result.input_tokens,
                           "output_tokens": len(result.token_ids),
                           "finish_reason": result.finish_reason,
                           "seconds": round(result.seconds, 4), "seed": seed,
                           "exclusion": result.exclusion}, ensure_ascii=False)
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
        }, "optional": exclusion_input()}

    def generate(self, prompt, max_new_tokens, seed, temperature, excluded_tags=""):
        return run_completion(prompt, max_new_tokens, seed, temperature, excluded_tags)


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
        }, "optional": exclusion_input()}

    def expand(self, tags, length, max_new_tokens, seed, temperature, excluded_tags=""):
        if not tags.strip():
            raise ValueError("Enter at least one tag before running TIPO.")
        if length not in ("short", "long", "very_long"):
            raise ValueError("Unknown TIPO target length.")
        # This baseline's official tag-only format remains separate from semantic control research.
        # Do not flatten multiple characters/pipes into this field.
        input_prompt = f"target: <|{length}|>\ntag: {tags.rstrip().rstrip(',')}, "
        text, info = run_completion(input_prompt, max_new_tokens, seed, temperature, excluded_tags)
        # Only the first tag field belongs in a tag string; preserve all raw output in metadata.
        meta = json.loads(info)
        meta["raw_completion"] = text
        combined, generated = append_completion(tags, finished_completion(text, meta["finish_reason"]))
        return combined, generated, json.dumps(meta, ensure_ascii=False)


def control_inputs():
    base = RSTIPONativeExpand.INPUT_TYPES()["required"]
    return {**{name: base[name] for name in ("length", "max_new_tokens", "seed", "temperature")},
            "common_weight": ("FLOAT", {"default": 0.5, "min": 0., "max": 4., "step": 0.1,
                "tooltip": "Common source-token attention multiplier. 0 blocks it for the character. Not an image prompt weight."}),
            "character_weight": ("FLOAT", {"default": 1.5, "min": 0., "max": 4., "step": 0.1,
                "tooltip": "Character source-token attention multiplier. 1 is neutral. Stronger does not guarantee correct attributes."})}


def run_owned(common, character, length, max_new_tokens, seed, temperature,
              common_weight, character_weight, pipe=None, expand_common=False, excluded_tags=""):
    import comfy.model_management as mm
    from comfy.utils import ProgressBar
    sources = [('common_prompt', common), ('character', character)]
    if pipe is not None:
        source = getattr(pipe, 'source_pipe', None) if hasattr(pipe, 'slots') else pipe
        if not common.strip():
            sources.append(('pipe common', getattr(source, 'prompt', '') or ''))
        for i, row in enumerate(getattr(pipe, 'slots', getattr(pipe, 'characters', ())), 1):
            if hasattr(pipe, 'slots') and (pipe.prompt_count is None or row.number > pipe.prompt_count or not row.enabled or row.kind != 'character'):
                continue
            sources.append((f'character {i}', row.prompt))
    validate_sources(parse_excluded(excluded_tags), sources)
    with LOCK:
        model, tokenizer = acquire_model()
        def run(scene, person):
            bar = ProgressBar(max_new_tokens)
            return expand_owned(model, tokenizer, scene, person, length=length,
                max_new_tokens=max_new_tokens, seed=seed, temperature=temperature,
                common_weight=common_weight, character_weight=character_weight,
                excluded_tags=excluded_tags,
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
            **control_inputs()}, "optional": exclusion_input()}

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
                "tooltip": "Empty uses the pipe scene. Draft pipes without a source return this scene as STRING; connect it to the sampler."}),
                **exclusion_input()}}

    def expand(self, building_pipe, common_prompt="", expand_common=False, **settings):
        return run_owned(common_prompt, "", pipe=building_pipe, expand_common=expand_common, **settings)


class RSTIPOPrompt:
    CATEGORY = "RS/TIPO Native"
    RETURN_TYPES = ("RS_MC_BUILD_PIPE", "STRING") + ("STRING",) * 10
    RETURN_NAMES = ("building_pipe", "common_prompt") + tuple(f"character{i}" for i in range(1, 11))
    FUNCTION = "expand"
    OUTPUT_NODE = True
    DESCRIPTION = ("Expand character STRINGs using the common scene and action context derived automatically from all active character inputs. Outputs the same results as STRINGs and an RS building pipe. "
        "0–10 characters; out-of-count inputs are ignored. Optional incoming pipe keeps LoRA, mask, position and camera settings. "
        "Source protection conservatively rejects unrequested appearance, composition and recognized role conflicts. Experimental: this is not a general relationship planner.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "character_count": ("INT", {"default": 2, "min": 0, "max": 10,
                "tooltip": "0 = common prompt only. Only slots 1–N are expanded; saved connections above N are ignored."}),
            "expand_common": ("BOOLEAN", {"default": False,
                "tooltip": "Expand the shared scene too. Source protection removes recognized participant-specific additions; review remaining tags."}),
            **control_inputs(),
            "advanced": ("BOOLEAN", {"default": False,
                "tooltip": "Show token limit, seed, temperature and source-attention weights. Does not reset their values."}),
        }, "optional": {
            "common_prompt": ("STRING", {"forceInput": True, "multiline": True,
                "tooltip": "Shared scene and interaction. Character inputs supply each participant’s appearance and role; their action context is gathered automatically."}),
            **{f"character{i}": ("STRING", {"forceInput": True, "multiline": True,
                "tooltip": f"Character {i}: positive prompt. Original text and weights are preserved."}) for i in range(1, 11)},
            "protect_source": ("BOOLEAN", {"default": True,
                "tooltip": "Keep supplied roles and appearance. Add action details and limited basic posture/gaze support; reject unrelated props and redundant tags. Limited English vocabulary. / 역할과 외형을 지키며 행동·기본 자세를 보강. 무관한 소품과 중복은 줄입니다."}),
            "building_pipe": (PIPE_TYPES, {"tooltip": "Optional: retain existing LoRA, masks, positions, references and deferred cameras. STRING inputs replace the prompt table."}),
            **exclusion_input(),
            # Append new widgets so old positional widgets_values retain the
            # excluded_tags index even without our named restore properties.
            "max_added_tags": ("INT", {"default": 6, "min": 1, "max": 32,
                "tooltip": "Maximum added tags per output, not a target to fill. Does not change the decoder token budget. / 출력별 추가 태그 상한입니다. 개수를 채우려고 추가하지 않습니다."}),
        }, "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"}}

    def expand(self, character_count=2, common_prompt="", expand_common=False,
               building_pipe=None, prompt=None, unique_id=None, advanced=False,
               protect_source=True, excluded_tags="", max_added_tags=6, **values):
        from .interface import (validate_inputs, expand_strings, pipe_adapter,
                                make_building_pipe, pipe_is_requested)
        characters = validate_inputs(common_prompt, character_count, values)
        if type(max_added_tags) is not int or not 1 <= max_added_tags <= 32:
            raise ValueError('TIPO max_added_tags must be between 1 and 32. / 추가 태그 상한은 1~32입니다.')
        validate_sources(parse_excluded(excluded_tags), [('common_prompt', common_prompt),
            *((f'character {i}', text) for i, text in enumerate(characters, 1))])
        if not characters and pipe_is_requested(prompt, unique_id):
            raise ValueError("TIPO: a building pipe needs at least one character. Use common_prompt STRING for a scene-only expansion. / 빌딩 파이프는 최소 1명이 필요합니다. 전역만 쓸 때는 STRING 출력을 연결해 주세요.")
        adapter = pipe_adapter()
        if adapter is None and (building_pipe is not None or pipe_is_requested(prompt, unique_id)):
            raise ValueError("TIPO building pipe needs ComfyUI_RS_MultiCharacter_Sampler. Install it, or use the individual STRING outputs. / 빌딩 파이프는 RS 멀티캐릭터 확장이 필요합니다. STRING 출력은 단독 사용 가능합니다.")
        # Validate/normalize pipe before allocating the TIPO model.
        draft = make_building_pipe(adapter, building_pipe, common_prompt, characters) if adapter else None
        settings = {name: values.get(name, schema[1]["default"] if len(schema) > 1 else schema[0][0])
                    for name, schema in control_inputs().items()}
        import comfy.model_management as mm
        from comfy.utils import ProgressBar
        from .scene_context import scene_context
        shared_context, roles = scene_context(common_prompt, characters)
        slot_index = -1
        with LOCK:
            model, tokenizer = acquire_model()
            def run(scene, person):
                nonlocal slot_index
                if person is not None:
                    slot_index += 1
                context, _ = scene_context(scene, characters, slot_index if person is not None else None)
                bar = ProgressBar(settings["max_new_tokens"])
                output, report = expand_owned(model, tokenizer, scene, person, interaction=context,
                    interaction_weight=settings["common_weight"] if person is not None else 1.0,
                    protect_source=protect_source,
                    max_added_tags=max_added_tags, grounded_additions=True,
                    excluded_tags=excluded_tags,
                    peer_actions=tuple(action for i, actions in enumerate(roles) if i != slot_index for action in actions) if person is not None else (),
                    **settings,
                    interrupt=mm.throw_exception_if_processing_interrupted, progress=bar.update_absolute)
                report["context_source"] = "common_and_character_inputs"
                return output, report
            scene, expanded, reports = expand_strings(common_prompt, characters, run, expand_common=expand_common)
        pipe = make_building_pipe(adapter, draft, scene, expanded) if adapter else None
        info = {"model": MODEL_NAME, "revision": MODEL_REVISION,
                "protect_source": protect_source,
                "max_added_tags": max_added_tags,
                "excluded_tags": list(parse_excluded(excluded_tags)),
                "character_count": character_count, "shared_context": shared_context,
                "participant_roles": roles, "context_source": "common_and_character_inputs",
                "building_pipe_available": pipe is not None,
                "seconds": round(sum(row.get("seconds", 0.) for row in reports), 4), "slots": reports}
        return {"ui": {"tipo_report": [json.dumps(info, ensure_ascii=False)]},
                "result": (pipe, scene, *expanded, *("" for _ in range(10-len(expanded))))}
