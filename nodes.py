import json

from .generation import complete
from .expansion import expand_owned, expand_pipe, append_completion, finished_completion
from .runtime import LOCK, MODEL_NAME, MODEL_REVISION, acquire_model
from .exclusion import parse_excluded, validate_sources
from .metadata import metadata_inputs, metadata_fields, UNSPECIFIED
from .tipo_native_lora import adapter_choices


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
    # Keep serialized class IDs and socket indexes for saved workflows.
    DEPRECATED = True
    CATEGORY = "RS/TIPO Native/Legacy"
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
    # Keep serialized class IDs and socket indexes for saved workflows.
    DEPRECATED = True
    CATEGORY = "RS/TIPO Native/Legacy"
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
    # Keep serialized class IDs and socket indexes for saved workflows.
    DEPRECATED = True
    CATEGORY = "RS/TIPO Native/Legacy"
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
    # Keep serialized class IDs and socket indexes for saved workflows.
    DEPRECATED = True
    CATEGORY = "RS/TIPO Native/Legacy"
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
        "Choose Gemma TIPO (one adapted Gemma, Korean/English) or standalone TIPO 2.1 (English). Missing models download on first use. 0–10 characters; out-of-count inputs are ignored. Optional incoming pipe keeps LoRA, mask, position and camera settings. "
        "Review original, output and model details in the result panel. Experimental: meanings and roles can still be omitted or changed.")

    @classmethod
    def INPUT_TYPES(cls):
        from .language import MODES
        try:
            from .gemma_runtime import model_choices
            language_models = model_choices()
        except (ImportError, KeyError):  # Import/schema checks outside ComfyUI.
            language_models = ['auto']
        return {"required": {
            "character_count": ("INT", {"default": 2, "min": 0, "max": 10,
                "tooltip": "0 = common prompt only. Only slots 1–N are expanded; saved connections above N are ignored."}),
            "expand_common": ("BOOLEAN", {"default": False,
                "tooltip": "Rewrite the shared scene in the selected English output format. Off returns it unchanged. Review generated details."}),
            **control_inputs(),
            "advanced": ("BOOLEAN", {"default": False,
                "tooltip": "Show Gemma base, token limit, seed and temperature. Does not reset their values."}),
        }, "optional": {
            "common_prompt": ("STRING", {"forceInput": True, "multiline": True,
                "tooltip": "Shared scene and interaction. Character inputs supply each participant’s appearance and role; their action context is gathered automatically."}),
            **{f"character{i}": ("STRING", {"forceInput": True, "multiline": True,
                "tooltip": f"Character {i}: positive STRING prompt. Only active slots are read. Review facts and roles in the output. / {i}번 긍정 프롬프트. 활성 번호만 읽습니다."}) for i in range(1, 11)},
            "protect_source": ("BOOLEAN", {"default": True,
                "tooltip": "Keep supplied roles and appearance. Add action details and limited basic posture/gaze support; reject unrelated props and redundant tags. Limited English vocabulary. / 역할과 외형을 지키며 행동·기본 자세를 보강. 무관한 소품과 중복은 줄입니다."}),
            "building_pipe": (PIPE_TYPES, {"tooltip": "Optional: retain existing LoRA, masks, positions, references and deferred cameras. STRING inputs replace the prompt table."}),
            **exclusion_input(),
            # Append new widgets so old positional widgets_values retain the
            # excluded_tags index even without our named restore properties.
            "max_added_tags": ("INT", {"default": 6, "min": 1, "max": 32,
                "tooltip": "Requested detail budget per field in prose or tags; not a target to fill. / 자연어·태그의 필드별 묘사 보강 요청량입니다."}),
            **metadata_inputs(),
            "input_language": (list(MODES), {"default": "auto",
                "tooltip": "Legacy saved control; ignored. All languages use the selected Gemma base with its TIPO LoRA. / 이전 워크플로 호환용 숨김 항목."}),
            "language_model": (language_models, {"default": "auto",
                "tooltip": "Gemma TIPO only: auto uses a local unmerged Gemma 4 E2B or downloads Q4. Ignored by TIPO 2.1. / Gemma TIPO 전용 베이스 선택."}),
            "translation_max_tokens": ("INT", {"default": 2048, "min": 256, "max": 8192,
                "tooltip": "Legacy saved control; ignored. Use generation_max_tokens. / 이전 번역 설정은 사용하지 않습니다."}),
            "tipo_lora": (adapter_choices(), {"default": "pilot-002",
                "tooltip": "TIPO LoRA on the same Gemma; published choices download on first run. scale-20pct-001: experimental, CC BY-NC 4.0. local:* uses installed weights. / 공개본은 첫 실행 때 자동 다운로드. scale-20pct-001은 비상업용 실험본. 결과 패널에서 적용본 확인."}),
            "output_format": (["tags", "natural_language"], {"default": "tags",
                "tooltip": "Output style for the same common/character STRING sockets. / 같은 STRING 출력의 태그·자연어 형식."}),
            "generation_max_tokens": ("INT", {"default": 2048, "min": 256, "max": 8192,
                "tooltip": "Gemma: joint response limit. TIPO 2.1: per-output limit, input + output <=4096. / 모델의 문맥 상한을 넘으면 오류로 안내합니다."}),
            "model_family": (["gemma_tipo", "tipo_2_1"], {"default": "gemma_tipo",
                "tooltip": "Gemma TIPO: adapted Gemma for Korean/English. TIPO 2.1: standalone English model. Selected assets download once; no chained models. / 실행할 단일 모델 계열 선택."}),
        }, "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"}}

    def expand(self, character_count=2, common_prompt="", expand_common=False,
               building_pipe=None, prompt=None, unique_id=None, advanced=False,
               protect_source=True, excluded_tags="", max_added_tags=6,
               rating=UNSPECIFIED, quality=UNSPECIFIED, quality_era=UNSPECIFIED,
               input_language='auto', language_model='auto', translation_max_tokens=2048,
               tipo_lora='pilot-002', output_format='tags', generation_max_tokens=2048,
               model_family='gemma_tipo', **values):
        from .interface import validate_inputs, pipe_adapter, make_building_pipe, pipe_is_requested
        from .gemma_runtime import run_scene, status_context
        characters = validate_inputs(common_prompt, character_count, values)
        excluded = parse_excluded(excluded_tags)
        if model_family not in ('gemma_tipo', 'tipo_2_1'):
            raise ValueError('Select Gemma TIPO or TIPO 2.1. / 모델 계열을 선택해 주세요.')
        if model_family == 'tipo_2_1':
            from .tipo_runtime import validate_request
            validate_request(common_prompt, characters, excluded, output_format, generation_max_tokens)
        validate_sources(excluded, [('common_prompt', common_prompt),
            *((f'character {i}', text) for i, text in enumerate(characters, 1))])
        if type(max_added_tags) is not int or not 1 <= max_added_tags <= 32:
            raise ValueError('TIPO max_added_tags must be between 1 and 32.')
        if type(generation_max_tokens) is not int or not 256 <= generation_max_tokens <= 8192:
            raise ValueError('TIPO generation_max_tokens must be between 256 and 8192.')
        if not characters and pipe_is_requested(prompt, unique_id):
            raise ValueError('TIPO: a building pipe needs at least one character. Use common_prompt STRING for a scene-only expansion.')
        metadata = metadata_fields(rating=rating, quality=quality, quality_era=quality_era)
        adapter = pipe_adapter()
        if adapter is None and (building_pipe is not None or pipe_is_requested(prompt, unique_id)):
            raise ValueError('TIPO building pipe needs ComfyUI_RS_MultiCharacter_Sampler. Install it, or use STRING outputs.')
        draft = make_building_pipe(adapter, building_pipe, common_prompt, characters) if adapter else None
        seed = values.get('seed', 0)
        temperature = values.get('temperature', .3)
        with LOCK, status_context(unique_id):
            try:
                settings = dict(output_format=output_format,
                    length=values.get('length', 'long'), max_added_tags=max_added_tags,
                    expand_common=expand_common, excluded=excluded, metadata=metadata,
                    max_tokens=generation_max_tokens, temperature=temperature, seed=seed)
                if model_family == 'gemma_tipo':
                    scene, expanded, model_info = run_scene(common_prompt, characters,
                        model=language_model, adapter=tipo_lora, **settings)
                else:
                    from .tipo_runtime import run_scene as run_tipo
                    scene, expanded, model_info = run_tipo(common_prompt, characters,
                        protect_source=protect_source, **settings)
            except BaseException:
                from .gemma_runtime import status
                status('stopped')
                raise
        # English tag sources are immutable. Use the generated text only for
        # additions, not as authority to replace an identity, weight or role.
        if output_format == 'tags':
            from .language import has_korean
            from .expansion import append_completion, split_tags
            def preserve(source, output):
                if has_korean(source):
                    return output
                combined, added = append_completion(source, output)
                if not added:
                    return source
                additions = split_tags(added)[:max_added_tags]
                return source.rstrip().rstrip(',') + ', ' + ', '.join(additions)
            scene = preserve(common_prompt, scene) if expand_common else common_prompt
            expanded = tuple(preserve(source, output) for source, output in zip(characters, expanded))
        pipe = make_building_pipe(adapter, draft, scene, expanded) if adapter else None
        def slot_report(slot, source, output, enabled=True):
            row = {'slot': slot, 'original': source, 'output': output}
            if output_format == 'tags':
                from .language import has_korean
                if has_korean(source):
                    row['expansion_status'] = 'translated' if enabled else 'disabled'
                else:
                    _, added = append_completion(source, output)
                    additions = split_tags(added)
                    row.update(added_tags=additions, added_tag_count=len(additions),
                        expansion_status=('disabled' if not enabled else
                            'expanded' if additions else 'unchanged'))
            return row
        reports = [slot_report('common', common_prompt, scene, expand_common)]
        reports += [slot_report(i, source, output)
                    for i, (source, output) in enumerate(zip(characters, expanded), 1)]
        info = {**model_info, 'model_family': 'Gemma TIPO' if model_family == 'gemma_tipo' else 'TIPO 2.1', 'output_format': output_format,
            'character_count': character_count, 'building_pipe_available': pipe is not None,
            'context_source': ('joint_common_and_all_characters' if model_family == 'gemma_tipo' else 'owner_separated_with_shared_action_context'), 'slots': reports, 'metadata': metadata,
            'max_added_tags': max_added_tags,
            'legacy_controls_ignored': ['common_weight', 'character_weight', 'max_new_tokens',
                'input_language', 'translation_max_tokens'] + (['protect_source'] if model_family == 'gemma_tipo' else ['language_model', 'tipo_lora'])}
        return {'ui': {'tipo_report': [json.dumps(info, ensure_ascii=False)]},
                'result': (pipe, scene, *expanded, *('' for _ in range(10-len(expanded))))}
