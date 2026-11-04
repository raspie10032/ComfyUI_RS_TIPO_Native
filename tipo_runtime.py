"""Standalone TIPO 2.1 on the same STRING/pipe interface, without Gemma."""
from dataclasses import asdict
import re

from .generation import complete
from .expansion import expand_owned
from .exclusion import normalize, validate_sources
from .grounding import model_text
from .scene_context import scene_context
from .download import MODEL_NAME, MODEL_REVISION


def validate_request(common, characters, excluded, output_format, max_tokens):
    if output_format not in ('tags', 'natural_language'):
        raise ValueError('Unknown TIPO 2.1 output format.')
    if type(max_tokens) is not int or not 1 <= max_tokens <= 4096:
        raise ValueError('TIPO 2.1 supports up to 4096 output tokens; its input and output share a 4096-token context. / 출력 상한을 4096 이하로 줄여 주세요.')
    if any(re.search(r'[\u1100-\u11ff\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]', s)
           for s in (common, *characters, *excluded)):
        raise ValueError('TIPO 2.1 needs English prompts and exclusions. Select Gemma TIPO for Korean input. No translation model is loaded. / 한국어는 Gemma TIPO를 선택해 주세요.')
    validate_sources(excluded, [('common_prompt', common),
        *((f'character {i}', s) for i, s in enumerate(characters, 1))])


def run_scene(common, characters, *, output_format='tags', length='long',
              max_added_tags=6, expand_common=False, excluded=(), metadata=None,
              max_tokens=2048, temperature=.3, seed=0, protect_source=True,
              common_weight=.5, character_weight=1.5):
    validate_request(common, characters, excluded, output_format, max_tokens)
    from .runtime import acquire_model
    from .gemma_runtime import status
    import comfy.model_management as mm
    from comfy.utils import ProgressBar
    status('tipo-loading')
    model, tokenizer = acquire_model()
    status('tipo-generating')
    calls = []
    header = ''.join(f'{k}: {v}\n' for k, v in (metadata or {}).items())

    def run(source, index):
        bar = ProgressBar(max_tokens)
        sampling = dict(max_new_tokens=max_tokens, temperature=temperature,
            seed=(seed + (0 if index is None else index + 1)) % (2**64),
            excluded_tags=', '.join(excluded),
            interrupt=mm.throw_exception_if_processing_interrupted,
            progress=bar.update_absolute)
        context, roles = scene_context(common, characters, index)
        if output_format == 'tags':
            settings = {}
            if metadata:
                if 'rating' in metadata:
                    settings['rating'] = 'explicit' if metadata['rating'] == 'nsfw, explicit' else metadata['rating']
                if 'quality' in metadata:
                    quality = metadata['quality'].split(', ')
                    settings['quality'] = quality[0]
                    if len(quality) > 1:
                        settings['quality_era'] = quality[1]
            output, report = expand_owned(model, tokenizer, common,
                None if index is None else source, length=length,
                common_weight=common_weight, character_weight=character_weight,
                interaction=context, peer_actions=tuple(c for j, role in enumerate(roles)
                    if j != index for c in role), protect_source=protect_source,
                max_added_tags=max_added_tags, grounded_additions=True, **settings, **sampling)
        else:
            # Use the learned tag_to_long protocol, never a Gemma chat prompt.
            # Other participants' appearance is deliberately absent.
            tags = model_text(source)
            # A global "two women" prefix makes tag_to_long describe a whole
            # scene even when this output belongs to one participant. Keep the
            # character caption owner-local; don't seed it with the headcount.
            prompt = header + f'target: <|{length}|> <|tag_to_long|>\ntag: {tags}\nlong:'
            result = complete(model, tokenizer, prompt, **sampling)
            if result.finish_reason == 'length':
                raise ValueError('TIPO 2.1 caption reached its token limit. Increase generation_max_tokens within the context limit. / 자연어 출력이 잘렸습니다.')
            output = result.text.split('<|eos|>', 1)[0].split('\n', 1)[0].strip()
            if len(output.split()) < 4:
                raise ValueError('TIPO 2.1 returned an empty or incomplete caption. / 자연어 출력이 불완전합니다.')
            for phrase in re.findall(r'\([^()]+:\s*[+-]?(?:\d+(?:\.\d*)?|\.\d+)\s*\)', source):
                if phrase not in output:
                    raise ValueError('TIPO 2.1 caption lost a weighted phrase; use tags or Gemma TIPO. / 가중치 구절이 누락됐습니다. 태그 또는 Gemma TIPO를 사용해 주세요.')
            report = {**asdict(result), 'input': prompt}
        if any(phrase in normalize(output) for phrase in excluded):
            raise ValueError('TIPO 2.1 output contains an excluded phrase. / 제외 구절이 출력에 포함됐습니다.')
        calls.append({'slot': 'common' if index is None else index + 1, **report})
        return output

    scene = run(common, None) if expand_common and common.strip() else common
    slots = tuple(run(source, i) for i, source in enumerate(characters))
    return scene, slots, dict(model=MODEL_NAME, revision=MODEL_REVISION,
        backend='native-pytorch-comfy-modelpatcher', model_family='TIPO 2.1',
        model_calls=len(calls), seconds=round(sum(c['seconds'] for c in calls), 4),
        raw_output='\n\n'.join(c['text'] for c in calls), calls=calls,
        limitations=('English only. Tags share recognized action context. Natural-language captions use owner-local input; '
            'TIPO can invent appearance or lose actions and does not guarantee role or appearance preservation. '
            'For coordinated character prose, prefer Gemma TIPO.'))
