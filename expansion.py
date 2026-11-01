"""Prompt-only editing: keep source strings and all non-prompt pipe fields."""
from dataclasses import asdict, is_dataclass, replace

from .control import controlled_prompt
from .generation import complete
from .grounding import guard_additions


def append_completion(original, completion):
    first = completion.split("<|eos|>", 1)[0].split("\n", 1)[0].strip(" ,")
    seen = {t.strip().casefold() for t in original.split(",")}
    added = []
    for tag in first.split(","):
        tag = tag.strip()
        if tag and tag.casefold() not in seen:
            seen.add(tag.casefold())
            added.append(tag)
    suffix = ", ".join(added)
    separator = " " if original.rstrip().endswith(",") else ", "
    return original + (separator + suffix if suffix and original.strip() else suffix), suffix


def finished_completion(completion, finish_reason):
    if finish_reason == "length" and "\n" not in completion and "<|eos|>" not in completion:
        completion = completion.rsplit(",", 1)[0] if "," in completion else ""
    return completion


def expand_owned(model, tokenizer, common, character=None, *, common_weight=0.5,
                 character_weight=1.5, length="long", interaction="", interaction_weight=1.0,
                 protect_source=True, **sampling):
    prompt, attention = controlled_prompt(tokenizer, common, character, length=length,
        common_weight=common_weight, character_weight=character_weight,
        interaction=interaction, interaction_weight=interaction_weight)
    result = complete(model, tokenizer, prompt, attention_control=attention, **sampling)
    original = common if character is None else character
    # A token cap can stop halfway through a tag. Only a model stop/EOS/newline
    # establishes that the final tag is complete; never send a cut-off fragment
    # into the image prompt.
    completion = finished_completion(result.text, result.finish_reason)
    _, proposed = append_completion(original, completion)
    rejected = []
    if protect_source:
        kept, rejected = guard_additions(original, proposed.split(","), common=common,
            interaction=interaction, character=character is not None)
        completion = ", ".join(kept)
    expanded, added = append_completion(original, completion)
    return expanded, {**asdict(result), "input": prompt, "original": original,
                      "added_tags": added, "output": expanded,
                      "common_weight": common_weight, "character_weight": character_weight,
                      "interaction": interaction, "interaction_weight": interaction_weight,
                      "protect_source": protect_source, "proposed_tags": proposed,
                      "rejected_tags": rejected,
                      "owner_tokens": {str(o): attention.owners.count(o) for o in set(attention.owners)}}


def expand_pipe(pipe, run, *, common_prompt="", expand_common=False):
    """Operate on the shared immutable pipe protocol without importing node packs.

    Draft builders retain inactive/text slots and deferred camera/pose settings.
    A final NAI context retains its original class, sampling settings and snapshots.
    """
    if not is_dataclass(pipe):
        raise ValueError("Connect an RS building/character pipe or NAI context.")
    builder = hasattr(pipe, "slots") and hasattr(pipe, "prompt_count")
    source = getattr(pipe, "source_pipe", None) if builder else pipe
    common = common_prompt if common_prompt.strip() else (getattr(source, "prompt", None) or "")
    report = []
    output_common = common
    if expand_common and common.strip():
        output_common, row = run(common, None)
        report.append({"slot": "common", **row})
    if builder:
        if pipe.prompt_count is None:
            raise ValueError("Connect Character Prompts before TIPO so its active slots are known.")
        rows = pipe.slots
        targets = [(i, s.number, s) for i, s in enumerate(rows)
                   if s.number <= pipe.prompt_count and s.enabled and s.kind == "character"]
    elif hasattr(pipe, "characters"):
        rows = pipe.characters
        targets = [(i, i + 1, c) for i, c in enumerate(rows)]
    else:
        raise ValueError("Unsupported RS pipe. Connect a prompt or final character pipe.")
    changed = list(rows)
    for index, number, character in targets:
        if not is_dataclass(character) or not isinstance(character.prompt, str) or not character.prompt.strip():
            raise ValueError(f"TIPO character {number}: connect a nonempty prompt first.")
        # All targets see the original common text, never another generated caption.
        prompt, row = run(common, character.prompt)
        changed[index] = replace(character, prompt=prompt)
        report.append({"slot": number, **row})
    if builder:
        updated_source = _replace_prompts(source, output_common, None) if source is not None else None
        result = replace(pipe, slots=tuple(changed), source_pipe=updated_source)
    else:
        result = _replace_prompts(pipe, output_common, tuple(changed))
    return result, output_common, report


def _replace_prompts(pipe, common, characters):
    changes = {"prompt": common if common or pipe.prompt is not None else None}
    if characters is not None:
        changes["characters"] = characters
    # NAI's immutable snapshot also carries geometry/provenance. Its bridge
    # restores that geometry and then overlays these authoritative captions.
    # Keep the snapshot intact, including its possibly different text syntax.
    return replace(pipe, **changes)
