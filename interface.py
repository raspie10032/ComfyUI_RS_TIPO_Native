"""STRING-facing orchestration, with an optional registered RS pipe adapter."""
import importlib
from dataclasses import replace

MAX_CHARACTERS = 10


def validate_inputs(common, count, values):
    if type(count) is not int or not 0 <= count <= MAX_CHARACTERS:
        raise ValueError("TIPO: choose 0–10 characters. / 캐릭터 수는 0~10입니다.")
    if not isinstance(common, str):
        raise ValueError("TIPO: connect a STRING to the common prompt. / 전역 프롬프트는 STRING입니다.")
    characters = []
    for i in range(1, count + 1):
        value = values.get(f"character{i}", "")
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"TIPO character {i}: connect a nonempty STRING, or reduce character_count. / {i}번 STRING을 연결하거나 캐릭터 수를 줄여 주세요.")
        characters.append(value)
    if not characters and not common.strip():
        raise ValueError("TIPO: connect a common prompt or at least one character. / 전역 또는 캐릭터 프롬프트를 연결해 주세요.")
    return tuple(characters)


def expand_strings(common, characters, run, *, expand_common=True):
    """Each output sees the original scene, never another generated caption."""
    scene, reports = common, []
    if expand_common and common.strip():
        scene, report = run(common, None)
        reports.append({"slot": "common", **report})
    for number, original in enumerate(characters, 1):
        expanded, report = run(common, original)
        reports.append({"slot": number, **report, "output": expanded})
    outputs = tuple(row["output"] for row in reports if row["slot"] != "common")
    return scene, outputs, reports


def pipe_adapter():
    # Use the registered extension's actual types; don't vendor a second pipe
    # definition or load either extension's __init__ again under an alias.
    import nodes
    cls = nodes.NODE_CLASS_MAPPINGS.get("RSMultiCharacterPrompt")
    if cls is None:
        return None
    module = importlib.import_module(cls.__module__)
    compat = importlib.import_module(cls.__module__.rsplit(".", 1)[0] + ".pipe_compat")
    return module, compat


def make_building_pipe(adapter, incoming, common, characters):
    if not characters:
        return None  # A scene-only STRING has no valid regional pipe.
    module, compat = adapter
    pipe = compat.building_pipe(incoming)
    pipe = module.apply_role(pipe, "prompt", len(characters), lambda i: dict(
        prompt=characters[i-1], enabled=True, kind="character", literal_text="")) if characters else pipe
    if pipe is None:
        pipe = module.BuildPipe()
    source = pipe.source_pipe
    if source is None:
        source = replace(module.finalize(pipe, apply_cameras=False), prompt=common)
    else:
        source = replace(source, prompt=common)
    return replace(pipe, source_pipe=source)


def pipe_is_requested(prompt, unique_id):
    if not isinstance(prompt, dict):
        return False
    return any(isinstance(link, (list, tuple)) and len(link) == 2
               and str(link[0]) == str(unique_id) and link[1] == 0
               for node in prompt.values() if isinstance(node, dict)
               for link in node.get("inputs", {}).values())
