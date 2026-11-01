"""Check installed MC/NAI immutable classes and conversion; no API requests."""
import argparse
from dataclasses import replace
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("rs_tipo_native", ROOT / "__init__.py",
                                            submodule_search_locations=[str(ROOT)])
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
from rs_tipo_native.expansion import expand_pipe
from rs_tipo_native.interface import make_building_pipe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comfy", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for name, folder in (("fixture_mc", "comfyui-rs-multicharacter-sampler"),
                         ("fixture_nai", "ComfyUI_RS_NAI_API_Request")):
        module = types.ModuleType(name)
        module.__path__ = [str(args.comfy / "custom_nodes" / folder)]
        sys.modules[name] = module
    # Import the actual installed definitions, skipping package __init__ and
    # full extension/ComfyUI initialization. Stub only the node registry bridge.
    chars = importlib.import_module("fixture_mc.characters")
    builder = importlib.import_module("fixture_mc.character_builder")
    mask_module = importlib.import_module("fixture_mc.region_mask")
    nai = importlib.import_module("fixture_nai.generation_context")
    bridge = importlib.import_module("fixture_nai.local_bridge")
    nodes = types.ModuleType("nodes")
    nodes.NODE_CLASS_MAPPINGS = {"RSMultiCharacterPrompts": chars.RSMultiCharacterPrompts}
    sys.modules["nodes"] = nodes

    region = mask_module.RegionMask(4, 4, bytes([255, 255]))
    first = chars.Character("(blonde hair:1.2), blue eyes", "no glasses", .5, .5,
        lora_name="first.safetensors", region_mask=region)
    second = chars.Character("black hair, brown eyes", "no hat", .75, .5,
        lora_name="second.safetensors", bbox=(.6, .2, .9, .8))
    literal = chars.TextRegion("large white lettering", "bad spelling", .5, .15,
        (.2, .05, .8, .25), "HELLO")
    source = chars.CharacterPipe((first, second), "campus", "low quality",
        text_regions=(literal,), interaction_json='{"preset":"custom","delay_steps":6,"decrement":0.2}')
    def run(scene, person):
        return (scene if person is None else person) + ", sunshine", {"added_tags": "sunshine"}

    cases = []
    edited, common, _ = expand_pipe(source, run, expand_common=True)
    assert edited.prompt == "campus, sunshine" and common == edited.prompt
    assert source.characters[0].prompt == first.prompt
    assert edited.characters[0].region_mask is region
    for original, updated in zip(source.characters, edited.characters):
        assert replace(updated, prompt=original.prompt) == original
    assert edited.text_regions[0] is literal
    assert replace(edited, prompt=source.prompt, characters=source.characters) == source
    cases.append("actual final CharacterPipe: all non-prompt fields preserved")

    draft = importlib.import_module("fixture_mc.pipe_compat").building_pipe(source)
    draft = replace(draft, camera_settings=((('yaw', 30.),),))
    edited, common, _ = expand_pipe(draft, run, expand_common=True)
    assert edited.camera_settings == draft.camera_settings
    # Don't apply deliberately minimal camera options; verify normal finalization.
    final = builder.finalize(edited, apply_cameras=False)
    assert final.characters[0].region_mask.tensor().equal(region.tensor())
    assert final.characters[0].lora_name == first.lora_name
    assert final.text_regions == source.text_regions
    assert final.prompt == common
    cases.append("actual sourced BuildPipe: finalization preserves mask/LoRA/text/scene")

    context = nai.NAIContext(prompt="campus", negative_prompt="low quality",
        model="NovelAI Diffusion V5 Full", sampler="k_euler_ancestral", steps=30,
        cfg_scale=5., scheduler="native", cfg_rescale=0., prefer_brownian=True,
        variety_boost=False, characters=tuple(bridge.nai_character_list(source)),
        source_seed=12345, source_action="generate", source_size=(1024, 1024),
        local_record_json=bridge.local_record(source))
    edited, _, _ = expand_pipe(context, run, expand_common=True)
    assert type(edited) is nai.NAIContext
    assert edited.local_record_json == context.local_record_json
    assert replace(edited, prompt=context.prompt, characters=context.characters) == context
    restored = edited.as_local_character_pipe()
    assert restored.characters[0].region_mask.tensor().equal(region.tensor())
    assert restored.characters[0].lora_name == first.lora_name
    assert "sunshine" in restored.characters[0].prompt
    assert "sunshine" in restored.prompt
    assert restored.text_regions == source.text_regions
    cases.append("actual NAIContext: settings/snapshot preserved; bridge restores edited prompt and original mask/LoRA/text")
    compat = importlib.import_module("fixture_mc.pipe_compat")
    adapter = (builder, compat)
    assembled = make_building_pipe(adapter, None, "scene expanded", ("first expanded", "second expanded"))
    final = builder.finalize(assembled)
    assert final.prompt == "scene expanded"
    assert tuple(c.prompt for c in final.characters) == ("first expanded", "second expanded")
    for incoming in (draft, source, context):
        assembled = make_building_pipe(adapter, incoming, "scene expanded", ("first expanded", "second expanded"))
        final = builder.finalize(assembled, apply_cameras=False)
        assert final.prompt == "scene expanded" and len(final.characters) == 2 and not final.text_regions
        assert final.characters[0].region_mask.tensor().equal(region.tensor())
        assert final.characters[0].lora_name == first.lora_name
        assert final.characters[0].uc == first.uc
        assert final.interaction_json == compat.building_pipe(incoming).interaction_json
        if incoming is draft:
            assert assembled.camera_settings == draft.camera_settings
    short = make_building_pipe(adapter, assembled, "scene", ("only first",))
    assert len(builder.finalize(short, apply_cameras=False).characters) == 1
    assert make_building_pipe(adapter, None, "scene", ()) is None
    cases.append("STRING interface: common + individual outputs finalize identically; incoming BuildPipe/final/NAI geometry and metadata preserved; count shrink excludes stale slots")
    args.output.write_text(json.dumps({"cases": cases, "all_passed": True,
        "scope": "real installed classes and conversion with deterministic text callback; no TIPO/image/NovelAI inference"},
        ensure_ascii=False, indent=2) + "\n")
    print("PIPE PROTOCOL PASS", *cases, sep="\n")


if __name__ == "__main__":
    main()
