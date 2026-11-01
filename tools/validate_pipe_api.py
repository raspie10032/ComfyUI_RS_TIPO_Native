"""Smoke controlled nodes through a running ComfyUI without image/API generation."""
import argparse
import json
from pathlib import Path
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8188")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    def get(path):
        return json.load(urllib.request.urlopen(args.url + path, timeout=5))

    def post(graph):
        req = urllib.request.Request(args.url + "/prompt", json.dumps({"prompt": graph,
            "client_id": "rs-tipo-owned-validation"}).encode(), headers={"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(req, timeout=5))["prompt_id"]

    info = get("/object_info")
    assert "RSTIPONativePipeExpand" in info and "RSTIPONativeControlledExpand" in info
    if any(get("/queue").values()):
        raise RuntimeError("ComfyUI has other queued jobs; run this smoke when idle.")

    def defaults(node):
        result = {}
        for name, schema in info[node]["input"].get("required", {}).items():
            options = schema[1] if len(schema) > 1 else {}
            if "default" in options:
                result[name] = options["default"]
            elif isinstance(schema[0], list):
                result[name] = schema[0][0]
        return result

    def node(kind, **values):
        return {"class_type": kind, "inputs": {**defaults(kind), **values}}

    report = {"url": args.url, "registered": [n for n in info if n.startswith("RSTIPO")], "cases": []}
    def run(case, graph):
        pid = post(graph)
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            records = get("/history/" + pid)
            if pid in records:
                record = records[pid]
                assert record["status"]["status_str"] == "success", record["status"]
                report["cases"].append({"case": case, "record": record})
                args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
                print(case, "PASS", flush=True)
                return record["outputs"]
            time.sleep(.25)
        raise TimeoutError(f"Inspect live prompt {pid}; do not resubmit after observation timeout.")

    common = "university campus, autumn, afternoon"
    first = "(blonde hair:1.2), blue eyes, pink cardigan"
    second = "black hair, brown eyes, yellow sweater"
    controlled = node("RSTIPONativeControlledExpand", common_prompt=common,
        character_prompt=first, max_new_tokens=32, seed=8910)
    out = run("controlled", {"1": controlled,
        "2": node("PreviewAny", source=["1", 0]), "3": node("PreviewAny", source=["1", 2])})
    assert out["2"]["text"][0].startswith(first)

    prompt_node = node("RSMultiCharacterPrompt", character_count=4,
        character1=first, character2=second, character3="HELLO", character3_kind="text",
        character4="hidden character", character4_enable=False)
    position = node("RSMultiCharacterPosition", character_count=3,
        character1_x=.25, character2_x=.75, character3_geometry="box",
        character3_x=.5, character3_y=.15, character3_width=.6, character3_height=.2)
    before_preview = None
    for after in (False, True):
        graph = {"1": prompt_node,
            "2": {"class_type": position["class_type"], "inputs": dict(position["inputs"])},
            "3": node("RSTIPONativePipeExpand", common_prompt=common, expand_common=True,
                max_new_tokens=32, seed=91),
            "4": node("RSMultiCharacterFinalize"),
            "5": node("PreviewAny", source=["4", 0]),
            "6": node("PreviewAny", source=["3", 1]),
            "7": node("PreviewAny", source=["3", 2])}
        graph["2"]["inputs"]["building_pipe"] = ["3" if after else "1", 0]
        graph["3"]["inputs"]["building_pipe"] = ["1" if after else "2", 0]
        graph["4"]["inputs"]["building_pipe"] = ["2" if after else "3", 0]
        out = run("position_after" if after else "position_before", graph)
        rendered = out["5"]["text"][0]
        assert first in rendered and second in rendered and "HELLO" in rendered
        assert "hidden character" not in rendered
        if after:
            assert rendered == before_preview, "Position order changed the finalized character pipe"
        else:
            before_preview = rendered
        details = json.loads(out["7"]["text"][0])
        assert [row["slot"] for row in details["slots"]] == ["common", 1, 2]
    # Final-pipe input has a different runtime class and connection validation path.
    graph["4"]["inputs"]["building_pipe"] = ["2", 0]
    graph["2"]["inputs"]["building_pipe"] = ["1", 0]
    graph["3"]["inputs"]["building_pipe"] = ["4", 0]
    graph["5"]["inputs"]["source"] = ["3", 0]
    out = run("final_pipe", graph)
    assert "HELLO" in out["5"]["text"][0]
    print("CONTROLLED COMFY PIPE API PASS", flush=True)


if __name__ == "__main__":
    main()
