"""API and model-lifecycle smoke test for an isolated ComfyUI test server."""
import argparse
import json
from pathlib import Path
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8196")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    def get(path):
        return json.load(urllib.request.urlopen(args.url + path, timeout=5))

    def post(path, data):
        req = urllib.request.Request(args.url + path, json.dumps(data).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as response:
            raw = response.read()
        return json.loads(raw) if raw else None

    def poll(pid):
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            h = get("/history/" + pid)
            if pid in h:
                return h[pid]
            time.sleep(0.25)
        raise TimeoutError(f"Still running; inspect existing prompt {pid}, do not resubmit.")

    def run(node):
        graph = {"1": node, "2": {"class_type": "PreviewAny", "inputs": {"source": ["1", 0]}},
                 "3": {"class_type": "PreviewAny", "inputs": {"source": ["1", 1]}}}
        if node["class_type"] == "RSTIPONativeExpand":
            graph["4"] = {"class_type": "PreviewAny", "inputs": {"source": ["1", 2]}}
        return post("/prompt", {"prompt": graph, "client_id": "rs-tipo-native-validation"})["prompt_id"]

    def completion(seed, limit=32, prompt=None):
        return {"class_type": "RSTIPONativeCompletion", "inputs": {
            "prompt": prompt or "target: <|very_long|>\ntag: 1girl, outdoors, long hair, ",
            "max_new_tokens": limit, "seed": seed, "temperature": 0.5}}

    infos = get("/object_info")
    assert all(n in infos for n in ("RSTIPONativeCompletion", "RSTIPONativeExpand"))
    queue = get("/queue")
    if queue["queue_running"] or queue["queue_pending"]:
        raise RuntimeError("Test server has other jobs; refusing lifecycle tests.")
    # Whitelisted isolated server only. Do not free models or interrupt the production instance.
    custom = [v for v in infos.values() if "custom_nodes" in v.get("python_module", "")]
    if any("ComfyUI_RS_TIPO_Native" not in v.get("python_module", "") for v in custom):
        raise RuntimeError("This must be an isolated server with only the TIPO extension enabled.")

    report = {"url": args.url, "registered": [n for n in infos if n.startswith("RSTIPO")]}
    cases = []
    for name, node in (
        ("greedy", {"class_type": "RSTIPONativeCompletion", "inputs": {
            "prompt": "target: <|long|>\ntag: 1girl, garden, ", "max_new_tokens": 64,
            "seed": 0, "temperature": 0}}),
        ("weighted_expand", {"class_type": "RSTIPONativeExpand", "inputs": {
            "tags": "(blonde hair:1.2), blue eyes, pink cardigan", "length": "long",
            "max_new_tokens": 128, "seed": 0, "temperature": 0.5}}),
        ("overflow", completion(0, 128, "garden " * 4100)),
    ):
        record = poll(run(node))
        assert record["status"]["status_str"] == ("error" if name == "overflow" else "success")
        if name == "overflow":
            errors = [m[1] for m in record["status"]["messages"] if m[0] == "execution_error"]
            assert errors and "context overflow" in errors[0]["exception_message"]
        if name == "weighted_expand":
            assert record["outputs"]["2"]["text"][0].startswith(node["inputs"]["tags"])
        cases.append({"case": name, "record": record})
        print(f"{name}: {record['status']['status_str']}", flush=True)
    report["cases"] = cases
    report["before_free"] = get("/system_stats")["devices"]
    post("/free", {"unload_models": True, "free_memory": True})
    # ComfyUI processes the free flag on its queue worker's short idle timeout.
    time.sleep(2)
    report["after_free"] = get("/system_stats")["devices"]
    report["reload"] = poll(run(completion(95)))
    assert report["reload"]["status"]["status_str"] == "success"
    report["after_reload"] = get("/system_stats")["devices"]
    pid = run(completion(129, 1024))
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if get("/queue")["queue_running"]:
            break
        time.sleep(0.05)
    post("/interrupt", {})
    report["interrupt"] = poll(pid)
    assert any(m[0] == "execution_interrupted" for m in report["interrupt"]["status"]["messages"])
    report["after_interrupt"] = poll(run(completion(3)))
    assert report["after_interrupt"]["status"]["status_str"] == "success"
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"COMFY API/LIFECYCLE PASS: {args.output}", flush=True)


if __name__ == "__main__":
    main()
