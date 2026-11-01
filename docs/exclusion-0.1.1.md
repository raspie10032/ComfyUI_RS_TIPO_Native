# 0.1.1: generation-time excluded tags and model acquisition

## User controls

All five TIPO nodes accept optional `excluded_tags`. The main Prompt node exposes
a multiline field. Separate tags with commas/newlines, for example
`3d, white_shirt, long sleeves`. Normalization applies NFKC, case folding,
underscore-to-space conversion and collapsed whitespace; duplicates collapse.
Existing supported weight wrappers are removed for comparison only. Source
strings keep their original weights and spelling.

The list applies to all active generated outputs. A source conflict is reported
before model acquisition, including the offending scene/character label. Inactive
character inputs are ignored. Empty exclusions use the original sampling path.
The existing `protect_source` control is independent and remains enabled by default.

## What is controlled

`generation.py` checks each sampled next-token candidate before appending it or
running the next forward pass. `exclusion.py` decodes generated history with that
candidate, separates tag fields and checks normalized tag equality. Candidates
that reach an excluded tag are masked out of the current scores and sampling is
repeated, including top-k/top-p/min-p. Greedy and top-k=1 decoding can select an
allowed alternative. An exhausted candidate set raises an explicit error.

This uses decoded text, so it is not limited to one encoding of a tag. A rejected
candidate never reaches output IDs, repetition history or the KV cache. The
blocklist is not inserted into positive model context. Each request owns its
constraint and RNG. This is inference-time token control, not a learned negative
embedding or model fine-tuning. Normal completion formatting and optional source
protection still run afterward; the paired exclusion test disables source
protection to demonstrate that blocking does not depend on a removal pass.

Limits: synonym meanings, natural-language descriptions and arbitrary role
conflicts are not understood. Checking an unfinished tag means that a short
excluded prefix can constrain a longer tag as well. A model may choose awkward
alternatives such as `long sleeveless` instead of `long sleeves`. This is a
lexical exclusion tool, not proof of improved image quality or interaction accuracy.
The result panel separately lists generation candidate rejections and source
protection's post-generation rejections. Counts are rejected sampling attempts,
not a count of removed final tags or probability of semantic compliance.

## Actual-model comparisons

GPU: RTX 4070 Ti SUPER, native FP16 official TIPO-v2.1-1B-A200M at revision
`f5a318524a4ab30cdbbf51816cf406170f454e65`. Each scene has two characters; source
attention weights 0.5/1.5, temperature 0.5, seeds 913/914, up to 96 tokens.
`protect_source=False` isolates the new control. First generate a baseline, then
exclude eight non-source tags from that baseline. Re-run with the same seeds,
then repeat with underscores substituted for spaces. This deliberately tests
known baseline continuations; it is not an unbiased general quality benchmark.

| Scene | Candidate attempts blocked, two character calls | Excluded normalized tags in completed added-tag list | Space/underscore generated IDs |
| --- | ---: | ---: | --- |
| Cup exchange | 29 | 0 | Identical |
| Dance | 29 | 0 | Identical |
| Hug | 19 | 0 | Identical |

There were six baseline calls and twelve constrained calls. New raw continuations
were inspected: unwanted exact tags were blocked, but alternative expressions
and unrelated additions still occurred. Changing exclusions can also alter the
rest of the sampled sequence. No image-generation improvement is claimed.

Reproduce from the checkout:

```sh
python tools/validate_exclusion.py /path/to/ComfyUI/models/tipo/TIPO-v2.1-1B-A200M --output validation/exclusion-0.1.1.json
python tools/validate_exclusion_api.py --port 8188 --output validation/exclusion-api-current.json
```

## Real ComfyUI and frontend checks

With source protection ON, expanded the common scene and two character inputs,
48 tokens each, for both space and underscore exclusion lists. Both requests
succeeded on each of these Linux backends/frontends:

| ComfyUI | Frontend | Request pairs | Space/underscore generated IDs |
| --- | --- | --- | --- |
| 0.3.26 | 1.11.8 | Pass | Identical |
| 0.3.64 | 1.27.10 | Pass | Identical |
| 0.41.0 | 1.55.17 | Pass | Identical |

The same installed modern Python/PyTorch environment served all backends. This
does not prove old PyTorch or Windows support. Isolated browser checks loaded
the example, changed counts 2/10/1/0/3, checked socket separation, connected a
seed input, and serialized/structured-cloned/reloaded the workflow. The new
exclusion value survived reload and appeared in the API prompt. Current and
0.3.26 pages had no uncaught errors. The old frontend 1.27.10 still reports its
previously isolated core navigation-settings error, while these checks pass;
see [compatibility details](compatibility.md).

Automated checks: 62 Python tests and both JavaScript fixture files pass. They
include alternate token encodings, substring distinction, greedy/top-k retries,
empty/no-hit regression, request isolation, source errors before model load,
download reuse/resume/hash/cancellation and extra model paths.

## Automatic model download

First node execution searches the normal and configured extra `tipo` paths. If
no complete official model is found, it downloads the three pinned data files
to the existing partial folder or the normal model folder. Total is
1,983,963,136 bytes (about 1.85 GiB). There is no download during extension import.
ComfyUI receives byte progress and checks interruption during transfer and hash
verification. Retrying uses Hugging Face's download cache; a marker prevents a
failed download being mistaken for a complete install. One-shot ComfyUI
cancellation exceptions are preserved as cancellation, not recast as network errors.
Transfer errors give a concise retry message without exposing the remote URL.

New/incomplete downloads are checked against pinned SHA-256 values. Existing
complete model folders are reused offline based on file presence and sizes;
this fast path does not re-hash an old install on every generation.

The network smoke test downloaded the official config and tokenizer into a
temporary folder, hard-linked the existing 1.85-GiB weights without copying or
redownloading them, and verified all three hashes and progress events. A full
fresh 1.85-GiB network transfer was not repeated. Interrupted/corrupt/network
paths use controlled test fixtures; they are distinct from that live download.
Temporary smoke files were removed. CLI `tools/download_model.py` uses the same
implementation for optional manual prefetch.

Local raw evidence is ignored under `validation/exclusion-*.json`,
`validation/download-0.1.1.json` and `validation/compatibility/socket-layout-exclusion-*`.
The source and reproducible procedures are kept in Git; model weights are not.
