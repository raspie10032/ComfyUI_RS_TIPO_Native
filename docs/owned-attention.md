> Historical standalone TIPO implementation/test record. The current active node uses Gemma + TIPO LoRA. See [current usage](usage.ko.md) and [Gemma validation](gemma-tipo-validation.md).

# Input ownership and controlled native TIPO — 2026-11-01

The native execution baseline is now extended with actual input-span attention
control and immutable RS/NAI pipe editing. This is an experimental controller,
not a guarantee that TIPO produces consistent or useful new tags.

## What the controller changes

`Controlled Expand` reads a common scene and one character. `Pipe Expand`
processes each enabled character independently, using the original scene for
every character. Other characters' captions never enter that sequence or its KV
cache. No synthetic character-ID instructions are added to TIPO's learned format.

The tokenizer encodes the complete native-format prompt once. Character offsets
identify header, common and current-character token spans. At every decoder
layer, before attention softmax, the controller adds `log(weight)` to the source
token's attention score. Thus 1 is neutral, 0 blocks that key, and 0.5/1.5 changes
the relative score contribution by a factor of three. This does **not** guarantee
three times the resulting attention probability or semantic influence.

Ownership also limits reads during initial input processing: headers read
headers; common tokens read headers/common; character tokens read
headers/common/their own character. Generated history has neutral weight.
Both initial processing and cached decoding obey causality. The repetition
penalty sees only readable prompt tokens and the current target's generated
history. Hidden tokens cannot indirectly influence that score processor.

TIPO is a mixture-of-experts model. In FP16, grouping unrelated tokens into an
expert's matrix multiplication changed rounding on readable tokens, even after
attention isolation. Strict isolation therefore also separates expert-dispatch
batches by owner when other owners or zero-weight spans are present. Ordinary
single-owner positive-weight calls retain the original batching path.

Image-prompt weights such as `(blonde hair:1.2)` remain byte-for-byte in the
output prefix. These strings are not converted into TIPO attention multipliers;
the two node controls supply those multipliers separately. A common-only
expansion uses ordinary common attention, even if the per-character common
weight is zero. The control's tooltip states that zero blocks common input for
the **character** generation.

## Connecting the nodes

Use **RS TIPO Native Controlled Expand (Experimental)** to inspect one pair of
scene/character inputs. Outputs are the expanded original, added tags, and JSON
with raw generation, token counts, timing, weights and provenance.

Use **RS TIPO Native Pipe Expand (Experimental)** with:

```text
RS Character Prompts → [LoRA / Position / Camera in either order]
                     → TIPO Pipe Expand → Finalize → Sampler / Detailer
```

The Character Prompts node must precede TIPO, so active captions and their count
are known. The other editing roles may be before or after it. It also accepts
an already finalized character pipe or a NAI context and returns the same
dataclass type. Text, inactive and out-of-count rows are not expanded. Existing
captions, weights, negatives, coordinates, masks, LoRAs, references, deferred
camera/pose settings and provenance fields are preserved.

For a new building pipe with no embedded scene, connect the node's
`common_prompt` STRING output to the sampler's common prompt input. Its
`building_pipe` output carries the character changes. For a sourced final pipe
or NAI context, the scene is updated in the pipe as well.

**Expand common is off by default.** When enabled, scene expansion runs
separately and may invent appearance attributes that then affect all characters.
Character calls still read the original scene, never a different character's
completion or the newly expanded scene. This controls information access; it
does not suppress model priors or apply a semantic conflict classifier.

Begin with the experimental common 0.5 / character 1.5 defaults. Compare 1 / 1
for neutral source weighting, or 0 / 1 to remove the common input from character
generation. Larger weights did not consistently improve the tested cases.

## Actual checkpoint study

Environment: RTX 4070 Ti SUPER, PyTorch 2.14.0+cu130, FP16; pinned
TIPO-v2.1-1B-A200M revision `f5a318524a4ab30cdbbf51816cf406170f454e65`.
Temperature 0, seed 72157, up to 96 generated tokens per completion.

Three scenes (campus walking, a café, rainy streets), each with two distinct
hair/eye/clothing combinations, were used. There were three flattened-input
baselines plus 24 independent controlled outputs: six characters × four weight
settings. Original prompt prefixes were retained in all controlled outputs.

| Common / character | Outputs with a listed conflicting tag | Median generation time |
| --- | ---: | ---: |
| 0 / 1 | 0 / 6 | 1.82 s |
| 1 / 1 | 1 / 6 | 1.53 s |
| 0.5 / 1.5 | 0 / 6 | 1.49 s |
| 0.25 / 2 | 2 / 6 | 1.60 s |

The metric checks exact matches against a small, manually supplied list of
conflicting tags for these six inputs. It misses synonyms, garment layering,
unwanted composition changes and many semantic contradictions. These numbers
are **not** a prompt-quality score, a stable optimum or an image-quality result.
The strongest setting invented conflicting eye/hair tags in two cases. Zero
common input also lost useful scene context, despite its zero counted conflicts.

Mechanism checks using the real checkpoint:

- Zero-weight common tokens were changed to different valid vocabulary IDs:
  target final logits remained exactly equal (maximum absolute difference 0).
- A blocked second owner's tokens were changed: target final logits also
  remained exactly equal. Both sampled paired generations matched all 16
  generated token IDs, including repetition-penalty handling.
- Changing source weights changed final logits by up to 27.6875 in the selected
  probe. Unit weighting versus the ordinary path differed by up to 0.125 in
  FP16 because explicit float masks can select a different SDPA numerical path.
  Bitwise neutrality is not claimed on the actual checkpoint.
- Peak PyTorch allocated memory during the study was 1,994,836,992 bytes
  (about 1.86 GiB). This excludes the CUDA driver and other processes' VRAM.

## Runtime and pipe evidence

The idle local `comfyui.service` was restarted with the updated project symlink.
All four native node classes registered. Real `/prompt` execution on port 8188
passed these four cases: standalone controlled expansion; Position before pipe
expansion; Position after pipe expansion; and finalized character-pipe input.
The cases included two active characters, an unchanged literal text slot, and a
disabled fourth character. Finalization succeeded in both role orders.
The finalized outputs were equal with Position before and after TIPO.

The updated source passed 20 CPU tests. The official baseline parity comparison
was rerun after the decoder changes: all three inputs retained identical prefill
logits and matched all 96 greedy output tokens, with the previously documented
FP16 projection differences of up to 0.125.

Additional checks imported the actual installed MC and NAI class definitions
without loading their full extensions. With a deterministic text callback,
they verified mask/LoRA/text/scene preservation through actual BuildPipe
finalization and NAIContext → local-pipe conversion. That callback test proves
data preservation, not TIPO quality or a paid NovelAI request. Native pipe API
execution used the actual TIPO checkpoint; no image generation or NAI API call
was submitted.

## Reproduction and limitations

```bash
python -m unittest discover -s tests -v
python tools/validate_control.py /path/to/TIPO-v2.1-1B-A200M --output validation/owned-control.json
python tools/validate_control.py /path/to/TIPO-v2.1-1B-A200M --mechanism-only --output validation/owned-mechanism.json
python tools/validate_pipe_api.py --url http://127.0.0.1:8188 --output validation/owned-comfy-api.json
python tools/validate_pipe_protocols.py /path/to/ComfyUI --output validation/owned-pipe-protocols.json
```

Create `validation/` first when using a fresh clone. Use the ComfyUI environment
for the commands. The API smoke requires MC and PreviewAny nodes, submits small
TIPO jobs and requires an idle queue. Raw local outputs are ignored by Git.

This iteration does not establish image-generation improvement, ideal weights,
all languages, Windows, non-CUDA hardware or public-package readiness. The
baseline Completion/Expand nodes remain available for comparisons. Full semantic
ownership is still limited by the pretrained model's own learned associations.
