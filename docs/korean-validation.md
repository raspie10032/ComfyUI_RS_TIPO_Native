> Historical standalone TIPO experiment. Superseded by Gemma+TIPO LoRA; see gemma-tipo-validation.md.

# Korean input: native Gemma → TIPO text validation

Validated 2026-11-02 on the Korean input implementation following `bab6342`.
This record evaluates **text inputs, intermediate translations and final prompt
outputs**. No diffusion image generation or NovelAI request was used.

## Implementation and environment

`RS TIPO Prompt` reads the common input and all active character inputs together
using host-native **base Gemma 4**, then routes English clauses into the existing
TIPO expansion. It does not use the existing Gemma TIPO adapter or retrain either
model. Clause IDs, unchanged English spans and weight syntax are program-owned.
See [setup and exact behavior](korean-input.md).

- Isolated ComfyUI **0.41.0**, frontend **1.55.17**, port 8194.
- RTX **5060 Ti 16 GB**, Python 3.12, PyTorch **2.14.0+cu130**.
- Local `gemma4-e2b-native-bf16.safetensors`, native `CLIP` text generation,
  greedy decoding, seed 42, translation limit 4096 for the main audit.
- Pinned TIPO v2.1 1B A200M; `rating=safe`, `quality=masterpiece`,
  `quality_era=newest`, TIPO limit 96 tokens per output, temperature 0.5,
  common/character weights 0.5/1.5, source protection enabled.
- The separate default-settings check omits all new language options and succeeds
  with `auto` and a 2048-token conversion budget. This does not prove that every
  ten-character or very verbose input fits that default budget.
- The production 4070 process was left running. The installation is a symlink to
  this checkout; it needs a restart to import the changed Python node code.

## Main audit

Four scenes each have English tags, Korean tags and Korean sentences: cup
exchange, partner dance, hair braiding and garden reading (12 requests).
Five further requests cover mixed-language weights, three-person tea preparation,
common-only Korean scenery, explicit English mode and a conflicting exclusion.

**17 requests: 16 successful executions and one expected source/exclusion error.**
There were **11 successful Korean conversions** and **44 TIPO output targets**
across the 16 successes. These are execution counts, not a semantic accuracy
percentage. All final-run conversions completed without a corrective second
generation. The main audit checks successful target outputs retain their
converted source prefix. It also compares the English `auto`/`english` pair's
generated token IDs, text, accepted tags and rejection lists for equality.

| Case | Intermediate translation inspected | Observation |
| --- | --- | --- |
| Cup, tags and sentences | `offering cup` vs `receiving cup`, blonde/blue eyes vs brown/green eyes, left vs right | Participant actions and supplied appearances remain in their slots. |
| Dance, tags and sentences | `leading dance`, hand on other's waist vs `following dance`, hand on other's shoulder | Roles and contact targets remain distinct. |
| Braiding, tags and sentences | `having hair braided by another person`, hands on own lap vs `braiding ... hair`, holding another person's hair | Actor and recipient remain distinct; their hands are not reassigned by translation. |
| Garden reading, tags and sentences | brown hair, green eyes, yellow sweater, open book with both hands, wooden bench, looking down | Specified appearance and activity remain represented across common/character inputs. |
| Mixed language and weights | `(금발:1.2), blue_eyes` → `(blonde hair:1.2), blue_eyes`; `[파란 원피스]` → `[blue dress]` | English bytes and weight/group syntax remain unchanged by code. |
| Three-person tea | holding empty teacup with both hands / pouring tea into first woman's teacup / placing cookies on plate on table | The three different tasks remain assigned to the three original slots. |
| Common-only scenery | rainy evening, empty cafe, wooden table, street lamp outside window | Works with `character_count=0`. |
| Exclusion conflict | positive `blue_eyes`, excluded `파란 눈` → `blue eyes` | Stops with a human-readable conflict error before loading TIPO; does not delete the source. |

The earlier pilots exposed missing clause IDs, unnecessarily long translations,
omitted right-side cues and omitted adult/count terms. Prompt guidance and
structural validation were improved. Explicit recognized source terms are now
retained and recorded when omitted, instead of being silently lost. For example,
the final three-person raw common translation used the noncanonical tag
`3adultgirls`; the recorded converted input is
`3girls, adult, 3adultgirls, garden, preparing tea` after source-term restoration.
This retains the explicit count/adulthood but also demonstrates that raw model
tag wording is still imperfect. It must not be described as flawless translation.

## What this does not establish

The translated scenes above are substantially more faithful than feeding Korean
directly to TIPO. That does **not** establish complete semantic checking or general
relationship understanding. Source facts may still be paraphrased or omitted in
unseen examples. The English source retention check verifies routing/preservation,
not whether Gemma translated every meaning correctly.

The existing TIPO addition selector is a separate problem. In this audit it still
accepted unrelated common-scene additions, e.g. `sidelocks`, dance `necktie`/`suit`,
and tea `elf`/`white capelet`. The new conversion does not fix these pre-existing
global-expansion errors. Inspect **raw Gemma output, converted source, TIPO
proposals, accepted additions and final output** separately. For conservative
use, leave `expand_common=OFF` and inspect additions before downstream use.

## Compatibility and frontend

- Python: **94 unit tests pass**. Generation-independent tests cover joint input,
  original English/weights, clause ordering, retry/error behavior, explicit source
  facts, source/exclusion conflicts and failure before TIPO loading.
- JavaScript: existing interface, socket and download/status test files pass.
- Current real frontend: Korean STRING connections, save/reload, `structuredClone`,
  advanced model hiding, distinct socket positions, actual queue execution and
  source/conversion/result display pass; no captured page errors.
- Legacy ComfyUI **0.3.26**, frontend **1.11.8**: English TIPO runs from the actual
  frontend after save/reload. Korean input explicitly errors because that host
  has no native Gemma 4 generation. This is not Korean support on old ComfyUI.
  Its server logs print the environment's installed frontend package version;
  the runtime's selected web root is the pinned 1.11.8 frontend.
- No Windows execution or older PyTorch validation was performed for this change.

## Reproduction

Use an isolated current ComfyUI with only this extension enabled, the native
Gemma checkpoint in `text_encoders`, and the official TIPO weights available.
Keep the production server and any active queues separate.

```sh
python tools/validate_korean.py --port 8194 \
  --cases tests/fixtures/korean_cases.json \
  --output validation/korean-conversion/comfy.json
python -m unittest discover -s tests -v
node tests/test_interface.mjs
node tests/test_sockets.mjs
node tests/test_download_status.mjs
# Optional actual frontend audit; install Playwright and its Chromium separately.
node tools/check_language_ui.cjs 8194
```

The audit saves a prompt handle before waiting. If observation times out, inspect
that same handle; do not submit the same test again without checking it.

Local, ignored evidence: `validation/korean-conversion/final4-comfy.json`,
`defaults.json`, `validation/korean-ui.json`, `korean-ui.png`,
`korean-sockets.log`, `korean-unit-tests.log`, `korean-legacy.log`,
`korean-legacy-error.json`, and `metadata-ui-8195.json`/`.png`.
The synthetic source cases and audit tools are committed; model weights, runtime
copies, screenshots and raw local histories are excluded from Git.
