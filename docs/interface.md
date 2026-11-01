# STRING interface and shared interaction — 2026-11-01

Implemented locally on `dev/native-tipo`; no GitHub publication.

## Inputs / outputs

`RSTIPOPrompt` has STRING inputs `common_prompt`, `character1`–`character10`,
and optional `interaction_prompt`. The count selects active character slots.
Output index 0 is building pipe, 1 common STRING, 2–11 character STRINGs. A
connected inactive socket keeps its index and connection, but returns an empty
STRING and contributes no character to the pipe. Missing active inputs raise a
readable error before loading the model. A results panel uses execution UI data
and does not put DOM elements into workflow serialization.

Optional incoming building/final/NAI pipes are normalized with the installed
MultiCharacter adapter. Prompt values come from STRING inputs; geometry, LoRA,
UC, references, deferred cameras and interaction settings are retained. Literal
text slots must be created downstream: all selected STRING slots are characters.

## Shared scene and participant roles

Example:

- common: `2girls, university cafe, afternoon, wooden table`
- interaction: `Two women sit across a cafe table. The woman on the left offers
  a cup with both hands. The woman on the right reaches out to accept that same
  cup. They look at each other.`
- character 1: `(blonde hair:1.2), blue eyes, pink cardigan, on left, offering cup,
  extending arms`
- character 2: `black hair, brown eyes, yellow sweater, on right, receiving cup,
  reaching out`

Each expansion reads the same original scene and shared caption plus that
character's own source. The caption uses the existing learned
`target: <|long|> <|long_to_tag|> / long: / tag:` format, with source owner 0 and
attention multiplier 1. This follows the installed KGen `apply_tipo_prompt`
long-to-tag ordering rather than inventing chat instructions or role delimiters.
It also remains in the downstream common STRING and pipe. Original participant
roles and prompt weights are preserved verbatim. The model is not trained as a
multi-agent role planner; independent caches do not prove coordinated semantics.

## Verification

- 26 Python checks: native math/control, original source preservation, input/count
  validation, no model allocation on malformed pipe requests, output indices,
  learned caption format and shared source ownership.
- JavaScript socket fixture: count 10 → 2 → 10 → 0 → 3, retained connected
  inactive slots, stable output indices, serialization. Module syntax checked.
- Real installed RS/NAI class fixtures: no image/API inference; STRING-built pipe
  finalizes to the same common/character strings; incoming mask/LoRA/UC/camera/
  interaction metadata retained; shrinking count removes stale active slots.
- Resident ComfyUI actual model inference: offer/receive cup, hug, dance;
  same inputs with shared caption OFF/ON = 6 requests. All graphs executed and
  finalized successfully, source weights remained, character 3 stale input was
  excluded, and unused character STRING returned empty. These are execution
  checks, not image quality scores. No image or NovelAI generation was run.
- Browser: node searchable and creates in the existing ComfyUI frontend; input
  sockets and DOM results panel render. Count interaction/reload restoration is
  covered only by the socket fixture so far, not a full browser round trip.

Raw resident execution records are in ignored `validation/interface-live.json`.
Class fixture report: `validation/interface-pipe-protocols.json`.

Example: `examples/shared-interaction.json` uses core STRING/Preview nodes and
is copied into the local workflow list as `RS TIPO - Shared Interaction`. The
JSON has been structurally checked and its exact STRING flow executed successfully
on the resident server (one additional real-model request, 7 total). Browser
reload round-trip remains unverified. Raw record:
`validation/shared-interaction-example-live.json`.

## Initial semantic findings

The shared caption changes generated tags but does not solve role consistency.
In offer/receive, shared caption added `looking at another` and `sitting` to both
slots. However, the blonde source still acquired `brown hair` in one output.
In dance, unwanted `hand on own hip` appeared; other additions included
unrequested hair lengths, clothing and art styles. Hug outputs also contained
unrequested attributes. These are model-generated contradictions, not evidence
that another character's private source was read.

Thus neither the interface nor successful requests prove accurate joint
interaction expansion. The subsequent [source protection experiment](source-protection.md)
adds conservative filtering, weight-syntax normalization for model inputs and
observable rejection reasons. The unchanged model is still not a reliable role
planner. Do not treat intact source strings or filtering as proof that all
appended tags are semantically correct.
