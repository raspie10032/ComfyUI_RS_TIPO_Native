# STRING interface and shared interaction — 2026-11-01

Initial implementation was developed on `dev/native-tipo`; the public version is at
https://github.com/raspie10032/ComfyUI_RS_TIPO_Native.

## Inputs / outputs

`RSTIPOPrompt` has STRING inputs `common_prompt`, `character1`–`character10`.
There is no separate interaction input. The count selects active character slots.
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
- character 1: `(blonde hair:1.2), blue eyes, pink cardigan, on left, offering cup,
  extending arms`
- character 2: `black hair, brown eyes, yellow sweater, on right, receiving cup,
  reaching out`

Each expansion reads the original common scene, its own complete character
source and a caption derived automatically from recognized action/position
clauses across active characters (`scene_context.py`). Other characters' private
appearance is excluded; the target's own role is repeated at the end of the
caption. Input-slot numbers do not imply screen positions. The caption uses the
learned `long_to_tag` format and is shown in the results panel. Unknown clauses
remain private instead of being guessed. This is an English lexical projection,
not a semantic planner, and output-role correctness is not guaranteed.

The inferred caption is used only during TIPO inference; it is not appended to
the image model's common STRING or pipe scene. The former low-level
`interaction` implementation remains an internal caption transport, not a user
input. Historical validation records below describe the earlier explicit-input
prototype; current checks are recorded in [automatic context](automatic-context.md).

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

Example: `examples/shared-interaction.json` uses core Primitive STRING nodes and
the TIPO results panel, and is copied into the local workflow list as `RS TIPO - Shared Interaction`. The
JSON has been structurally checked and its exact STRING flow executed successfully
on the resident server (one additional real-model request, 7 total). Browser
reload round-trip was unverified in this initial experiment. It is now covered by
the [legacy/current browser follow-up](compatibility.md). Raw initial record:
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
