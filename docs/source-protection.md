# Interaction expansion adjustment — 2026-11-01

## Behavior

Each participant still reads the original shared scene and caption, together
with its own prompt in an independent native model sequence. No participant's
private prompt is added to another sequence. Write each person's role in that
person's STRING: for example, `offering cup` / `receiving cup`, or
`leading dance, hand on other waist` / `following dance, hand on other shoulder`.
The shared caption supplies their joint scene; it is not an automatic role
assignment language.

Explicit numeric weight wrappers are removed only from model input text.
The output retains the user's original spelling and weights, then appends new
tags. Literal escaped parentheses, character names and artist prefixes remain.
A token limit may cut through a tag; the unfinished last tag is discarded.

`protect_source` defaults to ON. It checks only **new additions** and removes
recognized identity, wardrobe, art-style and composition changes. Shared-scene
additions also exclude participant actions and appearance. Lexical checks
reject known action conflicts, unrequested orientation, self-directed gestures
in explicit partner actions, contradictory expressions, and a changed partner
contact target. The results panel exposes Original, Added, Excluded (with
reasons), and Expanded. OFF is available for a raw-output comparison; unfinished
tags are still removed. Non-prompt pipe settings are untouched.

This is conservative lexical protection, **not trained semantic role
understanding**. It can discard useful detail and miss unrecognized conflicts,
synonyms, negation or unknown incompatibilities between otherwise valid additions.
It does not invent complementary roles from an ambiguous scene or correct a
contradiction already in the source. Review the additions before image sampling.

Follow-up protection compares recognized conflicts against **accepted additions**
as well as supplied source tags. Standalone shared posture, setting and gaze tags
also constrain every character; prose describing different roles is not applied
as a shared posture. Unrequested comic panels/bubbles are protected composition
changes. Basic open-mouth emoticons are checked against closed-mouth tags.
See the [follow-up experiment and older ComfyUI tests](compatibility.md).

## Real model comparison

`tools/validate_grounding.py` runs on the installed idle ComfyUI at 8188. Model:
native TIPO v2.1 FP16. Three scenes: cup transfer, hug, partner dance. Each has a
shared caption and two distinct original character prompts. Common expansion
is enabled, temperature 0, seed 8910, output cap 96 tokens per slot,
common/character attention 0.5/1.5. Every case is repeated with protection
OFF/ON; model input and sampling settings are otherwise unchanged. Raw proposed
tags match between each pair. This measures the filtering stage, not an
improvement in the model's understanding or generated image quality.

| Protection | Requests / expanded slots | Appended tags | Rejected tags | Server execution total |
|---|---:|---:|---:|---:|
| OFF | 3 / 9 | 256 | 0 | 15.29 s |
| ON | 3 / 9 | 82 | 164 | 14.79 s |

The difference in total counts includes normalized redundant tags; not every
removed redundant tag is recorded as a rejection. Timing differences here are
small and are not evidence of a speed improvement.

Observed examples:

- Transfer: keep original `(blonde hair:1.2)` and `offering cup`, append
  `holding cup`, `sitting`, `looking at another`; reject newly proposed
  `brown hair`. The receiver's unrelated `drinking` is rejected.
- Dance: preserve `leading` / `following` and contact targets. Keep `dancing`,
  `looking at another`; reject `3d`, `hand on own hip` and a generated hand on
  the back when the supplied target is shoulder/waist.
- Hug: preserve both original roles and identity; reject unrequested
  `hug from behind` and added wardrobe/appearance changes. Remaining tags are
  not a guarantee that image generation will depict the intended embrace.

Raw records: ignored `validation/grounding-live.json`, summary
`validation/grounding-summary.json`, and log `validation/grounding-live.log`.
35 Python checks pass, including source weights, action/contact conflicts,
scene-only protection, opt-out, incomplete tags and immutable pipe behavior.
The JavaScript socket fixture and module syntax checks also pass.

Resident UI verification: run the existing `RS TIPO - Shared Interaction`
workflow through the actual Run button; result panel shows the original shared
caption, source protection ON, per-character additions and exclusions with
reasons. Its two-character model time was 2.9092 s. Full browser save/reload and
count-change round-trip were unverified at this initial stage; the subsequent
[compatibility study](compatibility.md) covers them on three frontends.
No image generation or NovelAI API request was run.

## Rejected inference variant

Local exploratory runs used paired-context next-token guidance. Increasing its
strength produced more strange tags and gestures rather than dependable role
coordination. It is not enabled in the node. Probe inputs/results are retained
locally in ignored `validation/guidance-probe*`, `role-probe*`, and
`short-probe*` artifacts. Native owned-attention control is unchanged.
