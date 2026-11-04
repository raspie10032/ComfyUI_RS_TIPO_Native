# Korean input with Gemma TIPO

Korean and English enter the same Gemma base with the selected TIPO LoRA.
There is no separate Gemma translation stage or Kohaku/TIPO model handoff.

Connect common and character STRINGs to RS TIPO Prompt, set character_count,
and choose tags or natural_language. Enable expand_common if the common
prompt should also be rewritten into English. Disabled means exact passthrough.

All active slots are read together to interpret giving/receiving and other
shared actions. Output strings are routed back to the original slot numbers.
Tag mode renders the captions with that same adapted model. Known appearance
phrases from each caption seed its tag prefix; other tags remain experimental.

Use the result panel to compare the original, raw calls, and final outputs.
Literal exclusions normalize underscores and spaces. They do not recognize all
synonyms across languages. See gemma-tipo-validation.md for the actual test scope.

The earlier Korean→separate TIPO implementation has been superseded. Its
historical results do not evaluate Gemma TIPO.
