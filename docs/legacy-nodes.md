# One prompt node and legacy workflow compatibility

New workflows use **RS TIPO Prompt (Experimental)**. Model loading, automatic
model download, generation-time excluded tags, role-aware expansion, result
inspection, separate STRING outputs and the RS building-pipe output are already
available in that node. No extra TIPO loader or expansion node is needed.

## Common-only and character use

| Task | Settings / connections |
| --- | --- |
| Expand one common STRING | `character_count=0`, `expand_common=ON`; use `common_prompt` output |
| Pass the common STRING through | `expand_common=OFF` |
| Expand one character in its scene | `character_count=1`; connect common and character1 STRINGs |
| Expand multiple characters | `character_count=1..10`; connect each numbered STRING |
| Use the RS pipeline | Use the same node's `building_pipe` output; retain incoming LoRA/mask/position settings through its optional pipe input |

The STRING inputs are authoritative. The optional incoming pipe does **not**
automatically import its prompt table. Literal text slots belong downstream of
this node. Common-only mode has no regional building-pipe output.

## Why four old nodes are still registered

The early research nodes have ComfyUI's `DEPRECATED=True` flag and the category
`RS/TIPO Native/Legacy`. With the default deprecated-node filter, they no longer
compete with the main node in search or the add-node menu. ComfyUI's explicit
option to show deprecated nodes can make them visible again. Older frontends
that ignore that flag may still list the Legacy category.

Saved workflows and API prompts keep their existing class IDs, function names,
input schemas, output indexes and inference behavior. We do not silently replace
old nodes with the main node: their expansion policies differ.

| Existing class ID | Retained behavior | New workflow equivalent / difference |
| --- | --- | --- |
| `RSTIPONativeCompletion` | Raw official TIPO completion and diagnostic info | Low-level research only; not a normal prompt interface |
| `RSTIPONativeExpand` | Tag-only completion, without the main node's role/source protection | Common-only mode provides the supported user interface, with different filtering |
| `RSTIPONativeControlledExpand` | Common scene and one character with source-attention weights | Use character_count=1; the main node also derives interaction context and protects roles |
| `RSTIPONativePipeExpand` | Reads prompts directly from a draft/final pipe or NAI context, preserving inactive/text slots and other fields | Existing graphs keep working. To migrate, supply separate common/character STRINGs to the main node; it replaces the active prompt table |

For manual migration, keep a copy of the old graph, connect the original source
STRINGs to **RS TIPO Prompt**, then reconnect its named outputs. Review the result
panel before image generation; identical seeds do not imply identical expanded
text across these different policies. The legacy pipe editor has no automatic
one-click replacement because its inheritance and output types differ.

## 한국어

새 구성은 **RS TIPO Prompt 하나**로 전역·캐릭터 보강과 빌딩 파이프 출력을
처리합니다. 나머지 4개는 초기 연구용 노드이며 기본 추가 목록에서 숨깁니다.
기존 워크플로를 열면 연결과 동작을 그대로 유지하고, 폐기 노드 표시를 켜면
Legacy 분류에서 찾을 수 있습니다. 해당 표시 기능이 없는 구형 프런트엔드는
Legacy 분류가 보일 수 있습니다.

파이프 안의 문장을 직접 읽던 옛 Pipe Expand와 달리 통합 노드는 연결된
STRING 입력이 기준입니다. 옛 노드를 자동 교체하지 않으며, 수동 이전 시에는
원문 STRING을 연결하고 결과를 확인하세요.

## Validation (2026-11-02)

- Python: 77 tests passed; JavaScript: 3 test files passed.
- An isolated CPU ComfyUI server reported exactly one non-deprecated TIPO node
  through `/object_info`. Its frontend setting `Comfy.Node.ShowDeprecated`
  defaulted to false.
- In the actual frontend graph, all five class IDs could be instantiated,
  serialized and restored without missing-node errors. No inference-policy
  changes were made by this cleanup; no new image-generation comparison was run.
- The normal installation resolves to this source checkout. Its running process
  retains the previous registration until restarted; active generation was not
  interrupted.
