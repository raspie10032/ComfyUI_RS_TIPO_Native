# RS TIPO Native 가이드 — 원본 TIPO와 Gemma TIPO

**원본 TIPO 2.1과 Gemma 기반 TIPO LoRA를 같은 ComfyUI 노드에서 선택해 사용하는 방법**입니다.
기준일: 2026-11-04. 노드 이름은 **RS TIPO Native (Experimental)**입니다.

**v0.2.2**에는 Gemma Q4의 PLE 표를 RAM에 보관하는 처리와 LoRA·역양자화
최적화가 포함됩니다. 기존 워크플로우와 모델 선택을 그대로 사용하며 추가
의존성은 없습니다. 측정한 10인 자연어 생성의 최대 PyTorch 할당 VRAM은
약 1.78 GiB입니다. 시스템 RAM에는 PLE 표가 남으며 BF16에는 같은 절감이
적용되지 않습니다. [측정 조건과 한계](gemma-inference-optimization.ko.md)

## 먼저, 원저작자와 출처

**TIPO 논문 저자분들과, 공식 TIPO 모델·Kohaku 아키텍처·KGen 구현을 공개한
KohakuBlueleaf(Shih-Ying Yeh) 및 각 프로젝트 기여자분들께 감사드립니다.**
이 확장은 그 공개 작업을 바탕으로 ComfyUI 네이티브 실행과 전역·캐릭터별
입출력, RS 파이프 연결 등을 추가한 독립 확장입니다.

- [TIPO 논문: Text to Image with Text Presampling for Prompt Optimization](https://arxiv.org/abs/2411.08127)
- [KBlueLeaf의 공식 TIPO 2.1 모델](https://huggingface.co/KBlueLeaf/TIPO-v2.1-1B-A200M)
- [KohakuBlueleaf / KohakUwULLM](https://github.com/KohakuBlueleaf/KohakUwULLM) · [KGen](https://github.com/KohakuBlueleaf/KGen)

Gemma 쪽의 기반 모델은 [Google DeepMind의 Gemma](https://huggingface.co/google/gemma-4-E2B-it)입니다.
현재 지원 베이스에는 [p-e-w의 heretic-ara 파생 모델](https://huggingface.co/p-e-w/gemma-4-E2B-it-heretic-ara)과
[mradermacher의 Q4 배포본](https://huggingface.co/mradermacher/gemma-4-E2B-it-heretic-ara-GGUF)을 사용하며,
[raspie의 TIPO LoRA](https://huggingface.co/raspie/gemma4-tipo-lora-pilot-002)를 적용합니다.
**Gemma TIPO는 이 확장에서 지원하는 별도 학습 경로**입니다. 공식 TIPO 체크포인트와 구분해 주세요.
각 코드·가중치의 라이선스와 사용한 부분은 [출처 상세](references.md)와 [NOTICE](../NOTICE)에 정리했습니다.

## 1. 어떤 일을 하는 노드인가요?

짧은 프롬프트를 읽고, 이미지 생성에 사용할 태그나 영어 설명을 보강합니다.
전역 프롬프트와 캐릭터별 프롬프트를 나눠 입력하고, 같은 번호의 STRING 출력을 받습니다.

| `model_family` | 실행하는 모델 | 입력 | 사용을 시작할 때 |
| --- | --- | --- | --- |
| `tipo_2_1` | 원저작자의 공식 `TIPO-v2.1-1B-A200M` 체크포인트 | 영어 | 짧은 영어 태그 + `tags` |
| `gemma_tipo` | 지원 Gemma 4 E2B 베이스 + 선택한 TIPO LoRA | 한국어·영어 | `pilot-002` + `natural_language` 또는 `tags` |

모델을 바꿔도 입력·출력 소켓은 같습니다. 기본 선택은 `gemma_tipo`이며,
원본 TIPO를 쓰려면 `model_family=tipo_2_1`로 바꿉니다.
선택한 모델 하나로 실행하며 Gemma 번역 뒤에 원본 TIPO를 연결하는 방식은 아닙니다.

생성 결과는 영어입니다. 다만 `expand_common`을 끄면 전역은 입력 원문을 그대로 반환하므로
한국어 전역도 그대로 남습니다. 전역까지 영어로 보강하려면 켜 주세요.

## 2. 설치와 첫 다운로드

저장소: [ComfyUI_RS_TIPO_Native](https://github.com/raspie10032/ComfyUI_RS_TIPO_Native)

**Comfy Registry:** [RS TIPO Native](https://registry.comfy.org/publishers/raspie10032/nodes/rs-tipo-native)
(`rs-tipo-native`, 게시자 `raspie10032`). Registry를 사용하는 ComfyUI Manager에서
이 이름으로 검색하거나 `comfy node install rs-tipo-native`로 설치할 수 있습니다.
Registry 배포의 지원 기준은 **ComfyUI 0.41.0 이상**이며, Gemma 경로에는 네이티브
Gemma 4 생성 API도 필요합니다. 코드 라이선스는 GPL-3.0-only이고 모델 가중치의
조건은 별도입니다. 특히 `scale-20pct-001` LoRA는 **CC BY-NC 4.0 비상업용**입니다.

수동으로 설치하려면 아래 명령을 사용합니다.

ComfyUI 폴더에서 다음을 실행합니다.

```bash
git clone https://github.com/raspie10032/ComfyUI_RS_TIPO_Native custom_nodes/ComfyUI_RS_TIPO_Native
python -m pip install -r custom_nodes/ComfyUI_RS_TIPO_Native/requirements.txt
```

`python`은 **ComfyUI를 실행하는 Python**으로 바꿔 주세요. 가상환경을 쓰면 그 환경에서 실행합니다.
Windows 포터블은 배포본 최상위 폴더에서 다음처럼 실행합니다.

```powershell
git clone https://github.com/raspie10032/ComfyUI_RS_TIPO_Native ComfyUI/custom_nodes/ComfyUI_RS_TIPO_Native
.\python_embeded\python.exe -m pip install -r .\ComfyUI\custom_nodes\ComfyUI_RS_TIPO_Native\requirements.txt
```

Windows 명령은 표준 포터블 폴더 구조 기준이며, Windows 실제 추론 검증은 아직 없습니다.
설치 후 ComfyUI 재시작 → 브라우저 새로고침 → **RS TIPO Native (Experimental)** 검색 순서로 진행합니다.
이전 **RS TIPO Prompt**와 내부 노드 ID `RSTIPOPrompt`가 같아 기존 연결을 이어 씁니다.

Gemma 경로에는 Gemma 4 네이티브 텍스트 생성 API가 있는 ComfyUI가 필요합니다.
해당 API가 없다는 오류가 나오면 ComfyUI를 업데이트해 주세요.
추론은 ComfyUI 모델 관리와 PyTorch로 수행하며, 별도 추론 서버 설치는 필요 없습니다.

**선택한 모델에 필요한 누락 파일은 첫 실행 때 자동 다운로드합니다.**
노드에 파일명·진행률·용량과 로딩 상태가 표시됩니다.

| 선택 | 첫 설치 다운로드량 | ComfyUI 기준 저장 위치 |
| --- | --- | --- |
| 원본 TIPO 2.1 | 약 1.85 GiB | `models/tipo/TIPO-v2.1-1B-A200M/` |
| Gemma TIPO 기본 Q4 경로 | 베이스·토크나이저·기본 LoRA 합계 약 3.27 GiB | `models/text_encoders/rs_gemma_q4/`, `models/loras/rs_gemma_tipo/` |

인식 가능한 로컬 Gemma 네이티브 BF16 베이스가 있으면 `language_model=auto`가 우선 사용합니다.
실제 다운로드량은 설치된 파일에 따라 다릅니다. **파일 크기는 필요한 VRAM 용량과 다릅니다.**
Gemma Q4 배포 파일도 이 노드에서는 네이티브 PyTorch 경로로 실행합니다.
다운로드 중단 후 다시 실행하면 Hugging Face 캐시의 부분 파일을 재사용합니다.
고정 리비전·SHA-256 검증, 출처·오프라인 설치는 [모델 다운로드 안내](native-models.md)를 보세요.

## 3. 기본 연결

```text
장면 STRING ──────→ common_prompt
캐릭터 1 STRING ──→ character1    RS TIPO Native → common_prompt STRING
캐릭터 2 STRING ──→ character2                   → character1 STRING
                                                → character2 STRING
기존 RS 파이프 ───→ building_pipe (선택)          → building_pipe (선택)
```

`common_prompt`에는 인원수·장소·공통 상황을, `character1`, `character2`에는
각자의 외형·의상·위치·담당 행동을 적습니다. 입력 타입은 STRING입니다.

`character_count`를 실제 사용할 인원수에 맞춥니다. 0~10명을 지원하며,
활성 번호를 벗어난 입력은 처리하지 않고 해당 STRING 출력은 빈 문자열이 됩니다.
전역만 보강할 때는 0명과 `expand_common=ON`으로 설정하고 전역 STRING 출력을 사용하세요.
RS 빌딩 파이프에는 최소 한 명의 캐릭터가 필요합니다.

**캐릭터 번호가 이미지의 좌우 위치를 정하지는 않습니다.** 프롬프트에 위치를 적고,
실제 영역 배치는 사용하는 이미지 생성 워크플로의 위치·마스크 설정으로 조절합니다.

## 4. 원본 TIPO 2.1로 시작하기

[원본 TIPO 영어 예제 JSON](../examples/tipo-2.1-english.json)을 ComfyUI로 불러오면 됩니다.
다음은 예제 입력이며, 생성 결과를 인용한 것이 아닙니다.

```text
common_prompt:
2girls, university cafe, afternoon, wooden table

character1:
(blonde hair:1.2), blue eyes, pink cardigan, on left, offering cup, extending arms

character2:
black hair, brown eyes, yellow sweater, on right, receiving cup, reaching out
```

| 항목 | 시작 설정 |
| --- | --- |
| `model_family` | `tipo_2_1` |
| `character_count` | `2` |
| `output_format` | `tags` |
| `expand_common` | `OFF` — 전역 원문 유지 |
| `protect_source` | `ON` — 알려진 외형·역할 충돌을 줄이는 보호 |

실행하면 공식 TIPO 체크포인트를 네이티브로 불러와 프롬프트를 보강합니다.
Gemma 전용 `language_model`과 `tipo_lora` 선택은 숨겨집니다.
한국어를 입력하면 다운로드·추론 전에 Gemma TIPO 선택 안내가 나옵니다.

`natural_language`도 선택할 수 있지만, 기존 확인에서는 외형을 임의로 추가하거나
행동을 빠뜨리는 사례가 있었습니다. 우선 태그 확장부터 결과를 확인해 주세요.
원문 보호는 제한된 영어 어휘에 대한 검사이므로 모든 의미 충돌을 잡지는 못합니다.

## 5. Gemma TIPO로 한국어 입력하기

[Gemma TIPO 한국어 예제 JSON](../examples/gemma-tipo-korean.json)을 불러오거나,
같은 노드에서 아래처럼 설정합니다. 다음 문장도 입력 예시입니다.

```text
common_prompt:
성인 여성 두 명이 오후 카페에서 컵을 주고받는다.

character1:
왼쪽의 금발 파란 눈 여성이 흰 셔츠를 입고 컵을 건넨다.

character2:
오른쪽의 갈색 머리 초록 눈 여성이 파란 드레스를 입고 컵을 받는다.
```

| 항목 | 시작 설정 |
| --- | --- |
| `model_family` | `gemma_tipo` |
| `tipo_lora` | `pilot-002` — 기본 다운로드 어댑터 |
| `character_count` | `2` |
| `output_format` | `natural_language` — 영어 설명 / `tags` — 영어 태그 |
| `expand_common` | `ON` — 한국어 전역도 영어로 보강 |
| `language_model` | 고급 설정의 `auto` |

Gemma가 전역과 활성 캐릭터 입력을 함께 읽고 번호별 영어 설명을 만듭니다.
`tags`에서는 **같은 LoRA가 적용된 같은 모델**로 그 설명을 태그화합니다.
생성된 설명에서 명시된 외형·방향을 우선 반영하고 근거 없는 추가 태그를 걸러내지만,
이 검사로 원래 입력의 모든 의미가 보존되지는 않습니다.

`pilot-001`, `image-30k`도 선택할 수 있습니다. 이름이나 데이터 규모만으로
어댑터 간 품질 우열이 확인된 것은 아닙니다. 처음에는 기본 `pilot-002`를 사용하세요.

11월 3일 학습 완료한 [공개 학습본](https://huggingface.co/raspie/gemma4-tipo-lora-scale-20pct-001)은
**`tipo_lora=scale-20pct-001`**로 선택합니다. 첫 실행 때 약 45.3 MiB를 자동으로 받고
크기·SHA-256을 검증합니다. 파일을 직접 설치하거나 별도 LoRA 로더를 연결할 필요는 없습니다.
이 학습본은 **CC BY-NC 4.0 비상업용 실험본**입니다. 기존 기본값은 유지됩니다.
[자동 다운로드 예제 JSON](../examples/gemma-tipo-scale-20pct.json)을 불러오면
이 학습본과 Gemma Q4 베이스가 선택되어 있습니다. `0.2.0` 이상이 필요합니다.

`local:<이름>`은 해당 컴퓨터에 설치한 별도 학습본입니다.
`local:scale-20pct-001`은 같은 가중치를 수동 설치했을 때의 선택명입니다.
[기존 로컬 학습본용 예제](../examples/gemma-tipo-trained-local.json)를 그대로 쓸 수도 있고,
그 예제의 `tipo_lora`를 `scale-20pct-001`로 바꾸면 공개 자동 다운로드 경로를 사용합니다.
노드를 업데이트하거나 새 로컬 어댑터를 추가했다면 서버를 재시작해 주세요.

## 6. 결과 확인과 이미지 생성 연결

실행 후 결과 패널에서 `All prompts` 또는 개별 번호를 선택합니다.

- **Output**: 실제 출력 프롬프트.
- **Source**: 입력 원문. 외형·의상·행동이 유지됐는지 대조합니다.
- **Model details / raw output**: 실행 모델, 적용 LoRA·rank·가중치 SHA-256과 원시 생성 기록.
- **Copy**: 현재 표시한 내용 복사.

결과 패널 자체는 워크플로에 저장되지 않습니다. 필요한 결과는 복사해 보관하세요.

**일반 연결:** 전역·캐릭터 STRING 출력을 사용 중인 텍스트 인코더나
프롬프트 입력 노드에 연결합니다. 모델의 태그·자연어 수용 방식에 맞춰 `output_format`을 고릅니다.
이 노드는 텍스트를 보강하므로 이미지 모델·샘플러 등 이미지 생성 부분은 별도로 연결합니다.

**RS 멀티캐릭터 연결:** `building_pipe` 출력을 RS 위치·마스크·LoRA 구성과
최종 파이프를 거쳐 샘플러에 연결합니다. 기존 파이프를 입력하면 위치·마스크·LoRA·참조 등
프롬프트 이외의 설정을 유지하면서 프롬프트를 교체합니다.
파이프 기능에는 [RS MultiCharacter Sampler](https://github.com/raspie10032/ComfyUI_RS_MultiCharacter_Sampler)가 필요합니다.
개별 STRING 출력은 그 확장 없이도 사용할 수 있습니다.

## 7. 알아두면 좋은 설정

| 설정 | 의미 |
| --- | --- |
| `expand_common` | 켜면 전역도 보강. 끄면 원문 그대로 반환 |
| `max_added_tags` | 기본 6. 필드별 추가 묘사 예산이며 반드시 채우는 개수가 아님 |
| `excluded_tags` | 제외할 태그·구절. 공백과 언더스코어를 동등하게 취급하는 문자열 기반 차단 |
| `rating`, `quality`, `quality_era` | 모델에 전달하는 보강 방향 메타데이터. 기본 `unspecified`로 시작 |
| `advanced` | 베이스, 토큰 상한, 시드, 온도 등 고급 설정 표시 |
| `generation_max_tokens` | Gemma는 전체 응답 한도. TIPO 2.1은 출력별 한도이며 입력+출력이 4096 토큰 이내여야 함 |

v0.2.5부터 Gemma는 태그 변환 전에 전역의 조명·분위기와 캐릭터의 시선·손동작·자세를
입력과 양립하는 범위에서 보강하도록 요청합니다. `max_added_tags`는 1~32 범위이며,
전역 확장에는 `expand_common=ON`도 필요합니다.
인물·의상·소품·장소를 임의로 새로 만들도록 요청하는 기능은 아닙니다.

결과 패널의 `common: +2 tags / 1: +1 tags`는 **최종 출력에서 원문 뒤에 붙은 문자열 태그 수**입니다.
원문 그대로면 `unchanged`, 전역 확장을 껐으면 `disabled`를 표시합니다.
같은 의미를 바꿔 쓴 태그까지 의미상 새 묘사로 검증한 숫자는 아닙니다.
한국어 입력은 번역 자체를 추가 태그로 세지 않고 `translated`로 표시합니다.
[수정 이유·실제 출력·검증 한계](gemma-expansion-fix.ko.md)를 참고하세요.

입력에 직접 적은 구절을 `excluded_tags`에도 지정하면 모순으로 오류를 알립니다.
제외 설정은 동의어·다른 언어 표현까지 모두 막는 의미 기반 필터가 아닙니다.

오래된 워크플로에 남은 `common_weight`, `character_weight`, `max_new_tokens`,
`input_language`, `translation_max_tokens`는 현재 공유 노드에서 무시됩니다.
출력 길이는 `generation_max_tokens`로 조절하세요.
`protect_source`는 원본 TIPO에서 사용하며 Gemma 모드에서는 숨겨집니다.

## 8. 문제가 생겼을 때

| 증상 | 확인할 내용 |
| --- | --- |
| 노드가 검색되지 않음 | ComfyUI의 Python에 의존성을 설치했는지, 시작 로그의 import 오류와 재시작·새로고침 여부 확인 |
| Gemma 네이티브 API 오류 | Gemma 4 텍스트 생성을 지원하는 ComfyUI 버전인지 확인 |
| TIPO 2.1에 한국어를 넣어 오류 | `model_family=gemma_tipo`로 변경하거나 영어로 입력 |
| 캐릭터 입력 누락 오류 | `character_count`와 활성 STRING 연결 수 확인 |
| 한국어 전역이 그대로 출력됨 | `expand_common=ON` 확인. 복구 경고가 있으면 해당 칸의 원문을 유지했을 수 있음 |
| `local:` 학습본이 없다고 나옴 | 일반 설치에서는 공개 기본 어댑터 `pilot-002`와 한국어 예제 사용 |
| 모델 크기·SHA-256 오류 | 오류가 지목한 파일을 별도 보관 후 재시도. 임의로 받은 다른 체크포인트로 대체하지 않기 |
| Gemma 복구 경고 | 모델 상세의 `warnings`, `calls`, `source_restored_fields` 확인. 원문 복구는 확장·번역 성공을 뜻하지 않음 |
| 출력 잘림·문맥 상한 | 입력을 줄이고 토큰 설정 확인. Gemma의 형식 오류는 1회 재시도 후 해당 원문/설명을 유지하며 경고. TIPO 2.1은 입력+출력 4096 한도 적용 |

0.2.11부터 Gemma가 완성된 JSON 뒤에 여분 괄호를 붙이거나 잘못된
이스케이프·마지막 쉼표를 출력한 경우, 내용과 슬롯 소유권이 명확하면 형식을 복구합니다.
빈 응답·불완전 JSON·슬롯 오류는 같은 모델로 1회 재시도하고, 끝내 실패한 부분은
그 칸의 원문을 유지합니다. 태그 변환만 실패하면 그 칸의 설명을 유지합니다.
패널에 복구 경고를 표시하며, 번역되지 않은 한국어를 번역 완료로 표시하지 않습니다.
CUDA 오류·메모리 부족·사용자 중단·설치 오류는 그대로 알려 줍니다.
자세한 범위는 [실행 경로 점검 기록](gemma-runtime-audit.ko.md)을 참고하세요.

원본 TIPO와 Gemma 양쪽의 실제 텍스트 추론 및 노드 전환·저장/불러오기 확인 기록은
[네이티브 모델 검증 범위](native-models.md#what-has-and-has-not-been-validated)에 있습니다.
한국어·학습본의 실제 출력과 누락 사례는 [학습본 노드 확인 기록](trained-node-validation.ko.md)을 참고하세요.

두 경로 모두 실험용입니다. 머리색 누락, 역할 오해, 불필요한 묘사 추가가 남아 있으므로
**출력과 원문을 대조한 후 이미지 생성에 사용해 주세요.**
노드 실행 성공은 이미지 품질 개선의 증거가 아니며, 이번 가이드 작성에서 새 GPU 추론이나 이미지 생성은 수행하지 않았습니다.
