# Gemma TIPO 0.2.12 기본값 — 지시문 + 약한 LoRA

2026-11-05 학습 완료한 새 Gemma 어댑터의 공개 기본 설정입니다.
TIPO의 아이디어와 원본 모델·구현을 공개한 TIPO 논문 저자분들,
KohakuBlueleaf 및 KGen 기여자분들께 감사드립니다. Gemma 기반 모델은
Google DeepMind의 작업이며, 이 문서의 학습본과 번호별 연결은 RS 확장입니다.

| 항목 | 설정 |
| --- | --- |
| `model_family` | `gemma_tipo` |
| `tipo_lora` | `presampling-fresh-r8-v1` — 자동 다운로드 |
| `tipo_lora_strength` | `0.25` |
| `temperature` | `0` — greedy |
| 지시문 | 한국어 번역, 조건 보존, 필드별 확장, 번호가 있는 JSON 출력 |
| 출력 | `tags` 또는 `natural_language` 직접 생성 |

강도는 **Gemma 텍스트 LoRA**에 적용됩니다. 이미지 생성용 마이마이 LoRA와는
독립적입니다. `0`으로 설정하면 어댑터를 다운로드하거나 패치하지 않고 같은
확장 지시문을 베이스 Gemma에 전달합니다. 결과 상세에는 요청 강도와 실제
패치 강도, 어댑터 SHA-256, 실행 경로를 기록합니다.

[공개 학습본](https://huggingface.co/raspie/gemma4-tipo-lora-presampling-fresh-r8-v1)의
두 파일 약 45.3 MiB를 첫 실행 때 자동 다운로드합니다. 고정 revision·크기·
SHA-256·학습 구조 검사를 거쳐 사용하며 저장 위치는
`models/loras/rs_gemma_tipo/presampling-fresh-r8-v1/`입니다.
새 설치도 같은 기본값을 사용합니다. 이전 `local:presampling-fresh-r8-v1`
선택은 설치된 파일과 같은 지시문을 계속 사용합니다.

RS 어댑터 기여분은 **CC BY-NC 4.0**입니다. 베이스와 원자료 조건은 계속
각 자료에 적용됩니다. 학습 프롬프트 출처:
[KohakuBlueleaf / titpop](https://huggingface.co/datasets/KBlueLeaf/titpop-dataset)
(원본 카드 CC BY-NC-ND 4.0). 한국어 사전:
[mengod-hf / 이오리네 및 기여자](https://huggingface.co/datasets/mengod-hf/danbooru-tags-kr)
(한국어 기여 CC BY-SA 4.0 및 별도 원자료 고지). 데이터셋과 가공 학습 레코드는
배포하지 않습니다. 모델 카드·NOTICE·원자료 LICENSE에 자세한 고지를 보존했습니다.

기존 워크플로우에 저장한 어댑터 선택과 값은 유지합니다.
[새 기본값을 넣은 2인 예제](../examples/gemma-tipo-presampling.json)를
불러오거나, 기존 노드에서 어댑터와 강도를 직접 선택하세요.
한국어 공통 프롬프트도 영어로 확장하려면 `expand_common`을 켭니다.
끄면 공통 입력 원문을 그대로 반환합니다.

태그는 새 지시문에서 직접 생성하므로 기존 학습본의 중간 설명→태그 호출을
추가하지 않습니다. 중복 정리, 원문 보존, 가중치·슬롯 검사, 추가 태그 예산과
제한된 오류 복구는 계속 적용합니다. 강도 변경은 캐시 식별에도 반영됩니다.

가중치 입력은 이제 [가중치 분리·확장·복구](gemma-weight-handling.ko.md) 방식으로
처리합니다. 숫자와 괄호는 노드가 보관하고, Gemma에는 내용만 전달합니다.
확장 후 원래 가중치를 복구하므로 가중치 표기 차이 때문에 새 묘사를 모두
원문으로 되돌리지 않습니다. 한국어 가중 구절은 같은 Gemma로 먼저 번역하여
가중치의 대상을 연결합니다.

이전 60회 비교에서 강도 0.25는 원시 JSON 15/15, 간단 확장 6/6이었지만
복잡한 조건 동시 검사는 3/9였습니다. 최신 연결에서도 의미·관계 보존은
확인이 필요하며, 모든 조건이나 10인 이미지의 정확성을 보장하지 않습니다.

릴리스 연결 검증: Python 196개·JavaScript 4종 통과. 비로그인 새 폴더에서
공개 어댑터 다운로드·해시 검증과 네트워크 호출 없는 재사용을 확인했습니다.
실제 ComfyUI에서 가중치 태그·한국어 2인·강도 0, 별도 Q4 한국어 2인을
실행해 강도 0.25의 패치 190개와 강도 0의 패치 0개를 확인했습니다.
네 번 모두 복구 경고 없이 완료했으며, 가중치 태그는 내부 재시도 1회 후
통과했습니다. 이번 배포 검사는 텍스트·연결 검증으로 이미지 품질 평가는 아닙니다.
