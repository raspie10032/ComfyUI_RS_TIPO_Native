# RS TIPO Native 사용법

이 확장은 이미지 생성 전에 **전역과 캐릭터별 프롬프트를 보강**합니다.
TIPO v2.1을 PyTorch로 실행하고, 모델 적재·해제는 ComfyUI가 관리합니다.
주력 노드는 **RS TIPO Prompt (Experimental)**입니다.

캐릭터들은 같은 장면·상호작용을 참고하면서 각자의 외형과 담당 행동을
보강합니다. 다른 캐릭터의 외형 STRING은 해당 캐릭터의 추론에 넣지 않습니다.
공유 행동을 참고하는 기능이며, 역할을 자동으로 설계하거나 이미지의 동작을
보장하는 기능은 아닙니다. 결과 패널을 확인한 뒤 이미지 생성에 사용하세요.

## 1. 설치와 모델 준비

1. ComfyUI의 `custom_nodes` 폴더에서 아래 명령으로 설치하거나, 저장소의
   Code → Download ZIP을 받아 폴더를 풀어 넣습니다.

   ```sh
   git clone https://github.com/raspie10032/ComfyUI_RS_TIPO_Native.git
   ```
2. **ComfyUI를 실행하는 Python**으로 의존성을 설치합니다. 시스템 Python이나
   다른 가상환경에 설치하면 ComfyUI에서 찾지 못할 수 있습니다.
3. 같은 Python으로 아래 다운로드 도구를 실행합니다. 실행 위치는 프로젝트
   폴더이며, 모델 경로는 실제 ComfyUI 경로로 바꿉니다.

```sh
python -m pip install -r requirements.txt
python -m pip install huggingface_hub
python tools/download_model.py /path/to/ComfyUI/models/tipo/TIPO-v2.1-1B-A200M
```

Windows 포터블판은 해당 배포본의 `python_embeded/python.exe`, 가상환경은
그 환경의 Python을 명시해서 실행하세요. 위 명령의 `python`을 그 실행 파일로
바꾸면 됩니다. Windows 실행 검증은 아직 하지 않았습니다.

필요한 핵심 파일은 다음과 같습니다.

```text
ComfyUI/models/tipo/TIPO-v2.1-1B-A200M/
  config.json
  tokenizer.json
  model.safetensors
```

`extra_model_paths.yaml`에 등록한 `tipo` 경로도 검색합니다. 각 검색 경로 아래에
`TIPO-v2.1-1B-A200M` 폴더가 있어야 합니다. 기존 GGUF는 사용할 수 없습니다.
다운로더는 고정된 공식 모델 리비전의 데이터만 받습니다. 노드 실행 중에는
자동으로 모델을 다운로드하지 않습니다.

ComfyUI를 재시작하고 브라우저를 새로고침한 뒤 `RS TIPO Prompt`를 검색합니다.
노드 안에서 모델을 불러오므로 별도 모델 로더는 연결하지 않습니다.
STRING 출력은 단독 사용 가능합니다. 빌딩 파이프를 쓰려면
`ComfyUI_RS_MultiCharacter_Sampler`도 설치되어 있어야 합니다.

## 2. 빠르게 시작하기

[예제 shared-interaction.json](../examples/shared-interaction.json)을 ComfyUI에
끌어다 놓거나 워크플로 열기로 불러옵니다. 기본 Primitive STRING 입력 노드와
TIPO 결과 패널만 사용하며, 이미지를 생성하거나 NAI에 요청하지 않습니다.

1. `character_count`를 2로 둡니다.
2. 전역·캐릭터 1·캐릭터 2 STRING을 각각 연결합니다.
3. `expand_common=OFF`, `protect_source=ON`으로 시작합니다.
4. 실행하고 결과 패널에서 캐릭터별 `Original`, `Added`, `Excluded`,
   `Expanded`를 확인합니다.
5. 확인한 출력 STRING 또는 빌딩 파이프를 이미지 생성 워크플로에 연결합니다.

이 노드는 입력 문자열을 읽습니다. 이미지 모델이 이미 인코딩한 CONDITIONING은
입력으로 받지 않습니다. 영문 태그와 영문 상호작용 설명으로 먼저 시험하세요.
한국어·일본어 역할 이해의 품질은 검증하지 않았습니다.

## 3. 입력마다 담당할 내용

| 입력 | 적을 내용 |
| --- | --- |
| `common_prompt` | 전체 인원수, 배경, 시간, 공통 상황 |
| `character1` … `character10` | 각자의 외형·의상·위치·담당 행동 |
| `building_pipe` (선택) | 기존 위치·마스크·LoRA·참조·카메라 등 설정 |

예를 들어 컵을 건네는 장면을 만듭니다.

**전역**

```text
2girls, university cafe, afternoon, wooden table, sitting, indoors
```

**캐릭터 1 — 왼쪽에서 컵을 건네는 사람**

```text
(blonde hair:1.2), blue eyes, pink cardigan, on left, offering cup, extending arms
```

**캐릭터 2 — 오른쪽에서 컵을 받는 사람**

```text
black hair, brown eyes, yellow sweater, on right, receiving cup, reaching out
```

별도 상호작용 입력은 없습니다. 전역에 공통 상황을 적고 각 캐릭터에
`offering`과 `receiving` 같은 담당 행동을 적으면, 노드가 활성 캐릭터의
행동·위치 절을 모아 공유 문맥을 구성합니다. 보강 대상의 역할을 다시
표시하며, 각 캐릭터의 전체 원문은 해당 캐릭터에만 전달합니다.
결과 패널 상단에서 실제로 수집한 문맥을 확인할 수 있습니다.

현재 자동 수집은 영어 행동·위치 태그를 보수적으로 인식하는 방식입니다.
알 수 없는 표현은 해당 캐릭터 원문에만 남고, 상대에게 공유되지 않을 수
있습니다. 미지정 역할이나 좌우 위치를 임의로 만들어 넣지는 않습니다.
캐릭터 번호는 출력·파이프 슬롯의 대응이며 화면 좌우 순서가 아닙니다.
수집한 역할 목록을 이미지 생성용 전역 출력에 통째로 덧붙이지 않습니다.
전역에 특정 캐릭터의 외형을 적으면 다른 캐릭터도 그 정보를 읽게 됩니다.

## 4. 출력 연결

### STRING으로 사용

```text
전역 STRING ────┐                      ┌─ common_prompt → 생성기의 전역 입력
캐릭터 1 STRING ┼─ RS TIPO Prompt ──────┼─ character1 → 1번 캐릭터 프롬프트
캐릭터 2 STRING ┘                      └─ character2 → 2번 캐릭터 프롬프트
```

빌딩 파이프를 사용하지 않아도 됩니다. 같은 실행의 보강 결과를 STRING과
파이프에 함께 내보내므로 출력마다 TIPO를 다시 실행하지 않습니다.

### RS 빌딩 파이프로 사용

```text
전역·각 캐릭터 STRING → RS TIPO Prompt → LoRA / 위치 / 마스크 → 최종 파이프 → 샘플러
                              └─ common_prompt STRING ─────────────→ 전역 입력
```

기존 파이프를 TIPO의 `building_pipe`에 연결해 설정을 이어받을 수도 있습니다.
주력 `RS TIPO Prompt`의 **프롬프트 원문은 STRING 입력이 기준**입니다.
파이프만 연결해서 그 안의 캐릭터 문장을 자동으로 가져오는 모드가 아닙니다.
선택한 활성 슬롯은 STRING 표로 갱신되며 글자 삽입 슬롯은 뒤에서 추가하세요.
기존 파이프를 직접 읽어 보강할 때는 아래의 `Native Pipe Expand`를 사용합니다.

위치·LoRA 같은 설정 노드는 TIPO 앞뒤에 둘 수 있습니다. 캐릭터 프롬프트를
다시 쓰는 노드를 TIPO 뒤에 놓으면 그 노드가 보강 문장을 덮어쓸 수 있으니
프롬프트를 정하는 순서는 확인하세요. 샘플러가 전역 입력을 따로 받는 경우
TIPO의 `common_prompt` 출력도 연결해야 최종 전역을 명시적으로 전달합니다.

`character_count`를 5→3으로 줄이면 4·5번은 실행하지 않고 해당 STRING 출력은
빈 문자열이 됩니다. 연결된 비활성 소켓은 나중에 다시 사용할 수 있도록
남아 있을 수 있습니다. **0명으로 전역만 보강하려면 `expand_common=ON`을
켜세요.** OFF이면 전역 원문을 그대로 전달하고 전역 보강 추론은 하지 않습니다.
0명은 전역 STRING만 사용하는 모드이며 빌딩 파이프를
소비하는 노드에 연결하면 오류로 안내합니다.

## 5. 조절항목

| 항목 | 기본값 | 의미 |
| --- | --- | --- |
| `character_count` | 2 | 활성 캐릭터 0~10명 |
| `expand_common` | OFF | 전역에도 태그를 추가할지 선택. OFF이면 전역 원문 유지 |
| `protect_source` | ON | 알려진 외형·의상·구도 변경과 역할 충돌 태그를 보수적으로 제외 |
| `length` | long | 모델이 학습한 목표 길이. short / long / very_long |
| `advanced` | OFF | 아래의 상세 설정 표시. 숨겨도 값은 유지 |
| `max_new_tokens` | 128 | 출력 토큰 상한. 길이 제한에 걸린 마지막 미완성 태그는 제거 |
| `seed` | 0 | 텍스트 샘플링 시드. 0도 유효. 이미지 생성 시드와 별개 |
| `temperature` | 0.5 | 텍스트 샘플링 다양성. 0은 가장 높은 점수의 토큰을 선택 |
| `common_weight` | 0.5 | 캐릭터 보강 시 전역 입력 토큰 참고 강도, 0~4 |
| `character_weight` | 1.5 | 자기 캐릭터 입력 토큰 참고 강도, 0~4 |

강도는 **TIPO 내부 어텐션**을 조절합니다. 이미지 생성 CFG나
`(blonde hair:1.2)`의 가중치와 다릅니다. 원문의 가중치는 출력에 보존하고,
TIPO 입력에는 지원하는 가중치 문법을 정리한 태그 내용을 전달합니다.

강도 1은 중립, 0은 해당 입력 구간을 차단합니다. `common_weight=0`이면
전역과 자동으로 모은 행동 문맥을 모두 차단합니다. 원래 전역은 출력에 남습니다.
전역만 보강하는 호출에는 `common_weight`가 캐릭터 호출처럼 적용되지 않습니다.
강도를 높여도 내용이 더 정확해진다고 보장하지 않습니다. 우선 기본값을 쓰고,
같은 입력·고정 시드·temperature 0에서 1/1과 비교해 보세요.

## 6. 결과 확인과 문제 해결

결과 패널의 캐릭터 선택기에서 `Original`(원문), `Added`(추가),
`Excluded`(제외 태그와 이유), `Expanded`(최종)을 확인합니다.
전역 보강을 끄면 전역은 별도 생성 결과 목록에 없고 캐릭터 결과가 표시됩니다.

- **활성 캐릭터 STRING이 비어 있음:** 해당 입력을 연결하거나 인원수를 줄입니다.
- **모델이 없음 / 폴더가 불완전함:** 모델 경로와 핵심 파일 3개를 확인합니다.
- **빌딩 파이프 확장이 없다는 오류:** RS 멀티캐릭터 확장을 설치하거나 STRING만 사용합니다.
- **문맥 길이 초과:** 전역·캐릭터 문장을 줄이거나 출력 토큰 상한을 줄입니다.
  4096 토큰 문맥을 넘기면 조용히 잘라 쓰지 않고 오류로 알립니다.
- **추가 태그가 거의 없음:** 보호 기능이 후보 대부분을 제외했을 수 있습니다.
  `Excluded`를 먼저 확인하세요. OFF 비교는 가능하지만 외형·역할 변경을 직접 검토해야 합니다.
- **의미가 어긋남:** 각자의 담당 행동을 원문에 명시하고 전역 상황과의 모순을 줄입니다.
  현재 보호 기능은 어휘 기반이며 모든 동의어·부정·관계 충돌을 이해하지 못합니다.
- **노드가 안 보이거나 옛 UI가 남음:** ComfyUI 재시작 후 브라우저를 새로고침하고
  서버의 확장 로드 오류를 확인합니다.

모델은 요청 중 전체 적재가 필요합니다. FP16 파라미터만 약 1.85GiB이며
추론·캐시에 추가 메모리를 씁니다. 캐릭터는 각각 독립 추론하므로 인원수가
늘면 텍스트 보강 시간도 늘어납니다. 부분 레이어 오프로딩·양자화는 미지원입니다.

## 7. 다른 노드는 언제 쓰나요?

| 노드 | 사용 목적 |
| --- | --- |
| RS TIPO Prompt | 보통 사용. 분리 STRING 입력 → 분리 STRING + 빌딩 파이프 |
| RS TIPO Native Completion | 공식 형식 원문과 원시 완성을 비교하는 저수준 시험 |
| RS TIPO Native Expand | 문자열 하나의 기본 보강. 캐릭터 여러 명을 합쳐 넣지 않기 |
| RS TIPO Native Controlled Expand | 전역 + 캐릭터 한 명의 어텐션 강도 시험 |
| RS TIPO Native Pipe Expand | 기존 빌딩/최종 파이프·NAI 컨텍스트의 활성 캐릭터 문장 보강 |

`Native Pipe Expand`는 캐릭터 프롬프트가 정해진 뒤 연결합니다. 비활성·글자
슬롯은 보강하지 않고 기존 설정을 보존합니다. 전역 STRING이 비면 기존 장면을
계승합니다. 장면을 저장하지 않는 새 초안 파이프에서는 이 노드의 전역 STRING도
생성기에 직접 연결하세요. 주력 STRING 노드와 입력 계승 규칙이 다릅니다.

## 검증 범위와 출처

Linux에서 ComfyUI v0.3.26 / 프런트엔드 1.11.8,
v0.3.64 / 1.27.10, 현재 0.41.0 / 1.55.17의 실제 텍스트 실행·화면 연결·저장
왕복을 확인했습니다. 같은 현대 Python/PyTorch 환경을 사용한 시험이며,
구형 PyTorch·Windows·모든 중간 버전의 지원을 보증하지 않습니다.
이미지 품질 개선과 일반적인 상호작용 정확도는 아직 미검증입니다.

- [레퍼런스와 각 자료에서 사용한 부분](references.md)
- [전역·캐릭터에서 자동으로 상호작용 문맥 구성 및 연결선 수정](automatic-context.md)
- [구형 호환성 보고서](compatibility.md)
- [역할·원문 보호의 실제 비교](source-protection.md)
- [어텐션 제어 방식과 수치 시험](owned-attention.md)

## 워크플로 PNG와 현재 제한

[워크플로 포함 PNG](../examples/shared-scene-workflow.png)를 원본으로 저장한 뒤
ComfyUI 화면에 끌어다 놓으면 예제를 불러올 수 있습니다. 노드 설치와 모델 준비는
먼저 완료해야 합니다. 미리보기나 JPEG로 변환한 파일에는 워크플로가 없을 수 있습니다.

이번 시험판에는 사용자가 원하지 않는 태그를 직접 지정하는 제외 목록 입력은
아직 없습니다. `protect_source`는 알려진 원문 충돌을 막는 별도의 보호 기능입니다.
