# Gemma 확장 복구 — 0.2.5

2026-11-04, 실제 ComfyUI 8188 노드 실행으로 확인했다.

## 원인과 수정

이전 요청은 장면의 사실을 보존하는 데 치우쳐 원문을 영어 문장으로
재표현하는 결과를 허용했다. 형식 오류 재시도에는 선택 디테일을 0개로
줄이라는 지시도 있었다. 이후 태그 변환은 중간 문장에 없는 내용을
만들지 않으므로, 새 디테일이 없는 문장은 확장되지 않은 태그로 돌아왔다.

0.2.5는 같은 Gemma+LoRA의 장면 계획에서 공통 조명·분위기와 인물별
시선·손동작·자세 등을 먼저 확장하도록 요청한다. 형식 오류를 고칠 때도
양수인 디테일 예산은 유지한다. 최종 태그 변환은 그 문장에 근거한 내용을
옮기며, 완전한 문장이 그대로 출력되면 예시를 포함한 요청으로 재시도한다.

영문 공통 프롬프트가 원문 재표현에 그친 경우에는 공통 필드만 짧은
자연어 요청으로 한 번 재시도한다. 이미 검증한 인물 설명은 보존한다.
이 검사는 보수적인 영문 단어 비교이므로 의미 정확도를 판정하지 않는다.
한국어 원문을 영어로 번역한 결과의 새 디테일 여부는 이 검사로 판단하지
않는다. 재시도 후에도 검사에 실패하면 기록을 포함한 오류를 낸다.

영문 태그 출력에는 기존과 같이 원본 입력을 보존한다. 가중치·이스케이프와
쉼표가 든 가중치 묶음을 변경하지 않고, 중복 제거와 개수 제한 후의
추가 태그를 붙인다. 중간 계획 입력에서만 빈 쉼표 항목을 정리한다.

## 실제 텍스트 비교

BF16과 `heretic_ara_q4`, `scale-20pct-001` LoRA, 시드 19,
`length=long`, `max_added_tags=6`, `max_tokens=2048`, `temperature=0.5`,
`output_format=tags`로 확인했다. 실제 모델 호출이며 이미지 생성 검증은 아니다.

2인 공통 입력:

```text
2people, cafe, (colorful theme:1.3), (smile, blush:1.5)
```

0.2.4의 같은 입력·시드는 원문 뒤에 `2girls, in a cafe, smiling, blushing`을
붙였다. 대부분 원문의 재표현이다. 0.2.5의 BF16 최종 출력은 다음과 같다.

```text
2people, cafe, (colorful theme:1.3), (smile, blush:1.5), soft lighting, gentle shadows, warm color palette, inviting atmosphere
```

Q4에서는 `soft lighting, evenly distributed across faces, gentle shadows,
depth to the scene`이 추가됐다. 인물별로도 컵을 향해 손을 뻗거나 컵을 보는
설명이 추가됐으며, 원본 영문 이름·의상·가중치 묶음은 유지됐다.

배경 없이 표정과 테마만 준 별도 1인 입력도 확인했다.

```text
1girl, (colorful theme:1.3), (open mouth, determined, one eye closed, :1.3)
```

이 입력에 BF16은 `soft lighting, gentle shadows, depth`, Q4는
`soft lighting, evenly distributed across the face, gentle shadows,
depth to the expression`을 추가했다. 원본 가중치 묶음은 보존됐다.
Q4 2인 입력에서 `expand_common=false`도 실행해 공통 출력이 원문과
완전히 같고, 인물 확장은 계속되는 것을 확인했다.

Python 검사 154개와 JavaScript 검사 스크립트 4개를 통과했다.
회귀 검사는 확장 예산, 공통 필드만 재시도, 인물 보존, 재시도 소진,
한국어 공통 재시도의 번역 보존 지시와 결과 집계를 포함한다.

## 결과 패널과 한계

결과 패널은 최종 필드별로 `+N tags`, `unchanged`, `disabled`,
`translated`를 표시한다. 세부 보고서의 `added_tags`, `added_tag_count`는
중복 제거·개수 제한 이후의 문자열 추가 수다. 의미상 새로운 디테일의
개수나 품질 점수가 아니다. 한국어 원문의 번역은 `translated`로 표시한다.

예를 들어 `wearing a blue jacket`처럼 원본 `blue jacket`을 바꿔 말한 태그나
캐릭터 이름의 다른 표기가 추가 개수에 포함될 수 있다. 모든 입력·시드에서
디테일이 추가되거나 의미가 보존된다고 보장하지 않는다. 한국어 점검 중에는
의상이 빠지는 결과가 나와, 먼저 모든 외형·의상을 번역한 뒤 동작을 확장하도록
지시를 보강했다. 같은 입력·시드의 Q4 재실행에서 파란 재킷·흰 드레스 보존과
손동작·시선 추가를 확인했다. 다만 한국어 공통 입력의 오후 조건 누락과
확장 없는 재표현은 0.2.4와 0.2.5 양쪽에서 관찰됐다. 한국어 공통 확장과
번역의 사실 보존을 일반적으로 해결한 패치라고 주장하지 않는다.

공통 필드 재시도와 태그 형식 재시도에는 실제 모델 호출이 추가되므로
항상 같은 속도라고 주장하지 않는다. 별도 모델, 추가 의존성이나 가중치
변경은 없고, 기존 네이티브 자원관리·PLE·LoRA 최적화는 유지한다.

원본 TIPO 저작자, KohakuBlueleaf와 기여자, Gemma 및 City96의 저작권과
라이선스를 유지한다. [NOTICE](../NOTICE)를 참고한다.
