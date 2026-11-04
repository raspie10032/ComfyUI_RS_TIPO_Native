# RS TIPO Native — 학습본 노드 적용, 2026-11-03

이번 마무리는 기존 20% 자료 추가학습본을 실제 ComfyUI 노드에 연결하고,
실험용 프롬프트 확장기로 편하게 사용하는 데 초점을 맞췄다.
새 학습이나 이미지 생성은 수행하지 않았다. 작은 모델의 의미 정확성을
완벽하게 해결했다는 검증이 아니다.

## 적용 가중치와 실행

- 노드: `RS TIPO Native (Experimental)`, 내부 ID `RSTIPOPrompt` 유지.
- Gemma 4 E2B heretic-ara Q4 베이스 + `local:scale-20pct-001` TIPO LoRA.
- LoRA rank 8, alpha 16, 실제 연결 대상 190곳.
- 가중치 SHA-256:
  `01dc2c9e5df07fd4437a146580158c2452e86a44b66543d5029de9d29889e28b`.
- 전역·활성 캐릭터 전체를 같은 모델로 읽고 자연어를 생성한다.
  태그 모드에서는 같은 어댑터가 붙은 같은 모델이 자연어를 태그로 변환한다.
- TIPO 2.1은 사용자가 선택하는 별도 독립 실행 경로다. Gemma 실행 중
  다른 모델로 전환하거나 연결하지 않는다.
- 공개 기본값 `pilot-002`는 유지하고, 이번 학습본은 명시적으로 선택한다.
  학습본 예제는 `examples/gemma-tipo-trained-local.json`.

## 노드 편의 기능

결과 패널에서 전체·전역·캐릭터 번호를 선택하고 최종 출력, 원문,
모델 상세·원시 출력을 구분해서 본다. 복사 버튼은 현재 보이는 내용을 복사한다.
모델 상세에는 실제 어댑터, rank, 가중치 checksum, 적용 대상 수가 표시된다.
원시 출력은 HTML로 실행하지 않으며 패널 DOM은 워크플로에 직렬화하지 않는다.
이름 변경은 기존 노드 ID·STRING 출력 번호·빌딩 파이프를 유지한다.

태그 생성 결과가 여러 줄이나 JSON이어서 형식 검증에 실패하면 동일 모델에
단일 태그 줄을 한 번 다시 요청한다. 실패 출력과 재요청 출력은 원시 기록에
남는다. 두 번째도 실패하면 설명을 포함한 오류를 낸다. 원문 반환을 성공으로
처리하거나 무한 재시도하지 않는다.

## 실제 입출력 확인

별도 ComfyUI 서버, RTX 5060 Ti에서 설치된 공개 노드 클래스를 실행했다.
한국어 꽃다발 전달·3인 연 만들기·전역만 있는 풍경, 영어 가중치가 있는
앞치마 묶기·짧은 우산 공유·캐릭터 수 축소 입력을 자연어/태그로 실행했다.
처음 12건 중 11건이 실행되었고, 도서관 태그 입력 한 건은 형식 오류였다.
수정 후 동일 실패 입력을 재실행했고, 다음 기록을 확인했다.

```text
최초 태그 응답:
The scene is depicted as follows: One woman reads in a quiet library.
TAGS:
one woman reads in a quiet library

동일 모델에 형식 재요청:
one woman reads in a quiet library
```

각 실제 요청은 STRING 소비 노드와 RS Finalize에 연결했다.
활성 개수를 벗어난 예전 캐릭터 입력은 출력과 파이프에서 제외됐고,
`(blonde hair:1.2)` 가중치가 유지됐다. 이 검사는 연결·실행의 증거이며
출력 의미가 모두 옳다는 점수로 해석하지 않는다.

브라우저에서도 학습본을 선택한 한국어 2인 예제를 실행했다. 새 워크플로에
저장한 어댑터와 STRING 입력이 재불러오기 후 유지되고, 네 입력 소켓의 높이가
각각 달랐다. 원문/출력/모델 상세 전환 및 복사 내용 일치도 확인했다.
이 요청의 모델 생성 시간은 26.6675초였다. 다른 GPU와 입력에서의 속도
보장은 아니며 다운로드·콜드 로드 시간을 포함한 수치도 아니다.

실제 캐릭터 출력:

```text
1: The blonde blue-eyed woman on the left is wearing a white shirt and is
handing a cup to the woman on the right. She is extending both hands toward
the cup and keeping the cup level.

2: The brown-haired green-eyed woman on the right is wearing a blue dress
and is receiving the cup from the left. She is reaching toward the cup with
both hands and looking at the cup.
```

한국어 꽃다발 입력에서는 금발 표현을 자연어에 남기지만 태그 변환에서
빠뜨리거나, 검은 머리를 자연어에서 누락하는 사례가 남았다.
태그에 자연어 구절이 섞이고 중복 표현도 생긴다. 이번 모델은 작은 실험용
확장기이며, 결과를 확인하고 필요하면 편집해서 쓰는 수준으로 제공한다.

## 재현 기록

- Python: 191개 중 184개 통과, 환경 의존 항목 7개 건너뜀.
- JavaScript: 소켓, 인터페이스, 다운로드 상태, 결과 패널 4개 파일 통과.
- 원시 기록: Git에서 제외한 `validation/trained-node-20261103/`.
  `first-api.json`은 최초 실패를 보존하고,
  `tag-format-retry.json`은 수정 후 재실행,
  `browser.json`은 실제 브라우저 실행과 모델 식별 정보를 기록한다.
- 실제 API 재현 도구: `tools/validate_trained_node.py`.
- 별도 로컬 시험 서버 전용 브라우저 도구: `tools/check_trained_ui.cjs`.
  별도 Playwright 설치와 Chromium 실행 경로가 필요하며 일반 사용자 브라우저
  프로필을 사용하지 않는다.
