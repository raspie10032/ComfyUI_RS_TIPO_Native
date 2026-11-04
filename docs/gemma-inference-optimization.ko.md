# Gemma TIPO 추론 비용 정리 — 2026-11-04

## v0.2.2 배포 범위

아래 네 단계의 추론 최적화를 v0.2.2에 포함한다. 원본 TIPO의 실행 경로와
기존 0.2.1의 Gemma 출력 복구는 유지한다. Gemma BF16의 PLE를 CPU로 옮기는
변경은 포함하지 않는다. 추가 패키지, 별도 추론 엔진, 상주 GPU 캐시는 없다.

각 단계는 서로 다른 직전 상태와 비교했다. 따라서 표의 시간 감소율을 합하거나,
모든 표를 0.2.1 대 0.2.2의 단일 비교로 해석하면 안 된다. 아래 테스트 개수는
당시 연구 체크아웃 전체의 기록이다. 배포 체크아웃에서는 별도로 Python 131개와
JavaScript 4개를 검사했다. 이미지 모델과의 교대 생성은 이 측정 범위에 없다.

배포 체크아웃의 실제 RTX 5060 Ti Q4 + `scale-20pct-001` 2인 생성도 별도로
확인했다. 버퍼 재사용 전후 토큰/최종 출력과 슬롯 검사가 일치하고, 최대
allocated는 1761.90→1729.00 MiB였다. 오프로딩·재로딩·취소·샘플러 복원·
취소 후 재생성과 금지어 검사를 통과했다. 마지막 unload 후 allocated는
32 MiB, ComfyUI 로드 모델은 0개였다. 이 재검사는 2인 입력 한 쌍이며,
기존 2·6·10인 전체 실험을 새로 반복한 것으로 세지 않는다.

공개된 [실측 요약 JSON](gemma-inference-benchmark-results.json)은 장치·버전·측정값,
원본 로컬 기록의 SHA-256 및 생성 토큰/최종 출력의 해시를 보관한다. 아래
`validation/...` 경로는 로컬 원본 보관 위치이며 배포 패키지에 포함하지 않는다.

공개 0.2.1 코드에서 비교용 파일을 얻어 v0.2.2 체크아웃의
`tools/benchmark_gemma_inference.py`로 다시 측정할 수 있다. `c04702c`는 0.2.1
배포 커밋이다. 다음 명령은 전체 변경을 함께 비교하는 새 실험이며, 아래의
단계별 기록과 동일한 실행 조건을 재현하는 명령은 아니다.

```bash
git show c04702c:gemma_runtime.py > /tmp/rs-tipo-runtime-0.2.1.py
git show c04702c:tipo_gguf_ops.py > /tmp/rs-tipo-ops-0.2.1.py
git show c04702c:vendor/gguf_torch/dequant.py > /tmp/rs-tipo-dequant-0.2.1.py
CUDA_VISIBLE_DEVICES=0 python tools/benchmark_gemma_inference.py \
  --comfy-root /path/to/ComfyUI \
  --baseline-runtime /tmp/rs-tipo-runtime-0.2.1.py \
  --baseline-gguf-ops /tmp/rs-tipo-ops-0.2.1.py \
  --baseline-dequant /tmp/rs-tipo-dequant-0.2.1.py \
  --compare-ple-placement --lifecycle-checks \
  --model heretic_ara_q4 --adapter scale-20pct-001 \
  --counts 2 6 10 --repeats 2 --output /tmp/rs-tipo-benchmark.json
```

ComfyUI 환경의 Python과 이미 설치된 모델을 사용한다. 모델이 없으면 모델
로더가 기존 정책에 따라 다운로드할 수 있다. 이 스크립트는 이미지 생성이나
유료 API를 호출하지 않는다. Comfy Registry 패키지에는 개발용 `tools/`를
제외하므로 스크립트는 GitHub 체크아웃에서 사용한다.

## 적용한 변경

`gemma_runtime.token_constraints()`는 금지 태그가 없을 때 ComfyUI 샘플러에
logits와 난수 생성기를 그대로 전달한다. 이 경우 매 토큰마다 하던 전체 vocabulary
유한값 검사, 추가 `token.item()` 동기화, 누적 토큰 전체 디코딩과 정규화를 생략한다.
취소 확인, 진행률, 샘플러 복원은 유지한다. 금지 태그가 있으면 기존 차단 경로를 쓴다.

`tipo_gguf_ops.GGUFOps.Embedding.project()`는 연속된 출력 행을 `slice`로 고른다.
기존 `arange`와 advanced indexing에 의한 packed-weight 복사본이 필요 없어졌다.
입력 embedding의 임의 토큰 gather와 출력층의 청크 역양자화는 유지한다.
FP32 역양자화, 추론 dtype 변환, LoRA 및 행렬 연산의 수치 경로를 바꾸지 않는다.
새 추론 엔진, 프리빌트 휠, 컴파일러 또는 추가 의존성은 도입하지 않는다.

## 검증 방법과 범위

변경 직전 로컬 파일을 별도로 저장하고 같은 프로세스에서 before/after를 비교했다.
이미 존재하던 다인물·의상 관련 수정은 양쪽에 동일하게 포함한다. GPU 테스트는
설치된 실제 ComfyUI 네이티브 경로, 실제 모델과 `scale-20pct-001` LoRA 190개
대상을 사용했다. 원본 TIPO, Google Gemma 및 포함된 GGUF 코드의 저작권·라이선스는
기존 고지대로 유지한다.
실행한 ComfyUI는 `/home/raspie/AI/ComfyUI`이며 서비스와 별도 Python 프로세스로
측정했다. 마운트된 `ComfyUI_0_NAI` 체크아웃을 실행한 시험은 아니다.

한국어 공통 프롬프트와 외형·의상·소품·가중치가 다른 2·6·10인을 입력했다.
단순히 최종 문장만 보지 않고 생성 token ID와 파싱된 모든 슬롯을 비교한다.
인물 수, `(hair:1.15)` 가중치, 눈·의상·소품의 명시적 단어와 다른 인물의 머리색
유입도 검사한다. 이는 고정된 회귀 예제이며 전체 의미 보존이나 이미지 품질에 대한
포괄적 보장은 아니다. 자연어 출력을 검사했고 이미지 생성과 NAI API는 실행하지 않았다.

측정값은 해당 Python 프로세스의 PyTorch CUDA allocated/reserved와 프로세스 RSS다.
다른 ComfyUI·데스크톱 프로세스를 포함하는 GPU 전체 사용량과는 다르다.
각 실행 후 CUDA를 동기화하며 RSS는 20ms 간격으로 샘플링한다.
최초 모델 로딩, 최초 CUDA 초기화 및 각 입력 길이의 초기 실행은 성능 비교에서
분리해야 한다. 특히 첫 before 실행과 직후 after 실행의 차이를 속도 향상으로
해석하면 안 된다. 짧은 반복 측정의 작은 차이는 잡음일 수 있다.

## 첫 변경의 실제 결과

- RTX 4070 Ti SUPER + `heretic_ara_q4`: 2·6·10인 각각 before/after 2회,
  총 12회 성공. 모든 반복에서 token ID와 최종 슬롯이 기준 실행과 일치했다.
  실행 시간 변화는 대략 -0.3%~+0.4%로, 유의미한 속도 향상 근거가 없다.
  최대 PyTorch allocated는 입력별 약 3576/3620/3670 MiB로 같았다.
- RTX 5060 Ti + `gemma4-e2b-native-bf16.safetensors`: 동일한 12회 검증을
  통과했다. 초기 실행을 제외하면 속도 변화는 작다. 최대 allocated는 약
  9954/10003/10055 MiB로 사실상 같다.
- 같은 BF16 모델에서 temperature=0.3, seed=42의 2인 샘플링 비교도
  생성 token ID 전체가 일치했다. CPU 테스트에서도 난수 생성기 상태 보존을 검사한다.
- 반복 후 모델을 보유한 상태의 allocated가 증가하지 않았다. ComfyUI
  `unload_all_models()`와 `soft_empty_cache()` 후 allocated는 Q4 약 8.1 MiB,
  BF16 32 MiB였다. 이는 모델을 CPU로 내린 결과이며 CPU RAM을 모두 해제한다는
  뜻은 아니다. 실제 이미지 모델과의 교대 생성까지 검증한 결과는 아니다.

후속 최종 비교에는 출력 행 slice 변경까지 포함하며, 입력별 4토큰 warmup을 추가한다.

## 두 변경을 합친 최종 Q4 비교

RTX 5060 Ti에서 동일 입력·시드로 before/after 각 1회씩 실행했다. 아래 시간은
전체 `run_scene()` 시간이고 VRAM은 해당 실행의 최대 PyTorch allocated다.

| 입력 | before | after | 최대 VRAM before / after | 생성 token ID·출력 |
| --- | ---: | ---: | ---: | --- |
| 2인 | 33.596초 | 32.883초 | 3599.90 / 3599.90 MiB | 완전 일치 |
| 6인 | 77.155초 | 75.891초 | 3644.17 / 3644.17 MiB | 완전 일치 |
| 10인 | 121.555초 | 120.851초 | 3691.29 / 3691.29 MiB | 완전 일치 |

모든 실행에서 인물 슬롯 수와 외형·의상·소품의 명시적 단어 보존 검사를 통과했다.
이번 단일 비교의 시간 감소는 약 0.6~2.1%다. 고정 순서의 짧은 비교라 확정적인
성능 향상으로 주장하지 않는다. 최대 VRAM과 프로세스 RSS 차이도 사실상 없다.
모델을 보유한 상태의 allocated는 약 3298 MiB였고 ComfyUI unload 후 32 MiB,
reserved 54 MiB, ComfyUI loaded-model 목록 0개였다. CPU 오프로딩 후 RSS는
증가하므로 이 결과를 시스템 전체 메모리 해제로 해석하면 안 된다.

같은 GPU에서 앞서 측정한 BF16 10인 after는 8.113/8.128초, 최대 allocated 약
10055 MiB였다. Q4의 낮은 저장 메모리가 이 구현에서 빠른 추론을 뜻하지는 않는다.
모델 파일과 양자화·실행 경로가 다르므로 이 수치를 양자화 방식만의 효과로
분리한 비교라고 주장하지 않는다.

## 다음 최적화의 근거

RTX 4070 Ti SUPER에서 최종 로컬 코드, Q4 모델, 동일 LoRA의 4토큰 실행을
별도로 프로파일링했다. CPU 호출 이벤트 기준 `GGUF dequantize` 1,240회,
`LoRA calculate_weight` 760회가 관측됐다. 이 샘플에서는 LoRA 190개 대상의
전체 가중치 반영이 토큰마다 반복된다. 총 실행시간으로부터 이들 비용의 비율을
계산하지 않는다. 중첩된 프로파일러 범위의 시간은 서로 더하면 중복될 수 있다.
이는 prefill과 decode가 섞인 짧은 샘플이며 장문 전체의 분포를 보장하지 않는다.

다음 실험 후보는 매번 큰 `B @ A` 행렬을 만들어 가중치에 더하는 대신,
기본 출력에 작은 LoRA 출력 `scale * B(A(x))`를 더하는 경로다. 설치된 ComfyUI의
`LoraDiff.h()`에도 이 방식의 구현이 있으므로 기존 네이티브 기능의 재사용을
먼저 검토했다. 이 시점의 1차 실험은 해당 경로를 적용하기 전의 결과다.
후속 구현과 검증은 아래 2차 실험에 기록한다. 수학적으로 같은 식이어도 낮은
정밀도의 연산 순서가 달라져 출력이 달라질 수 있다.

역양자화 비용은 FP32 임시 텐서와 dtype 변환을 별도로 검토한다. 풀어놓은 가중치의
상시 GPU 캐시는 VRAM을 늘릴 수 있고 ComfyUI가 회수하지 못하는 캐시는 자원관리
통합을 훼손하므로, 이번 변경에는 추가하지 않았다.

## 재현 도구와 로컬 증거

`tools/benchmark_gemma_inference.py`는 모델·어댑터, GPU, 생성 토큰, 지연시간,
최대 메모리, 슬롯 검사와 ComfyUI unload 결과를 JSON에 기록한다. 최종 결과의
token/slot 불일치 또는 보존 검사 실패는 비정상 종료 코드로 보고한다.
프로파일러는 역양자화·unpack·LoRA 계산과 주요 PyTorch 연산을 구분한다.
`--profile-only --profile-tokens 4`로 전체 비교 없이 프로파일링만 실행할 수 있다.

이 연구의 기준 파일과 원본 결과는 Git에서 제외되는
`validation/gemma-inference-20261104/` 아래에 보관한다.

```bash
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  /path/to/comfy/.venv/bin/python tools/benchmark_gemma_inference.py \
  --comfy-root /path/to/comfy \
  --model heretic_ara_q4 --adapter scale-20pct-001 \
  --baseline-runtime validation/gemma-inference-20261104/gemma_runtime.before.py \
  --baseline-gguf-ops validation/gemma-inference-20261104/tipo_gguf_ops.before.py \
  --counts 2 6 10 --repeats 1 --profile-tokens 4 \
  --output validation/gemma-inference-20261104/q4-final-5060ti.json
```

모델이 로컬에 준비된 환경에서 실행한 명령이다. baseline 파일은 측정 직전 상태를
보존한 로컬 스냅샷이며, 과거의 다른 버전을 비교하면 결과의 의미도 달라진다.

Python 전체 검사 210개 중 203개 통과, 선택 의존성 관련 7개 skip.
인터페이스·소켓·다운로드 상태·리포트 JavaScript 검사 4개 통과.
코드와 테스트는 로컬 변경이며 이 연구만으로 새 릴리스나 GitHub 반영을 뜻하지 않는다.

## 2차 실험: GGUF Linear LoRA 출력 계산

일반적인 2D Linear LoRA는 전체 `B @ A` 가중치를 토큰마다 만들지 않고,
`F.linear(x, W) + strength * B(A(x))`로 계산한다. 작은 LoRA 연산은 ComfyUI의
`LoRAAdapter.h()`를 재사용한다. `alpha / rank`와 양수·음수 강도, 여러 LoRA를
그대로 반영하며 원래 어댑터를 수정하지 않는다. 장치로 옮긴 작은 행렬은 호출
범위에서만 사용하고 상주 GPU 캐시나 별도 모델 보관소를 만들지 않는다.

패치 체인 전체가 지원되는지 먼저 검사한다. DoRA, LoCon/mid, reshape, offset,
가중치 변환 함수, model strength 변경 또는 native h API 미지원 환경은 기존
가중치 계산을 쓴다. 지원되는 패치만 일부 먼저 반영하는 일도 없다.
BF16 로딩 경로와 GGUF 역양자화 정밀도는 변경하지 않는다.

같은 입력으로 두 계산을 비교한 CPU 회귀 검사에는 다중 LoRA, 음수·0 강도,
구형 API, 특수 패치 전체 fallback, 가중치 복제 후 독립 상태와 어댑터 비변경을 포함한다.
실제 모델 검사는 별도 프로세스와 로컬 모델로 실행하며 ComfyUI 서비스 재시작,
추가 다운로드, 이미지 생성 또는 NAI API 호출은 필요하지 않다.

2차 baseline은 바로 위의 1차 최적화가 포함된 상태다. 따라서 1차 실험의 최초
baseline과 수치를 섞어 새 변경의 개선율을 계산하지 않는다.
스냅샷, 로그와 JSON은 `validation/gemma-lora-bypass-20261104/`에 보관한다.

### 실제 Q4 모델 비교

RTX 5060 Ti, `heretic_ara_q4`와 `scale-20pct-001` LoRA, temperature 0에서
입력별 4토큰 warmup 후 before/after 각 1회씩 측정했다. 아래 시간은 전체
`run_scene()` 시간이다.

| 입력 | before | after | 시간 감소 | 최대 VRAM before / after | 생성 token ID·출력 |
| --- | ---: | ---: | ---: | ---: | --- |
| 2인 | 33.200초 | 27.001초 | 18.7% | 3599.90 / 3599.90 MiB | 완전 일치 |
| 6인 | 77.118초 | 61.382초 | 20.4% | 3644.17 / 3644.17 MiB | 완전 일치 |
| 10인 | 119.794초 | 96.716초 | 19.3% | 3691.29 / 3691.29 MiB | 완전 일치 |

모든 실행에서 지정 슬롯 수와 외형·의상·소품 단어 보존 검사를 통과했다.
최대 VRAM과 프로세스 RSS 감소는 관측되지 않았다. 고정 순서의 단일 비교이므로
다른 모델·장치·입력에서 같은 개선율을 보장하지 않는다.

실제 LoRA 레이어 760회에서 기존 가중치 계산과 새 출력 계산을 나란히 비교했다.
NaN·Inf는 없었고, 최대 절대 오차는 4.0, 최대 상대 RMS 오차는 약 0.004887
(0.489%)였다. 내부 수치는 bit-exact하지 않다. 위 greedy 생성은 token ID까지
같았지만, 모든 시드와 입력에서 토큰 일치를 보장하는 결과는 아니다.

추가로 RTX 4070 Ti SUPER에서 temperature=0.3, seed=42의 2인 생성은
before → after → after → before 순서로 4회 비교했다. before는
20.271/20.086초, after는 16.661/16.636초였고 생성 token ID와 최종 슬롯이
모두 일치했다. 평균 시간 감소는 약 17.5%다. 샘플링 검사는 이 한 입력과
시드에 한정되며 6·10인의 샘플링까지 비교한 결과는 아니다.

### BF16 대조 검사

RTX 5060 Ti에서 `gemma4-e2b-native-bf16.safetensors`와 같은 어댑터로
before/after 각 1회씩 비교했다. BF16은 이번 GGUF Linear 변경의 대상이 아니다.

| 입력 | before | after | 최대 VRAM before / after | 생성 token ID·출력 |
| --- | ---: | ---: | ---: | --- |
| 2인 | 2.203초 | 2.209초 | 9953.64 / 9953.64 MiB | 완전 일치 |
| 6인 | 4.951초 | 5.007초 | 10003.25 / 10003.25 MiB | 완전 일치 |
| 10인 | 7.859초 | 7.769초 | 10054.27 / 10054.27 MiB | 완전 일치 |

의상·외형·소품 및 슬롯 검사도 모두 통과했다. 시간 차이는 약 ±1.2%로 작고,
BF16 경로의 성능 향상으로 해석하지 않는다. BF16 역시 unload/reload와
취소 후 복구 검사를 통과했으며 최종 CUDA allocated는 32 MiB였다.

### 반복 계산과 자원관리 검증

변경 후 4토큰 프로파일의 CPU 이벤트에서 `LoRA output bypass`는 760회,
`LoRA calculate_weight`는 0회였다. `GGUF dequantize`는 여전히 1,240회다.
이번 개선은 반복하는 전체 LoRA 가중치 계산을 없앤 결과이며, 역양자화 자체의
반복 횟수나 FP32 중간 텐서를 줄인 결과는 아니다. 원본 프로파일의 CPU/CUDA
이벤트는 따로 기록하며 두 종류를 합산해 호출 수를 부풀리지 않는다.

실제 ComfyUI unload/reload 검사에서 기존 `PackedWeight`를 inference mode
안에서 만들면 다음 로딩 시 `Inference tensors do not track version counter`로
실패하는 문제를 재현했다. 영구 가중치 래퍼를 만드는 순간만 inference mode를
해제해 version counter를 유지하도록 수정했다. packed 데이터는 복제하지 않는다.
`to(copy=True)` 인자 처리도 고쳤고, 이 경로와 clone의 version counter를
회귀 검사에 포함했다.

수정 후 Q4는 ComfyUI unload → reload, 생성 취소, 샘플러 원상 복구,
취소 후 재생성을 모두 통과했다. 재로딩과 취소 복구 후 4토큰 생성이 최초
warmup과 일치했다. 금지 태그가 설정된 경로도 실행해 기존 단어 경계 차단과
차단 hit 보고를 확인했다. 이는 모든 우회 표현이나 부분 문자열을 막는다는
검증은 아니다.

최종 unload 후 해당 테스트 프로세스의 CUDA allocated는 32 MiB,
reserved는 54 MiB, ComfyUI 로드 모델 목록은 0개였다. LoRA 출력 계산을
위한 별도 상주 GPU 캐시는 없다. CPU에 오프로딩한 모델과 프로세스 RSS까지
해제한다는 의미는 아니며, 실제 이미지 모델과 교대하는 생성은 아직 범위 밖이다.

최종 Python 검사 217개 중 210개 통과, 선택 의존성 관련 7개 skip.
JavaScript 검사 4개 통과. 이번 변경은 로컬 연구 코드이며 공개 릴리스는 별도다.

## 세 번째 변경: GGUF PLE를 CPU에 보관하고 행만 전송

Gemma의 `embed_tokens_per_layer`는 입력 토큰의 보조 임베딩을 조회하는 표다.
현재 `heretic_ara_q4`의 표는 논리적으로 262144 × 8960이며 Q6_K 저장 크기는
1,837.5 MiB다. 이번 변경은 양자화 정밀도나 모델 구조를 바꾸지 않고 이 표를
CPU에 보관한다. 기존 `unpack(..., rows=input)`이 CPU에서 필요한 행만 고른 뒤
packed 행을 GPU로 옮겨 동일한 PyTorch 역양자화와 연산을 한다.

GGUF 로더에서 기본 활성화한다. BF16 네이티브 모델과 일반 입력/출력 임베딩은
변경하지 않는다. 기존 ComfyUI·PyTorch 실행 경로를 유지하고 추가 의존성이나
별도 GPU 캐시를 만들지 않는다.

`Embedding._apply()`가 모델 전체의 `.to(device)`에서도 PLE 가중치를 CPU에
남기고, `GGUFModelPatcher._load_list()`가 해당 표를 로딩 및 부분 오프로딩
대상에서 제외한다. `model_size()`도 실제 GPU에 로드할 가중치만 집계한다.
PLE 가중치 패치가 있거나 이미 로드된 상태에서 배치를 바꾸려 하면 명시적으로
거부한다. 이 표의 LoRA와 GPU 사용 중 배치 전환은 지원 범위에 포함하지 않는다.

### PLE 배치만 비교하는 방법

이번 before 파일에는 앞서 적용한 LoRA 출력 연산과 slice 최적화가 이미 들어 있다.
각 모드 전환에서 ComfyUI로 완전히 unload한 뒤 PLE GPU 상주/CPU 행 전송을
선택하고, 같은 입력으로 4토큰 warmup을 실행한다. unload·reload·warmup 시간은
측정에서 제외한다. 같은 프로세스·모델·LoRA·입력·시드에서 PLE 위치만 비교한다.
메모리는 다른 프로세스를 제외한 PyTorch allocated/reserved이며 RSS도 기록한다.

측정과 원본 JSON은 `validation/gemma-ple-offload-20261104/`에 보관한다.
`tools/benchmark_gemma_inference.py --compare-ple-placement`가 배치 전환과
ComfyUI 로딩 메모리를 기록하며, `--lifecycle-checks`는 재로딩·취소·복구·금지어
검사를 추가한다. 비교 시 변경 직전 runtime/ops 파일을 인자로 함께 지정한다.

### RTX 5060 Ti 실제 결과

2·6·10인 입력 각각 GPU 상주/CPU 보관 1회씩 비교했다. 각 실행은 성공했고
모든 생성 token ID와 최종 공통/인물 슬롯이 일치했다. 명시적 외형·의상·소품과
다른 인물의 머리색 유입 회귀 검사도 통과했다.

| 입력 | GPU 상주 | CPU 행 전송 | 최대 allocated 전 / 후 | 생성 token ID·출력 |
| --- | ---: | ---: | ---: | --- |
| 2인 | 27.110초 | 27.486초 | 3600.59 / 1762.59 MiB | 일치 |
| 6인 | 61.952초 | 61.449초 | 3644.32 / 1806.32 MiB | 일치 |
| 10인 | 97.916초 | 97.781초 | 3691.37 / 1853.37 MiB | 일치 |

모든 입력에서 최대 allocated가 1,838 MiB(약 1.795 GiB) 줄었다. PLE 표의 실제
packed 크기 및 ComfyUI 모델 메모리 집계 차이는 1,837.5 MiB이며 CUDA 할당
단위 반올림 때문에 allocated 차이와 0.5 MiB 차이가 있다. 단일 비교에서 시간
변화는 +1.4% / -0.8% / -0.1%였다. 유의미한 속도 개선이나 모든 환경에서 무손실
전송 비용을 보장하는 결과로 해석하지 않는다.

ComfyUI 로드 가중치 집계는 3253.44→1415.94 MiB로 줄었고, 전체 모델이 로드된
상태에서도 PLE의 장치는 CPU였다. 완전 unload·reload, 2토큰 후 취소·샘플러
복원·취소 후 재생성을 통과했다. 마지막 unload 후 allocated 32 MiB,
reserved 54 MiB, ComfyUI loaded-model 목록 0개를 확인했다.

### 교차 검사와 자원관리

RTX 4070 Ti SUPER에서는 temperature=0.3, seed=42의 2인 입력을
before→after, after→before 순서로 총 4회 실행했다. 모든 token ID·슬롯이
일치했고, before 평균 16.333초, after 평균 16.315초로 차이가 거의 없었다.
최대 allocated는 3576.47→1738.47 MiB였다. 6·10인의 샘플링 전체 검증은 아니다.

BF16 대조에서는 2·6·10인 각 1회씩 before/after를 실행했다. 토큰과 슬롯이
모두 같고 최대 allocated도 각 입력에서 9930.77 / 9980.75 / 10032.62 MiB로
변하지 않았다. 이번 메모리 절감이 BF16에도 적용되었다고 해석하지 않는다.

별도 실제 Q4 검사에서 부분 unload로 ComfyUI 집계 313.41 MiB를 내렸다.
PLE는 계속 CPU에 남았고, 정상 reload 후 처음과 같은 4토큰을 생성했다.
완전 unload 후 GPU allocated는 8.125 MiB, 로드된 모델은 0개였다.
CPU 표 보존·모델 전체 이동·메모리 크기 재계산 회귀 검사를 추가했다.
Python 220개 중 213개 통과, 선택 의존성 7개 skip; JavaScript 4개 통과.

이는 VRAM을 시스템 RAM으로 옮긴 결과다. 프로세스 RSS는 실행 조건에 따라
약 1.7~2.0 GiB 늘었다. 시스템 전체 메모리 감소나 이미지 모델과 교대하는 실제
생성 검증으로 해석하지 않는다. 현재 로컬 코드에 적용한 상태이며 실행 중인
ComfyUI 프로세스에는 노드 재로딩 또는 서버 재시작 후 적용된다.

## 네 번째 변경: Q4_K/Q5_K 역양자화 결과 버퍼 재사용

City96의 PyTorch 역양자화 코드에서 `(d * q - dm)`는 큰 곱셈 결과와 뺄셈 결과를
별도로 할당했다. Q4_K/Q5_K에 한해 새로 할당한 곱셈 결과에 `sub_(dm)`를
적용해 같은 크기의 두 번째 실수 텐서를 없앴다. 연산 순서와 FP32 중간 정밀도는
유지한다. packed 원본이나 공유 가중치에는 쓰지 않고 상주 캐시도 추가하지 않는다.
City96의 저작권·라이선스를 유지하고 `vendor/gguf_torch/NOTICE.md`에 수정 범위를
명시했다. Q6_K와 BF16 네이티브 경로는 수정하지 않았다.

독립적인 `gguf.quants.dequantize`와 Q4_K/Q5_K 결과를 비교했다. 여러 유한한
스케일, 0과 음수, 임의 packed 비트를 포함하며 FP32 결과 및 FP16/BF16 변환이
일치하고 원본 bytes가 변하지 않는지 검사한다. 반복 호출도 같은 결과를 만든다.

재현 자료는 `validation/gemma-dequant-buffer-20261104/`에 보관한다.
`tools/benchmark_gemma_inference.py --baseline-dequant`는 baseline ops에서만
수정 직전 decoder를 사용해 디코더 변경 효과를 분리한다. 두 모드 모두 앞서
적용한 LoRA 출력 계산, slice 최적화, PLE CPU 보관을 활성화한다.

### 실제 가중치 하나의 임시 할당

RTX 5060 Ti에서 실제 `blk.15.ffn_gate.weight`(12288 × 1536, Q4_K)를 FP32로
풀었다. 출력 자체는 72 MiB다. GPU packed 입력을 올려둔 시점 대비 최대 추가
allocated가 167.90625→95.90625 MiB로 72 MiB 감소했다. 각 모드 3회 준비 후
20회 연속 decode를 재고, before/after 순서를 바꿔 4회 비교했다.
변경 전후 FP32 텐서는 `torch.equal`로 일치했다. 이는 가중치 한 개를 푸는 동안의
임시 메모리이며 모델 전체의 최대 VRAM이 72 MiB 줄어든다는 뜻은 아니다.

### 전체 생성 비교와 자원관리

RTX 5060 Ti에서 동일 Q4 모델, `scale-20pct-001`, temperature 0으로 2·6·10인을
각각 before/after 1회 비교했다. 입력별 4토큰 warmup 후 전체 `run_scene()`을
측정했다. PLE는 두 모드 모두 CPU에 두므로 아래 차이는 이번 버퍼 재사용만의
비교다. 모든 생성 token ID, 공통/인물 슬롯 및 외형·의상·소품 회귀 검사가 일치했다.

| 입력 | before | after | 최대 allocated before / after | 추가 절감 |
| --- | ---: | ---: | ---: | ---: |
| 2인 | 26.355초 | 26.182초 | 1761.90 / 1729.00 MiB | 32.91 MiB |
| 6인 | 62.267초 | 59.781초 | 1806.17 / 1773.26 MiB | 32.91 MiB |
| 10인 | 94.998초 | 96.725초 | 1853.29 / 1820.30 MiB | 32.99 MiB |

전체 생성의 최고점은 다른 연산과 겹치는 시점으로 결정되므로, 가중치 하나의
역양자화에서 줄인 72 MiB와 전체 최고점 감소량은 다르다. 시간은 약 -0.7% /
-4.0% / +1.8%로 양방향 변화했다. 고정 순서 각 1회의 측정이므로 속도 향상을
주장하지 않으며, 이번에 확인한 효과는 최대 allocated 약 33 MiB 추가 절감이다.

RTX 4070 Ti SUPER에서는 temperature=0.3, seed=42의 2인 입력으로
before→after→after→before 4회 비교했다. 모든 token ID와 슬롯이 일치했고,
최대 allocated는 1738.38→1705.47 MiB로 32.91 MiB 줄었다. before 평균
16.799초, after 평균 16.388초였다. 다른 입력·시드의 전체 샘플링 검증은 아니다.

5060 Ti에서 ComfyUI unload→reload, 생성 취소와 샘플러 복원, 취소 후 재생성,
금지 태그가 있는 기존 경로까지 통과했다. 마지막 unload 후 CUDA allocated는
32 MiB, reserved 54 MiB, ComfyUI 로드 모델은 0개였다. 4070 Ti SUPER 비교 후
unload에서도 allocated 8.125 MiB와 로드 모델 0개를 확인했다. PLE CPU 보관과
앞서 적용한 LoRA 출력 계산을 그대로 유지하며 추가 상주 캐시는 없다.

Python 전체 221개 중 214개 통과, 선택 의존성 관련 7개 skip. JavaScript 4개
통과, `git diff --check` 통과. 이 변경은
Q4_K/Q5_K decoder에 한정되며, 실제 모델 전체 생성 비교는 Q4 모델로 수행했다.
새로운 BF16 전체 비교나 이미지 생성은 실행하지 않았다. 코드와 기록은 로컬
적용 상태이며 실행 중인 ComfyUI에는 노드 재로딩 또는 서버 재시작 후 적용된다.
