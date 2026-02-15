# RSA (Recursive Self-Aggregation) for Claude Code

논문 ["Recursive Self-Aggregation Unlocks Deep Thinking in Large Language Models"](https://arxiv.org/abs/2509.26626)의
Algorithm 1 (Appendix B)을 Claude Code CLI 환경에서 바로 사용할 수 있도록 구현한 스크립트입니다.

## 핵심 원리

```
Step 0: 코드베이스 사전 분석 (--project-dir 지정 시)
          ↓
Round 0: N개 독립 솔루션 병렬 생성 (다양성 확보)
          ↓
Round 1: K개씩 랜덤 묶어서 → N개 개선된 솔루션
          ↓
  ...     (T회 반복)
          ↓
Round T: K개씩 랜덤 묶어서 → N개 수렴된 솔루션
          ↓
Final:   N개 중 랜덤 1개 선택 (논문 기본)
```

> 논문 §5.3: K>4는 컨텍스트 과부하로 성능 저하. 최종 단계에서 N개를 한번에 종합하는 대신
> **uniform random sampling**으로 선택하는 것이 Algorithm 1의 원래 방식입니다.

## 필요 조건

- **Python 3.6+**
- **Claude Code CLI** 설치 및 인증 완료

> `pip install anthropic`이나 `ANTHROPIC_API_KEY` 설정은 필요 없습니다.
> Claude Code CLI가 인증을 처리합니다.

## 빠른 시작

```bash
# 1. Claude Code CLI 설치 (아직 안 했다면)
#    https://docs.anthropic.com/en/docs/claude-code

# 2. 기본 실행
python rsa.py --task "내 투자 AI 어시스턴트 서비스의 아키텍처를 리뷰하고 개선 계획을 세워줘"

# 3. 파일에서 작업 읽기
python rsa.py --task-file example_task.md

# 4. 프로젝트 코드베이스 자동 분석 포함
python rsa.py --task-file example_task.md -P /path/to/project
```

## 사용법

### 기본 실행
```bash
python rsa.py --task "작업 설명"
```

### 파일에서 작업 읽기 (긴 설명일 때 추천)
```bash
python rsa.py --task-file example_task.md
```

### 프로젝트 분석 포함 (추천)
```bash
# -P로 프로젝트 경로 지정 → Step 0에서 자동 분석 → RSA에 컨텍스트 전달
python rsa.py --task-file example_task.md -P /path/to/project
```

`-P`를 지정하면 RSA 시작 전에 Claude가 프로젝트 디렉토리를 읽고 분석합니다.
분석 결과가 task에 자동으로 합쳐져서 더 구체적인 솔루션이 생성됩니다.

### 파라미터 조정
```bash
# 가볍게 (빠른 리뷰)
python rsa.py --task "..." -N 4 -K 2 -T 2

# 기본 (대부분의 작업에 추천)
python rsa.py --task "..." -N 4 -K 3 -T 3

# 풀파워 (중요한 의사결정)
python rsa.py --task "..." -N 8 -K 4 -T 4
```

### 병렬 워커 수 조정
```bash
# 기본: 4개 병렬 (N=4일 때 한 배치에 전부 실행)
python rsa.py --task "..." -W 4

# 2개씩 병렬 (API rate limit이 걱정되면)
python rsa.py --task "..." -W 2
```

### 최종 선택 모드
```bash
# 논문 기본: 랜덤 선택 (Algorithm 1, step 4)
python rsa.py --task "..." --final-mode random

# 전체 종합: N개를 한번 더 모아서 1개로 (K>4 경고에 주의)
python rsa.py --task "..." --final-mode aggregate
```

### 모델 선택 전략
```bash
# 기본값: Opus로 전부 (최고 품질)
python rsa.py --task "..."

# 경제적: Sonnet으로 생성, Opus로 종합
python rsa.py --task "..." --model sonnet --agg-model opus

# 경량: Sonnet으로 전부
python rsa.py --task "..." --model sonnet
```

모델 alias: `opus`, `sonnet`, `haiku` 또는 전체 모델 ID 사용 가능

## 파라미터 가이드

논문 실험 결과에 기반한 권장값:

| 파라미터 | 역할 | 기본값 | 논문 근거 |
|---------|------|-------|----------|
| **N** (population) | 초기 후보 수 | **4** | N이 K의 2~4배일 때 최적 |
| **K** (subset) | 한번에 종합할 수 | **3** | K=3~4이 sweet spot (§5.3) |
| **T** (rounds) | 반복 횟수 | **3** | 단조 증가하지만 수확체감. 3~5가 실용적 |
| **W** (workers) | 병렬 워커 수 | **4** | N과 동일하게 설정하면 최대 병렬화 |
| **final_mode** | 최종 선택 방식 | **random** | 논문 Algorithm 1: uniform random sampling |

### 파라미터 간 관계 (핵심!)
- **N이 크면** → K 또는 T도 키워야 함 (큰 집단이 섞이려면 더 많은 라운드 필요)
- **T를 늘리기 어려우면** → N을 줄여라 (작은 집단이 빠르게 수렴)
- **K > 4는 비추** → 컨텍스트가 너무 길어져서 모델 성능 저하

## 비용 추정

N=4, K=3, T=3, final_mode=random 기준:

| 단계 | CLI 호출 수 | 설명 |
|------|-----------|------|
| 사전 분석 | 0~1 | `--project-dir` 지정 시 1회 |
| 초기 생성 | 4 | N개 독립 솔루션 (병렬) |
| Round 1~3 | 4 × 3 = 12 | 각 라운드 N개 종합 (병렬) |
| 최종 선택 | 0 | 랜덤 선택 (LLM 호출 없음) |
| **합계** | **16~17** | |

> `--final-mode aggregate` 시 최종 종합 1회 추가.

## 출력 구조

```
rsa_output/
└── 20250213_143022/           # 실행 타임스탬프
    ├── config.json            # 사용된 설정
    ├── task.md                # 작업 (분석 결과 포함)
    ├── project_analysis.md    # 코드베이스 분석 결과 (-P 사용 시)
    ├── round_00_initial/      # 초기 N개 솔루션
    │   ├── solution_01.md
    │   ├── solution_02.md
    │   └── ...
    ├── round_01/              # 1차 종합 결과
    ├── round_02/              # 2차 종합 결과
    ├── round_03/              # 3차 종합 결과
    ├── FINAL_RESULT.md        # ★ 최종 결과
    └── SUMMARY.md             # 실행 요약 (소요 시간 포함)
```

중간 결과를 비교하면 품질이 라운드마다 향상되는 과정을 확인할 수 있습니다.

## CLI 옵션 전체

```bash
python rsa.py --help

# 주요 옵션
-t, --task TEXT          # 작업 설명 (텍스트)
-f, --task-file PATH     # 작업 설명 파일 (.md, .txt)
-N INT                   # Population size (기본: 4)
-K INT                   # Subset size (기본: 3)
-T INT                   # Rounds (기본: 3)
-W, --max-workers INT    # 병렬 워커 수 (기본: 4)
-P, --project-dir PATH   # 사전 분석할 프로젝트 디렉토리
--final-mode MODE        # 최종 선택: random (기본) 또는 aggregate
-m, --model MODEL        # 생성 모델 (opus, sonnet, haiku)
--agg-model MODEL        # 종합 모델
--claude-path PATH       # Claude Code CLI 경로 (기본: claude)
--max-budget-usd USD     # 최대 비용 제한
-o, --output DIR         # 출력 디렉토리 (기본: ./rsa_output)
-c, --config PATH        # 설정 파일 (JSON/YAML)
-q, --quiet              # 조용한 모드
```

## 프리셋

| 용도 | N | K | T | W | final_mode | 모델 | 비고 |
|------|---|---|---|---|------------|------|------|
| 빠른 피드백 | 4 | 2 | 2 | 4 | random | sonnet | 빠르고 경제적 |
| 일반 리뷰 | 4 | 3 | 3 | 4 | random | opus | 기본값, 고품질 |
| 중요한 계획 | 8 | 4 | 3 | 4 | random | opus | 더 많은 다양성 |
| 최고 품질 | 8 | 4 | 4 | 4 | random | opus | 최대 라운드 |

## 설정 파일 예시 (config.json)

```json
{
    "N": 4,
    "K": 3,
    "T": 3,
    "final_mode": "random",
    "model": "opus",
    "aggregation_model": "opus",
    "max_workers": 4,
    "claude_path": "claude",
    "disable_tools": true,
    "output_dir": "./rsa_output",
    "save_intermediate": true,
    "verbose": true
}
```

## 설계 노트

- **병렬 실행**: `ThreadPoolExecutor`로 N개 CLI 호출을 병렬 처리. 결과 순서 보존
- **프로세스 격리**: 각 CLI 호출은 고유 session ID + `/tmp` cwd + 환경변수 격리로 완전 독립
- **코드베이스 분석**: `--project-dir` 지정 시 RSA 전에 1회 분석하여 task에 컨텍스트 추가
- **Aggregation 프롬프트**: 논문 Appendix F 원문 스타일을 따름
- **최종 선택**: 논문 Algorithm 1 step 4의 uniform random sampling이 기본
- **도구 비활성화**: RSA 호출에서 `--allowedTools ""`로 도구를 차단하여 텍스트 생성만 수행
- **stdin 프롬프트**: aggregation 프롬프트가 수만 자에 달할 수 있어 stdin으로 전달
- **중첩 방지**: `CLAUDECODE` 환경변수를 제거하여 Claude Code 내부 실행 시 중첩 에러 방지

## 논문 참조

이 구현은 다음 논문의 알고리즘을 기반으로 합니다:

> Venkatraman, S., Jain, V., Mittal, S., et al. (2025).
> *Recursive Self-Aggregation Unlocks Deep Thinking in Large Language Models.*
> Mila – Quebec AI Institute, University of Montreal, et al.

핵심 발견:
- RSA로 Qwen3-4B가 DeepSeek-R1, o3-mini (high)급 성능 달성
- K=2만으로도 self-refinement 대비 큰 향상
- 성능은 T(라운드)에 따라 단조 증가
- K=3~4이 sweet spot, K>4는 성능 저하
- 최종 단계는 추가 aggregation 없이 random sampling
