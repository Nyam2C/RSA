# RSA (Recursive Self-Aggregation) for Claude Code

논문 ["Recursive Self-Aggregation Unlocks Deep Thinking in Large Language Models"](https://arxiv.org/abs/2507.xxxxx)의
Algorithm 1 (Appendix B)을 Claude Code CLI 환경에서 바로 사용할 수 있도록 구현한 스크립트입니다.

## 핵심 원리

```
Round 0: N개 독립 솔루션 생성 (다양성 확보)
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

- **Python 3.8+**
- **Claude Code CLI** 설치 및 인증 완료

> `pip install anthropic`이나 `ANTHROPIC_API_KEY` 설정은 필요 없습니다.
> Claude Code CLI가 인증을 처리합니다.

## 빠른 시작

```bash
# 1. Claude Code CLI 설치 (아직 안 했다면)
#    https://docs.anthropic.com/en/docs/claude-code

# 2. 실행
python rsa.py --task "내 투자 AI 어시스턴트 서비스의 아키텍처를 리뷰하고 개선 계획을 세워줘"

# 3. 파일에서 작업 읽기
python rsa.py --task-file example_task.md
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

### 파라미터 조정
```bash
# 가볍게 (빠른 리뷰)
python rsa.py --task "..." -N 4 -K 2 -T 2

# 기본 (대부분의 작업에 추천)
python rsa.py --task "..." -N 5 -K 3 -T 3

# 풀파워 (중요한 의사결정)
python rsa.py --task "..." -N 8 -K 4 -T 4
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
| **N** (population) | 초기 후보 수 | **5** | N이 K의 2~4배일 때 최적 |
| **K** (subset) | 한번에 종합할 수 | **3** | K=3~4이 sweet spot (§5.3) |
| **T** (rounds) | 반복 횟수 | **3** | 단조 증가하지만 수확체감. 3~5가 실용적 |
| **final_mode** | 최종 선택 방식 | **random** | 논문 Algorithm 1: uniform random sampling |

### 파라미터 간 관계 (핵심!)
- **N이 크면** → K 또는 T도 키워야 함 (큰 집단이 섞이려면 더 많은 라운드 필요)
- **T를 늘리기 어려우면** → N을 줄여라 (작은 집단이 빠르게 수렴)
- **K > 4는 비추** → 컨텍스트가 너무 길어져서 모델 성능 저하

## 비용 추정

N=5, K=3, T=3, final_mode=random 기준:

| 단계 | CLI 호출 수 | 설명 |
|------|-----------|------|
| 초기 생성 | 5 | N개 독립 솔루션 |
| Round 1~3 | 5 × 3 = 15 | 각 라운드 N개 종합 |
| 최종 선택 | 0 | 랜덤 선택 (LLM 호출 없음) |
| **합계** | **20** | |

> `--final-mode aggregate` 시 최종 종합 1회 추가 (합계 21회).
> 실제 비용은 각 호출 후 로그에 표시되며, SUMMARY.md에 총 비용이 기록됩니다.

## 출력 구조

```
rsa_output/
└── 20250213_143022/           # 실행 타임스탬프
    ├── config.json            # 사용된 설정
    ├── task.md                # 원본 작업
    ├── round_00_initial/      # 초기 N개 솔루션
    │   ├── solution_01.md
    │   ├── solution_02.md
    │   └── ...
    ├── round_01/              # 1차 종합 결과
    ├── round_02/              # 2차 종합 결과
    ├── round_03/              # 3차 종합 결과
    ├── FINAL_RESULT.md        # ★ 최종 결과
    └── SUMMARY.md             # 실행 요약 (비용 포함)
```

중간 결과를 비교하면 품질이 라운드마다 향상되는 과정을 확인할 수 있습니다.

## CLI 옵션 전체

```bash
python rsa.py --help

# 주요 옵션
-t, --task TEXT          # 작업 설명 (텍스트)
-f, --task-file PATH     # 작업 설명 파일 (.md, .txt)
-N INT                   # Population size (기본: 5)
-K INT                   # Subset size (기본: 3)
-T INT                   # Rounds (기본: 3)
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

| 용도 | N | K | T | final_mode | 모델 | 비고 |
|------|---|---|---|------------|------|------|
| 빠른 피드백 | 4 | 2 | 2 | random | sonnet | 빠르고 경제적 |
| 일반 리뷰 | 5 | 3 | 3 | random | opus | 기본값, 고품질 |
| 중요한 계획 | 8 | 4 | 3 | random | opus | 더 많은 다양성 |
| 최고 품질 | 8 | 4 | 4 | random | opus | 최대 라운드 |

## 작업 파일 예시

`example_task.md` 참고. 작업 파일 작성 형식:

```markdown
# 프로젝트 리뷰 요청

## 목표
- 구체적인 작업 목표 기술

## 현재 상태
- 기술 스택, 현 상황 설명

## 요청 사항
1. 분석/리뷰 포인트
2. 개선 방향

## 제약 조건
- 예산, 인력, 기한 등
```

## 설정 파일 예시 (config.json)

```json
{
    "N": 5,
    "K": 3,
    "T": 3,
    "final_mode": "random",
    "model": "opus",
    "aggregation_model": "opus",
    "claude_path": "claude",
    "disable_tools": true,
    "output_dir": "./rsa_output",
    "save_intermediate": true,
    "verbose": true
}
```

## 설계 노트

- **Aggregation 프롬프트**: 논문 Appendix F 원문 스타일을 따름. 과도한 프롬프트 엔지니어링 배제 ("We avoided [heavy prompt engineering] to prevent skewing results")
- **최종 선택**: 논문 Algorithm 1 step 4의 uniform random sampling이 기본. `--final-mode aggregate`로 전체 종합도 가능
- **도구 비활성화**: `--allowedTools ""`로 Claude Code의 모든 도구(Bash, Edit 등)를 비활성화하여 텍스트 생성만 수행
- **시스템 프롬프트**: 계획/분석 전문가로 역할 제한
- **stdin 프롬프트**: aggregation 프롬프트가 수만 자에 달할 수 있어 stdin으로 전달
- **중첩 방지**: `CLAUDECODE` 환경변수를 제거하여 Claude Code 내부 실행 시 중첩 에러 방지
- **비용 추적**: 각 CLI 호출의 비용을 누적 추적하여 SUMMARY.md에 기록

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
