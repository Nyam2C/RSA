#!/usr/bin/env python3
"""
Recursive Self-Aggregation (RSA) for Claude Code
=================================================
논문: "Recursive Self-Aggregation Unlocks Deep Thinking in Large Language Models"
(Venkatraman et al., 2025)

Claude Code에서 실행하여 서비스 리뷰, 계획 수립 등에 활용합니다.

사용법:
    python rsa.py --task "당신의 작업 설명" --config config.yaml
    python rsa.py --task-file task.md --config config.yaml
"""

import random
import json
import os
import argparse
import time
import subprocess
import shutil
from pathlib import Path
from typing import List, Optional
from datetime import datetime


# ============================================================
# 기본 설정
# ============================================================

DEFAULT_CONFIG = {
    # RSA 핵심 파라미터
    "N": 5,           # Population size (초기 생성 수)
    "K": 3,           # Aggregation subset size (한번에 묶을 수)
    "T": 3,           # Recursive steps (라운드 수)
    "final_mode": "random",  # 최종 선택: "random" (논문) 또는 "aggregate"

    # 모델 설정
    "model": "opus",              # CLI alias (opus, sonnet, haiku)
    "aggregation_model": "opus",  # 종합용 (동일 또는 상위 모델)

    # Claude Code CLI 설정
    "claude_path": "claude",      # CLI 실행 경로
    "max_budget_usd": None,       # 최대 비용 제한 (None=무제한)
    "disable_tools": True,        # 도구 비활성화 (계획/분석만 생성)
    "timeout": 1800,              # CLI 호출 타임아웃 (초, 30분)

    # 출력 설정
    "output_dir": "./rsa_output",
    "save_intermediate": True,    # 중간 결과 저장 여부
    "verbose": True,
}


# ============================================================
# Aggregation 프롬프트 (논문 Appendix F 기반)
# ============================================================

def build_aggregation_prompt(
    task: str,
    candidates: List[str],
    round_num: int,
    total_rounds: int
) -> str:
    """
    논문 Appendix F 기반 aggregation prompt.
    K=1이면 self-refinement, K>1이면 multi-trajectory aggregation.
    """

    if len(candidates) == 1:
        # Single-trajectory refinement (K=1)
        return f"""당신은 주어진 작업과 후보 솔루션을 받았습니다.
후보는 불완전하거나 오류를 포함할 수 있습니다.
이 솔루션을 개선하여 더 높은 품질의 솔루션을 작성하세요.
완전히 틀렸다면 새로운 전략을 시도하세요.

## 작업
{task}

## 후보 솔루션
{candidates[0]}"""

    else:
        # Multi-trajectory aggregation (논문 Appendix F)
        candidates_text = ""
        for i, c in enumerate(candidates, 1):
            candidates_text += f"\n{'='*60}\n후보 솔루션 {i}\n{'='*60}\n{c}\n"

        return f"""당신은 주어진 작업과 여러 후보 솔루션을 받았습니다.
일부 후보는 틀리거나 오류를 포함할 수 있습니다.
유용한 아이디어를 종합하여 단일 고품질 솔루션을 작성하세요.
후보들이 서로 다르다면 신중하게 올바른 경로를 선택하세요.
모든 후보가 틀렸다면 새로운 전략을 시도하세요.
주의: 서로 다른 전략의 장점만 무분별하게 합치지 마세요. 일관된 하나의 방향을 선택하세요.

## 작업
{task}

## 후보 솔루션들
{candidates_text}"""


# ============================================================
# RSA 핵심 로직
# ============================================================

class RSA:
    def __init__(self, config: dict):
        self.config = {**DEFAULT_CONFIG, **config}

        # Claude Code CLI 검증
        self.claude_path = self.config["claude_path"]
        self._validate_cli()

        # 중첩 실행 방지: CLAUDE_CODE 관련 환경변수 제거
        self.env = {k: v for k, v in os.environ.items()
                    if not k.startswith("CLAUDECODE")}

        # 비용 추적
        self.total_cost_usd = 0.0

        self.output_dir = Path(self.config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 실행 ID (타임스탬프 기반)
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def _validate_cli(self):
        """Claude Code CLI가 설치되어 있는지 검증"""
        if not shutil.which(self.claude_path):
            print(f"❌ Claude Code CLI를 찾을 수 없습니다: '{self.claude_path}'")
            print("   설치: https://docs.anthropic.com/en/docs/claude-code")
            raise SystemExit(1)

    def log(self, msg: str):
        if self.config["verbose"]:
            print(f"[RSA] {msg}")

    def _build_cmd(self, model: Optional[str] = None) -> List[str]:
        """CLI 명령어 구성"""
        model = model or self.config["model"]
        cmd = [
            self.claude_path, "-p",
            "--output-format", "json",
            "--no-session-persistence",
            "--model", model,
        ]
        if self.config.get("disable_tools", True):
            cmd.extend(["--allowedTools", ""])
        cmd.extend([
            "--system-prompt",
            "당신은 계획 수립 전문가입니다. 분석과 계획만 텍스트로 작성하세요. "
            "코드를 실행하거나 파일을 수정하지 마세요."
        ])
        if self.config.get("max_budget_usd"):
            cmd.extend(["--max-turns-budget",
                        str(self.config["max_budget_usd"])])
        return cmd

    def _parse_response(self, stdout: str) -> str:
        """CLI JSON 응답 파싱 및 비용 추적"""
        response = json.loads(stdout)
        if response.get("is_error"):
            raise RuntimeError(
                "CLI 응답 오류: {}".format(response.get("result", "unknown")))
        cost = response.get("cost_usd") or response.get("total_cost_usd", 0)
        self.total_cost_usd += cost
        self.log("    비용: ${:.4f} (누적: ${:.4f})".format(
            cost, self.total_cost_usd))
        return response["result"]

    def call_llm(self, prompt: str, model: Optional[str] = None) -> str:
        """Claude Code CLI를 통한 단일 LLM 호출"""
        max_retries = 3

        for attempt in range(max_retries):
            try:
                cmd = self._build_cmd(model)
                result = subprocess.run(
                    cmd,
                    input=prompt,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                    env=self.env,
                    timeout=self.config.get("timeout", 600),
                )
                if result.returncode != 0:
                    raise RuntimeError(
                        "CLI 오류 (exit {}): {}".format(
                            result.returncode, result.stderr.strip()))
                return self._parse_response(result.stdout)

            except subprocess.TimeoutExpired:
                self.log("타임아웃 ({}초). 재시도... ({}/{})".format(
                    self.config.get("timeout", 600),
                    attempt + 1, max_retries))
                time.sleep(5)
            except json.JSONDecodeError as e:
                self.log("JSON 파싱 오류: {}".format(e))
                if attempt == max_retries - 1:
                    raise
                time.sleep(5)
            except RuntimeError as e:
                if "rate" in str(e).lower() or "429" in str(e):
                    wait = 2 ** attempt * 10
                    self.log("Rate limit. {}초 대기 후 재시도... ({}/{})".format(
                        wait, attempt + 1, max_retries))
                    time.sleep(wait)
                else:
                    self.log("오류: {}".format(e))
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(5)

        return ""

    def call_llm_batch(
        self, prompts: List[str], model: Optional[str] = None
    ) -> List[str]:
        """여러 프롬프트를 순차 실행 (권한 충돌 방지)"""
        n = len(prompts)
        self.log("  순차 실행 시작: {}개".format(n))

        results = []
        for i, prompt in enumerate(prompts):
            self.log("  [{}/{}] 실행 중...".format(i + 1, n))
            text = self.call_llm(prompt, model)
            preview_lines = text.strip().splitlines()[:3]
            preview = "\n".join("    │ " + l for l in preview_lines)
            self.log("  [{}/{}] 완료\n{}".format(i + 1, n, preview))
            results.append(text)

        return results

    def save_population(self, population: List[str], step: int, label: str = ""):
        """인구(population)를 파일로 저장"""
        if not self.config["save_intermediate"]:
            return

        step_dir = self.run_dir / f"round_{step:02d}{f'_{label}' if label else ''}"
        step_dir.mkdir(parents=True, exist_ok=True)

        for i, solution in enumerate(population):
            filepath = step_dir / f"solution_{i+1:02d}.md"
            filepath.write_text(solution, encoding="utf-8")

        self.log(f"  → {step_dir}에 {len(population)}개 저장")

    # ----- Step 1: 초기 Population 생성 -----
    def generate_initial_population(self, task: str) -> List[str]:
        """N개의 독립적인 초기 솔루션 순차 생성"""
        N = self.config["N"]
        self.log("\n{}".format("=" * 60))
        self.log("Step 1: 초기 population 순차 생성 (N={})".format(N))
        self.log("{}".format("=" * 60))

        prompts = []
        for i in range(N):
            prompt = """다음 작업에 대해 당신만의 고유한 접근 방식으로 완성도 높은 솔루션을 작성하세요.
다른 관점이나 전략을 자유롭게 탐색하세요.

## 작업
{}

독창적이고 체계적인 솔루션을 작성하세요.""".format(task)
            prompts.append(prompt)

        population = self.call_llm_batch(prompts)
        self.save_population(population, 0, "initial")
        return population

    # ----- Step 2: 서브셋 샘플링 -----
    def subsample(self, population: List[str]) -> List[List[str]]:
        """
        N개 population에서 K개씩 랜덤 서브셋을 N개 생성.
        논문: uniform sampling without replacement.
        """
        N = len(population)
        K = self.config["K"]
        aggregation_sets = []

        for _ in range(N):
            indices = random.sample(range(N), K)
            subset = [population[idx] for idx in indices]
            aggregation_sets.append(subset)

        return aggregation_sets

    # ----- Step 3: Aggregation -----
    def aggregate(
        self,
        task: str,
        aggregation_sets: List[List[str]],
        round_num: int
    ) -> List[str]:
        """각 서브셋을 병렬로 종합하여 새로운 population 생성"""
        T = self.config["T"]

        prompts = []
        for subset in aggregation_sets:
            prompt = build_aggregation_prompt(
                task=task,
                candidates=subset,
                round_num=round_num,
                total_rounds=T
            )
            prompts.append(prompt)

        return self.call_llm_batch(prompts, model=self.config["aggregation_model"])

    # ----- 메인 실행 -----
    def run(self, task: str) -> str:
        """RSA 전체 파이프라인 실행"""
        N = self.config["N"]
        K = self.config["K"]
        T = self.config["T"]

        self.log(f"\n{'#'*60}")
        self.log(f"RSA 실행 시작")
        self.log(f"  N={N} (population), K={K} (subset), T={T} (rounds)")
        self.log(f"  모델: {self.config['model']}")
        self.log(f"  종합 모델: {self.config['aggregation_model']}")
        self.log(f"  출력: {self.run_dir}")
        self.log(f"{'#'*60}")

        # 설정 저장
        config_path = self.run_dir / "config.json"
        config_path.write_text(
            json.dumps(self.config, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

        # 작업 저장
        task_path = self.run_dir / "task.md"
        task_path.write_text(task, encoding="utf-8")

        # Step 1: 초기 population
        population = self.generate_initial_population(task)

        # Step 2~T: Recursive aggregation
        for t in range(1, T + 1):
            self.log(f"\n{'='*60}")
            self.log(f"Round {t}/{T}: Aggregation (K={K}씩 묶어서 종합)")
            self.log(f"{'='*60}")

            # 서브셋 샘플링
            aggregation_sets = self.subsample(population)

            # 종합
            population = self.aggregate(task, aggregation_sets, t)

            # 중간 결과 저장
            self.save_population(population, t)

        # Step 4: 최종 결과 선택
        final_mode = self.config.get("final_mode", "random")
        self.log(f"\n{'='*60}")

        if final_mode == "aggregate":
            # 전체 종합 방식 (K=N, 논문 §5.3에서 K>4 성능 저하 경고)
            self.log(f"최종 종합: {N}개 → 1개 (aggregate 모드)")
            self.log(f"{'='*60}")
            final_prompt = build_aggregation_prompt(
                task=task,
                candidates=population,
                round_num=T,
                total_rounds=T
            )
            final_result = self.call_llm(
                final_prompt,
                model=self.config["aggregation_model"]
            )
        else:
            # 논문 방식: uniform random sampling (Algorithm 1, step 4)
            self.log(f"최종 선택: {N}개 중 랜덤 1개 (random 모드)")
            self.log(f"{'='*60}")
            final_result = random.choice(population)

        # 최종 결과 저장
        final_path = self.run_dir / "FINAL_RESULT.md"
        final_path.write_text(final_result, encoding="utf-8")
        self.log(f"\n최종 결과 저장: {final_path}")

        # 실행 요약
        summary = self._generate_summary(task, population, final_result)
        summary_path = self.run_dir / "SUMMARY.md"
        summary_path.write_text(summary, encoding="utf-8")

        self.log(f"\n{'#'*60}")
        self.log(f"RSA 완료! 결과: {self.run_dir}")
        self.log(f"{'#'*60}\n")

        return final_result

    def _generate_summary(
        self, task: str, final_pop: List[str], final: str
    ) -> str:
        """실행 요약 생성"""
        return f"""# RSA 실행 요약

## 설정
- **Population (N):** {self.config['N']}
- **Subset size (K):** {self.config['K']}
- **Rounds (T):** {self.config['T']}
- **모델:** {self.config['model']}
- **종합 모델:** {self.config['aggregation_model']}
- **최종 선택:** {self.config.get('final_mode', 'random')}
- **CLI 경로:** {self.claude_path}
- **실행 시간:** {self.run_id}
- **총 비용:** ${self.total_cost_usd:.4f}

## 작업
{task[:500]}{'...' if len(task) > 500 else ''}

## 디렉토리 구조
```
{self.run_id}/
├── config.json          # 설정
├── task.md              # 원본 작업
├── round_00_initial/    # 초기 {self.config['N']}개 솔루션
│   ├── solution_01.md
│   └── ...
├── round_01/            # 1차 종합 결과
├── round_02/            # 2차 종합 결과
├── round_03/            # 3차 종합 결과
├── FINAL_RESULT.md      # 최종 결과 ★
└── SUMMARY.md           # 이 파일
```

## 참고
- 논문: "Recursive Self-Aggregation Unlocks Deep Thinking in LLMs"
- 각 라운드의 중간 결과를 비교하면 품질 향상 과정을 확인할 수 있습니다.
"""


# ============================================================
# CLI
# ============================================================

def load_config(config_path: Optional[str]) -> dict:
    """설정 파일 로드 (없으면 기본값)"""
    if config_path and Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            if config_path.endswith(".json"):
                return json.load(f)
            else:
                # YAML 지원 (선택적)
                try:
                    import yaml
                    return yaml.safe_load(f)
                except ImportError:
                    print("YAML 지원을 위해 pyyaml을 설치하세요: pip install pyyaml")
                    return {}
    return {}


def main():
    parser = argparse.ArgumentParser(
        description="RSA (Recursive Self-Aggregation) for Claude Code",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 기본 실행
  python rsa.py --task "내 SaaS 서비스의 아키텍처를 리뷰하고 개선 계획을 세워줘"

  # 파일에서 작업 읽기
  python rsa.py --task-file my_service_description.md

  # 파라미터 조정
  python rsa.py --task "..." -N 8 -K 4 -T 4

  # 모델 alias: opus, sonnet, haiku 또는 전체 모델 ID
  python rsa.py --task "..." --model sonnet --agg-model opus
        """
    )

    # 작업 입력
    task_group = parser.add_mutually_exclusive_group(required=True)
    task_group.add_argument("--task", "-t", type=str, help="작업 설명 (텍스트)")
    task_group.add_argument("--task-file", "-f", type=str, help="작업 설명 파일 (.md, .txt)")

    # RSA 파라미터
    parser.add_argument("-N", type=int, default=5, help="Population size (기본: 5)")
    parser.add_argument("-K", type=int, default=3, help="Aggregation subset size (기본: 3)")
    parser.add_argument("-T", type=int, default=3, help="Recursive rounds (기본: 3)")

    # 최종 선택 모드
    parser.add_argument("--final-mode", type=str, default=None,
                        choices=["random", "aggregate"],
                        help="최종 선택 방식: random (논문 기본) 또는 aggregate (전체 종합)")

    # 모델 설정
    parser.add_argument("--model", "-m", type=str, default=None,
                        help="생성 모델 (alias: opus, sonnet, haiku)")
    parser.add_argument("--agg-model", type=str, default=None,
                        help="종합 모델 (기본: 생성 모델과 동일)")

    # Claude Code CLI 설정
    parser.add_argument("--claude-path", type=str, default=None,
                        help="Claude Code CLI 경로 (기본: claude)")
    parser.add_argument("--max-budget-usd", type=float, default=None,
                        help="최대 비용 제한 (USD)")

    # 출력 설정
    parser.add_argument("--output", "-o", type=str, default="./rsa_output", help="출력 디렉토리")
    parser.add_argument("--config", "-c", type=str, default=None, help="설정 파일 (JSON/YAML)")
    parser.add_argument("--quiet", "-q", action="store_true", help="조용한 모드")

    args = parser.parse_args()

    # 설정 구성
    config = load_config(args.config)

    # CLI 인자가 설정 파일보다 우선
    if args.N != 5 or "N" not in config:
        config["N"] = args.N
    if args.K != 3 or "K" not in config:
        config["K"] = args.K
    if args.T != 3 or "T" not in config:
        config["T"] = args.T
    if args.final_mode:
        config["final_mode"] = args.final_mode
    if args.model:
        config["model"] = args.model
    if args.agg_model:
        config["aggregation_model"] = args.agg_model
    if args.claude_path:
        config["claude_path"] = args.claude_path
    if args.max_budget_usd is not None:
        config["max_budget_usd"] = args.max_budget_usd

    config["output_dir"] = args.output
    config["verbose"] = not args.quiet

    # 작업 로드
    if args.task_file:
        task = Path(args.task_file).read_text(encoding="utf-8")
    else:
        task = args.task

    # 유효성 검사
    assert config.get("K", 3) <= config.get("N", 6), \
        f"K({config['K']})는 N({config['N']}) 이하여야 합니다."
    assert config.get("T", 3) >= 1, "T는 1 이상이어야 합니다."

    # 실행
    rsa = RSA(config)
    result = rsa.run(task)

    # 최종 결과 출력
    if not args.quiet:
        print(f"\n{'='*60}")
        print("최종 결과 (FINAL_RESULT.md)")
        print(f"{'='*60}")
        print(result[:2000])
        if len(result) > 2000:
            print(f"\n... ({len(result)}자 중 처음 2000자만 표시)")


if __name__ == "__main__":
    main()
