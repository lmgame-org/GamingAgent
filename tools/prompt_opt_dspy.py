#!/usr/bin/env python3
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

# Optional DSPy import with friendly error
try:
    import dspy
except Exception as e:
    dspy = None

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = WORKSPACE_ROOT / "lmgame-bench" / "multi_agent_runner.py"
DEFAULT_BASE_PROMPT = WORKSPACE_ROOT / "gamingagent" / "configs" / "zoo_02_texasholdem" / "module_prompts.json"

# -----------------------------
# Utilities
# -----------------------------

def read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def get_from_dotted(cfg: Dict[str, Any], dotted: str) -> Any:
    cur = cfg
    for part in dotted.split("."):
        cur = cur[part]
    return cur

def set_in_dotted(cfg: Dict[str, Any], dotted: str, value: Any) -> None:
    cur = cfg
    parts = dotted.split(".")
    for p in parts[:-1]:
        cur = cur[p]
    cur[parts[-1]] = value

# -----------------------------
# DSPy rewriting
# -----------------------------

def configure_dspy(model_name: str, temperature: float, max_tokens: int, top_p: float, seed: Optional[int]) -> None:
    if dspy is None:
        raise RuntimeError("DSPy is not installed. Try: pip install dspy")
    kwargs = {"temperature": temperature, "max_tokens": max_tokens, "top_p": top_p}
    if seed is not None:
        kwargs["seed"] = int(seed)
    try:
        lm = dspy.OpenAI(model=model_name, **kwargs)
    except Exception:
        lm = dspy.LM(model=model_name, **kwargs)
    dspy.settings.configure(lm=lm)

class RewriterSignature(dspy.Signature):
    """Rewrite a prompt while preserving variables like {textual_representation}."""
    original = dspy.InputField()
    constraints = dspy.InputField()
    rewrite = dspy.OutputField()


def generate_rewrites_with_dspy(base_text: str, num_variants: int, constraints: str) -> List[str]:
    rewrites: List[str] = []
    seen = set()
    for i in range(num_variants):
        # Recreate program each time to encourage fresh sampling
        program = dspy.ChainOfThought(RewriterSignature)
        try:
            uniqueness_hint = (
                f"\nVARIANT_ID={i+1}. Produce a distinct paraphrase that differs in wording and emphasis. "
                f"Avoid repeating any prior variant content."
            )
            out = program(original=base_text, constraints=constraints + uniqueness_hint)
            cand = str(out.rewrite).strip()
            if cand and cand not in seen:
                rewrites.append(cand)
                seen.add(cand)
        except Exception:
            continue
    return rewrites

# -----------------------------
# Runner integration
# -----------------------------

FINAL_STANDINGS_HEADER = "Final chip standings:"
STANDING_LINE_RE = re.compile(r"^\s*#\d+:\s+(player_\d+)\s+-\s+(\d+)\s+chips\s*$")


def run_tournament(model: str, prompt_a: Path, prompt_b: Path, hands: int, extra_args: List[str]) -> Tuple[Dict[str, int], str]:
    """
    Run one tournament: player_0 uses prompt_a, player_1 uses prompt_b.
    Returns (final_chips_by_player, raw_stdout).
    """
    cmd = [
        sys.executable, str(RUNNER_PATH),
        "--game_name", "texasholdem",
        "--player_models", model, model,
        "--prompt_path_p0", str(prompt_a),
        "--prompt_path_p1", str(prompt_b),
        "--tournament_hands", str(hands),
        "--record_video", "false",
    ] + extra_args
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = proc.stdout
    chips: Dict[str, int] = {}
    if FINAL_STANDINGS_HEADER in out:
        grabbing = False
        for line in out.splitlines():
            if FINAL_STANDINGS_HEADER in line:
                grabbing = True
                continue
            if grabbing:
                m = STANDING_LINE_RE.match(line)
                if m:
                    player, chips_str = m.group(1), m.group(2)
                    chips[player] = int(chips_str)
                elif line.strip() == "":
                    break
    return chips, out


def eval_candidate_vs_baseline(model: str, candidate: Path, baseline: Path, hands: int, extra_args: List[str]) -> Dict[str, Any]:
    """
    Run two tournaments (A vs B and B vs A) and compute candidate advantage.
    Returns metrics including average delta and per-run chips.
    """
    chips1, out1 = run_tournament(model, candidate, baseline, hands, extra_args)
    chips2, out2 = run_tournament(model, baseline, candidate, hands, extra_args)

    def delta(chips: Dict[str, int], candidate_as_p0: bool) -> int:
        if candidate_as_p0:
            return int(chips.get("player_0", 0)) - int(chips.get("player_1", 0))
        else:
            return int(chips.get("player_1", 0)) - int(chips.get("player_0", 0))

    d1 = delta(chips1, candidate_as_p0=True)
    d2 = delta(chips2, candidate_as_p0=False)
    avg = (d1 + d2) / 2.0
    return {
        "candidate": str(candidate),
        "baseline": str(baseline),
        "hands": hands,
        "delta_run_1": d1,
        "delta_run_2": d2,
        "avg_delta": avg,
        "chips_run_1": chips1,
        "chips_run_2": chips2,
        "stdout_run_1": out1,
        "stdout_run_2": out2,
    }

# -----------------------------
# Main optimization loop
# -----------------------------

def main():
    ap = argparse.ArgumentParser("DSPy Prompt Optimizer for Texas Hold'em")
    ap.add_argument("--base_prompt", type=str, default=str(DEFAULT_BASE_PROMPT), help="Path to base prompt JSON")
    ap.add_argument("--section", type=str, default="reasoning_module.prompt", help="Dotted path to rewrite (e.g., reasoning_module.prompt)")
    ap.add_argument("--model", type=str, default="kimi-k2-0711-preview", help="Model name used in the runner")
    ap.add_argument("--dspy_model", type=str, default="gpt-4o-mini", help="DSPy LM model for rewriting prompts")
    ap.add_argument("--dspy_temperature", type=float, default=1.0, help="Temperature for DSPy LM (reasoning models may require 1.0)")
    ap.add_argument("--dspy_max_tokens", type=int, default=16000, help="Max tokens for DSPy LM (reasoning models may require >=16000)")
    ap.add_argument("--dspy_top_p", type=float, default=1.0, help="Top-p nucleus sampling for DSPy LM")
    ap.add_argument("--dspy_seed", type=int, default=None, help="Optional seed for DSPy LM (if backend supports it)")
    ap.add_argument("--hands", type=int, default=10, help="Tournament hands per evaluation")
    ap.add_argument("--num_variants", type=int, default=5, help="Number of candidate rewrites to generate")
    ap.add_argument("--generations", type=int, default=1, help="Generations of rewrite-selection")
    ap.add_argument("--keep_top_k", type=int, default=2, help="Top-k candidates to keep per generation")
    ap.add_argument("--work_dir", type=str, default=str(WORKSPACE_ROOT / "prompt_runs"), help="Where to store generated prompts and results")
    ap.add_argument("--extra_runner_args", type=str, nargs="*", default=[], help="Extra args forwarded to the runner")
    args = ap.parse_args()

    base_path = Path(args.base_prompt).resolve()
    work_root = Path(args.work_dir).resolve()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = work_root / f"dspy_opt_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    base = read_json(base_path)
    try:
        base_text = get_from_dotted(base, args.section)
    except Exception as e:
        print(f"ERROR: Could not read section '{args.section}' from {base_path}: {e}")
        sys.exit(1)

    # Configure DSPy
    try:
        configure_dspy(
            args.dspy_model,
            temperature=args.dspy_temperature,
            max_tokens=args.dspy_max_tokens,
            top_p=args.dspy_top_p,
            seed=args.dspy_seed,
        )
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    constraints = (
        "Preserve all JSON placeholders (e.g., {textual_representation}, {game_trajectory}, {reflection}). "
        "Keep output concise and within the same intent. Do not remove required output formats. "
        "Do not change action legality instructions or response format keys (thought:, action:)."
    )

    # Generation loop
    baseline_path = run_dir / "baseline.json"
    write_json(base, baseline_path)
    current_pool: List[Tuple[Path, float]] = [(baseline_path, 0.0)]

    for gen in range(args.generations):
        print(f"\n=== Generation {gen+1}/{args.generations} ===")
        # Generate candidates from the best current (first item)
        seed_path = current_pool[0][0]
        seed_cfg = read_json(seed_path)
        seed_text = get_from_dotted(seed_cfg, args.section)
        variants = generate_rewrites_with_dspy(seed_text, args.num_variants, constraints)
        if not variants:
            print("No variants generated. Stopping.")
            break
        candidates: List[Path] = []
        for i, v in enumerate(variants):
            cand_cfg = json.loads(json.dumps(seed_cfg))
            set_in_dotted(cand_cfg, args.section, v)
            cand_path = run_dir / f"gen{gen+1}_cand{i+1}.json"
            write_json(cand_cfg, cand_path)
            candidates.append(cand_path)

        # Evaluate each candidate against baseline
        results: List[Tuple[float, Path, Dict[str, Any]]] = []
        for cand in candidates:
            print(f"Evaluating {cand.name} vs baseline ({args.hands} hands × 2 seats)...")
            r = eval_candidate_vs_baseline(
                model=args.model,
                candidate=cand,
                baseline=baseline_path,
                hands=args.hands,
                extra_args=args.extra_runner_args,
            )
            write_json(r, run_dir / f"result_{cand.stem}.json")
            results.append((float(r["avg_delta"]), cand, r))

        # Select top-k (higher avg_delta is better)
        results.sort(key=lambda x: x[0], reverse=True)
        top = results[: max(1, args.keep_top_k)]
        print("\nTop candidates this generation:")
        for rank, (score, path, _) in enumerate(top, 1):
            print(f"  #{rank}: {path.name}  avg_delta={score:+.2f}")
        current_pool = [(path, score) for (score, path, _) in top]
        # Update baseline to the best candidate for next generation
        best_path = current_pool[0][0]
        if best_path != baseline_path:
            shutil.copyfile(best_path, baseline_path)

    print("\n=== Finished ===")
    print(f"Results and prompts saved under: {run_dir}")
    print(f"Best prompt: {current_pool[0][0]}  (avg_delta={current_pool[0][1]:+.2f})")


if __name__ == "__main__":
    main() 