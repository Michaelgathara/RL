#!/usr/bin/env python3

"""
Simple tester for the crossword RL environment.

Usage examples:

  # Basic smoke test and auto-solve one episode
  python3 test.py

  # Show 5 random samples and render each step while solving
  python3 test.py --samples 5 --render-steps

  # Limit environment step cap (if desired)
  python3 test.py --max-steps 200
"""

from __future__ import annotations

import argparse
import random
from typing import Any, Dict, List, Optional, Tuple

from crossword import load_environment


def _grid_dimensions(puzzle: Dict[str, Any]) -> Tuple[int, int]:
    rows = len(puzzle.get("solution", []))
    cols = len(puzzle["solution"][0]) if rows > 0 else 0
    return rows, cols


def _extract_across_answer(puzzle: Dict[str, Any], number: str = "1") -> Optional[str]:
    clues = puzzle.get("clues", {}).get("across", {})
    meta = clues.get(number)
    if not meta:
        return None
    start = meta.get("start", [0, 0])
    length = int(meta.get("length", 0))
    r, c = int(start[0]), int(start[1])
    if length <= 0:
        return None
    chars: List[str] = []
    for offset in range(length):
        ch = puzzle["solution"][r][c + offset]
        if ch == "#":
            break
        chars.append(ch)
    return "".join(chars) if chars else None


def smoke_test(max_steps: Optional[int], num_samples: int) -> Any:
    env = load_environment(max_steps=max_steps)
    print(f"Loaded {len(env.puzzles)} puzzles")
    for i in range(num_samples):
        p = random.choice(env.puzzles)
        rows, cols = _grid_dimensions(p)
        clue_meta = p.get("clues", {}).get("across", {}).get("1")
        clue_text = clue_meta.get("clue") if clue_meta else "(no 1-Across)"
        answer = _extract_across_answer(p, "1") or "(n/a)"
        print(f"Sample {i + 1}: {rows}x{cols} | 1-Across: {clue_text} | Answer: {answer}")
    return env


def solve_episode(env: Any, render_steps: bool) -> None:
    obs = env.reset()
    if render_steps:
        print("Initial grid:")
        env.render()

    total_reward: float = 0.0
    solved: bool = False

    for r, row in enumerate(env.solution_grid):
        for c, ch in enumerate(row):
            if ch == "#":
                continue
            obs, reward, done, info = env.step((r, c, ch))
            total_reward += reward
            if render_steps:
                print(f"Step {env.steps_taken}: wrote {ch} at ({r},{c}) -> reward {reward:.2f}")
                env.render()
            if done:
                solved = True
                break
        if solved:
            break

    print(f"Episode finished | steps={env.steps_taken}, total_reward={total_reward:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the crossword RL environment")
    parser.add_argument("--samples", type=int, default=1, help="Number of random samples to print")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional environment step cap")
    parser.add_argument("--render-steps", action="store_true", help="Render the grid after each action during solve")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    env = smoke_test(args.max_steps, args.samples)
    solve_episode(env, args.render_steps)


if __name__ == "__main__":
    main()


