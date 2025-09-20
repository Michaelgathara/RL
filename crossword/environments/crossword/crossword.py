"""
Crossword RL Environment
-----------------------

This module implements a simple crossword puzzle environment that can be used
for reinforcement learning (RL) research.  The environment exposes a
`load_environment` function which instantiates and returns a
`CrosswordEnv` object.  Each episode presents a single crossword puzzle to
the agent.  The agent interacts by placing letters on the grid and
receives rewards based on correctness and efficiency.
"""

from __future__ import annotations

import json
import csv
import random
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    # Gymnasium is optional; environment still runs without it
    gym = None
    spaces = None


class CrosswordEnv:
    """A simple crossword puzzle environment.

    Each episode draws a single puzzle from the provided dataset.  The
    puzzle is defined by its solution grid (a 2D list of uppercase
    characters) and a dictionary of clues.  When the agent starts an
    episode, the grid is empty and the clues are provided as part of the
    observation.  On each step the agent supplies an action consisting of
    ``(row, column, letter)``:

    * ``row`` (int): zero-based row index
    * ``column`` (int): zero-based column index
    * ``letter`` (str): a single character to write at that location

    Rewards: +1 for a correct letter, -0.5 for an incorrect letter or
    overwrite, -0.01 per step, +5 when the puzzle is fully solved.  Invalid
    moves (out of bounds or writing on a block) cost -1 and end the
    episode.  The episode also ends when the step limit is reached.
    """

    def __init__(self, puzzles: List[Dict[str, Any]], *, max_steps: Optional[int] = None) -> None:
        if not puzzles:
            raise ValueError("puzzles list may not be empty")
        self.puzzles: List[Dict[str, Any]] = puzzles
        self.max_steps: Optional[int] = max_steps
        self.current_puzzle: Optional[Dict[str, Any]] = None
        self.solution_grid: Optional[List[List[str]]] = None
        self.current_grid: Optional[List[List[Optional[str]]]] = None
        self.steps_taken: int = 0

        if spaces is not None:
            max_rows = max(len(p["solution"]) for p in puzzles)
            max_cols = max(len(p["solution"][0]) for p in puzzles)
            self.action_space: Optional[gym.Space] = spaces.Tuple(
                (
                    spaces.Discrete(max_rows),  # row index
                    spaces.Discrete(max_cols),  # col index
                    spaces.Discrete(26),        # letter index (0=A, 25=Z)
                )
            )
            self.observation_space: Optional[gym.Space] = spaces.Dict(
                {
                    "grid": spaces.Box(
                        low=0, high=27,
                        shape=(max_rows, max_cols),
                        dtype=int,
                    ),
                    "clues": spaces.Dict({}),
                }
            )
        else:
            self.action_space = None
            self.observation_space = None

    def reset(self) -> Dict[str, Any]:
        """Start a new episode and return the initial observation."""
        self.current_puzzle = random.choice(self.puzzles)
        self.solution_grid = [list(row) for row in self.current_puzzle["solution"]]
        self.current_grid = []
        for row in self.solution_grid:
            self.current_grid.append([None if cell != '#' else '#' for cell in row])
        self.steps_taken = 0
        return self._get_observation()

    def step(self, action: Union[Tuple[int, int, str], Tuple[int, int, int]]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Apply an action and return (observation, reward, done, info)."""
        if self.current_puzzle is None:
            raise RuntimeError("Environment must be reset before stepping")
        self.steps_taken += 1

        row, col, letter_val = action
        if isinstance(letter_val, int):
            if 0 <= letter_val < 26:
                letter = chr(ord('A') + letter_val)
            else:
                raise ValueError("letter index out of range (expected 0-25)")
        elif isinstance(letter_val, str) and len(letter_val) == 1:
            letter = letter_val.upper()
            if not letter.isalpha():
                raise ValueError("letter must be A-Z")
        else:
            raise ValueError(f"Invalid letter specification: {letter_val}")

        reward: float = 0.0
        done: bool = False
        info: Dict[str, Any] = {}

        # bounds check
        if not (0 <= row < len(self.solution_grid)) or not (0 <= col < len(self.solution_grid[0])):
            reward = -1.0
            return self._get_observation(), reward, True, info

        # check for block
        if self.solution_grid[row][col] == '#':
            reward = -1.0
        else:
            correct_letter = self.solution_grid[row][col]
            current_letter = self.current_grid[row][col]
            if current_letter is not None:
                # cannot overwrite
                reward = -0.5
            else:
                if letter == correct_letter:
                    reward = 1.0
                    self.current_grid[row][col] = correct_letter
                else:
                    reward = -0.5

        reward -= 0.01 

        if self._is_solved():
            reward += 5.0
            done = True
        elif self.max_steps is not None and self.steps_taken >= self.max_steps:
            done = True
        return self._get_observation(), reward, done, info

    def render(self) -> None:
        """Print the current grid to the console (for debugging)."""
        for row in self.current_grid:
            print("".join(cell if cell is not None else "." for cell in row))
        print()

    def _get_observation(self) -> Dict[str, Any]:
        rendered_grid: List[List[str]] = []
        for r, row in enumerate(self.current_grid):
            rendered_row: List[str] = []
            for c, cell in enumerate(row):
                if self.solution_grid[r][c] == '#':
                    rendered_row.append('#')
                elif cell is None:
                    rendered_row.append('-')
                else:
                    rendered_row.append(cell)
            rendered_grid.append(rendered_row)
        return {
            "grid": rendered_grid,
            "clues": self.current_puzzle["clues"] if self.current_puzzle is not None else {},
        }

    def _is_solved(self) -> bool:
        for r, row in enumerate(self.solution_grid):
            for c, solution_letter in enumerate(row):
                if solution_letter == '#':
                    continue
                if self.current_grid[r][c] != solution_letter:
                    return False
        return True


def _load_default_puzzles() -> List[Dict[str, Any]]:
    """Load puzzles from nytcrosswords.csv if present; otherwise fall back.

    CSV format is expected to have headers: Date,Word,Clue. We construct
    simple 1xN across-only puzzles from valid words to keep the environment
    lightweight. If the CSV is not available or yields no valid entries,
    we fall back to puzzles.json, and then to a built-in sample.
    """
    module_path = Path(__file__).resolve()
    csv_path = module_path.parent / "nytcrosswords.csv"

    if csv_path.exists():
        puzzles: List[Dict[str, Any]] = []
        max_puzzles: int = 200
        min_len: int = 3
        max_len: int = 8

        def _ingest(reader: csv.DictReader) -> None:
            nonlocal puzzles
            for row in reader:
                word_raw = (row.get("Word") or "").strip()
                clue = (row.get("Clue") or "").strip()
                if not word_raw or not clue:
                    continue
                word = word_raw.upper()
                if not word.isalpha():
                    continue
                if not (min_len <= len(word) <= max_len):
                    continue
                puzzle = {
                    "solution": [list(word)],
                    "clues": {
                        "across": {
                            "1": {
                                "start": [0, 0],
                                "length": len(word),
                                "clue": clue,
                            }
                        },
                        "down": {},
                    },
                }
                puzzles.append(puzzle)
                if len(puzzles) >= max_puzzles:
                    break

        for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
            try:
                puzzles.clear()
                with csv_path.open("r", encoding=enc, newline="") as f:
                    reader = csv.DictReader(f)
                    _ingest(reader)
                if puzzles:
                    return puzzles
            except UnicodeDecodeError:
                continue

        puzzles.clear()
        with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f)
            _ingest(reader)
        if puzzles:
            return puzzles

    json_path = module_path.parent / "puzzles.json"
    if json_path.exists():
        with json_path.open("r", encoding="utf-8") as f:
            puzzles_from_json: List[Dict[str, Any]] = json.load(f)
            for p in puzzles_from_json:
                if "solution" not in p or "clues" not in p:
                    raise ValueError("Each puzzle must contain 'solution' and 'clues'")
            return puzzles_from_json

    return [{
        "solution": [
            ["N", "A", "Y"],
            ["O", "R", "C"],
            ["T", "E", "A"],
        ],
        "clues": {
            "across": {
                "1": {"start": [0, 0], "length": 3, "clue": "Opposite of yes"},
                "2": {"start": [1, 0], "length": 3, "clue": "Mythical marine monster"},
                "3": {"start": [2, 0], "length": 3, "clue": "Hot beverage"},
            },
            "down": {},
        },
    }]


def load_environment(*, max_steps: Optional[int] = None) -> CrosswordEnv:
    """Factory function returning a CrosswordEnv."""
    puzzles = _load_default_puzzles()
    return CrosswordEnv(puzzles, max_steps=max_steps)
