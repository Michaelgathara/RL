# crossword

### Overview
- **Environment ID**: `crossword`
- **Short description**: Gymnasium-style RL environment where an agent fills a crossword grid letter-by-letter given clues.
- **Tags**: RL, games, grid, text, crossword, gymnasium


### Datasets
- **Primary dataset(s)**: NYT-style crossword clue–answer pairs (CSV)
- **Local file**: `nytcrosswords.csv` (expected columns: `Date,Word,Clue`)
- **Fallback**: If the CSV is unavailable or unsuitable, the loader falls back to `puzzles.json`, then to a tiny built-in sample.
- **Split sizes**: N/A (episodic RL; puzzles are sampled per episode)

Notes on loading:
- The default loader prefers `nytcrosswords.csv` alongside `crossword.py` and constructs lightweight 1×N across-only puzzles from alphabetic answers (length 3–8 by default).
- Robust CSV decoding is used (tries UTF-8/UTF-8-SIG/CP1252/Latin-1; then UTF-8 with replacement) to tolerate varied encodings.

### Task
- **Type**: multi-turn
- **Parser**: N/A (standard Gymnasium-like RL loop)
- **Episode**: One puzzle per episode; the agent writes letters into grid cells.

#### Observation
The observation is a dictionary:
- `grid`: 2D list of strings, where each cell is `#` (block), `-` (empty), or an uppercase letter.
- `clues`: `{ "across": {num -> {start, length, clue}}, "down": { ... } }` mirroring the puzzle specification.

For CSV-derived puzzles the grid is 1×N (across-only). JSON/fallback puzzles may have larger grids and down clues.

#### Action
Tuple `(row, column, letter)` where:
- `row`, `column`: zero-based indices
- `letter`: either an integer 0–25 (A–Z) or a single-character string `"A".."Z"`

#### Rewards (per step)
- `+1.0` for a correct letter placed in an empty, non-block cell
- `-0.5` for an incorrect letter or attempting to overwrite a filled cell
- `-1.0` for an out-of-bounds action (terminates the episode)
- `-1.0` for attempting to write on a block (`#`) cell (episode continues)
- `-0.01` step penalty each action
- `+5.0` bonus when the puzzle is completely solved

The episode ends when the puzzle is solved, an out-of-bounds action occurs, or `max_steps` is reached (if set).

### Quickstart
Run an evaluation with default settings:

```bash
uv run vf-eval crossword
```

Configure model and sampling:

```bash
uv run vf-eval crossword \
  -m gpt-4.1-mini \
  -n 20 -r 3 -t 1024 -T 0.7 \
  -a '{"max_steps": 400}'
```

Local smoke test (Python):

```python
from crossword.environments.crossword.crossword import load_environment

env = load_environment(max_steps=200)
obs = env.reset()
print(obs["grid"])  # show initial grid
env.render()         # pretty print grid
```

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `max_steps` | int | `null` | Optional cap on episode length. Episode also ends on solve. |

### Metrics

| Metric | Meaning |
| ------ | ------- |
| `reward` | Cumulative episode reward (step rewards + solve bonus). |
| `letters_correct` | Count of correct letters placed (if tracked by evaluator). |
| `completion` | Whether the grid was fully solved (boolean). |

