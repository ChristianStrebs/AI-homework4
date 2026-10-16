# Minimax and Alpha-Beta Pruning for Four-in-a-Row (6x6)

A Python agent for a two-player four-in-a-row game, built for CS/ECE 4750/7750
(Introduction to Artificial Intelligence), Homework 4. It plays with
**depth-limited minimax** and with **minimax plus alpha-beta pruning**. Both
use a hand-tuned heuristic evaluation, and it measures how pruning, move
ordering and search depth change the cost of the search.

- Pure Python 3, standard library only, with a make/unmake board, precomputed
  line tables and a win test that only checks lines through the last move.
- Alpha-beta gives the same answer as minimax but generates **82% fewer nodes**
  at depth 4. With better move ordering it generates **95% fewer**.
- 18 unit tests check the game rules, hand-computed heuristic values,
  minimax = alpha-beta agreement on random positions, and the reported results.
- Write-up: [`CS4750_HW4_Report.pdf`](CS4750_HW4_Report.pdf).

## Results

**Minimax vs. alpha-beta** (State A, X to move, 4-ply search):

| Method | Selected move | Minimax value | Nodes generated |
|---|---|---|---|
| Minimax, normal order | (row 4, col 5) | -71 | 87,190 |
| Alpha-beta, normal order | (row 4, col 5) | -71 | 15,746 |
| Alpha-beta, reverse order | (row 4, col 5) | -71 | 4,476 |

**Search depth** (State B, X to move, alpha-beta):

| Depth | Selected move | Minimax value | Nodes generated |
|---|---|---|---|
| 2 | (row 2, col 2) | -77 | 95 |
| 3 | (row 2, col 2) | -1 | 962 |
| 4 | (row 2, col 2) | -208 | 6,746 |

CPU times are in [`results.txt`](results.txt). They vary by machine; the moves,
values and node counts are deterministic.

## The game

- Players X and O take turns on a 6x6 grid. Four in a row (horizontal,
  vertical or diagonal) wins.
- A move may go on any empty cell next to an existing piece, horizontally,
  vertically or diagonally.
- Ties between moves go to the smaller column, then the smaller row. This is
  also the "normal" move ordering.
- A terminal state is worth +1000 for a win, -1000 for a loss and 0 for a tie.
- Non-terminal leaves are scored with
  `h = 100·open3(me) − 300·open3(opp) + 30·half3(me) − 80·half3(opp)
  + 10·open2(me) − 18·open2(opp) + 3·half2(me) − 7·half2(opp)`.
  Here "open" means empty cells at both ends of the run and "half" means an
  empty cell at one end.

## Run

Python 3.9 or newer (tested with 3.13). There is nothing to install.

```
python main.py            # both experiments, plus the depth-analysis evidence
python main.py --task 2   # minimax vs. alpha-beta and move ordering
python main.py --task 3   # search depth comparison
python -m unittest -v     # tests
```

Search any position. The file holds 6 lines of 6 symbols (`X`, `O` or `.`),
row 1 first:

```
python main.py --board my_board.txt --player X --depth 4            # alpha-beta, normal order
python main.py --board my_board.txt --player X --depth 4 --reverse  # alpha-beta, reverse order
python main.py --board my_board.txt --player X --depth 4 --minimax  # plain minimax
```

## Code layout

| File | Contents |
|---|---|
| `fourinrow/game.py` | Board, successor function, move ordering, win test, heuristic h(n) |
| `fourinrow/search.py` | Depth-limited minimax, alpha-beta, node counting, CPU timing, principal variation |
| `main.py` | Command-line runner for the experiments and for custom positions |
| `tests/` | Unit tests |

## Conventions

- Moves are written (row, col), 1-based.
- Depth d means a d-ply tree. Nodes at ply d are scored with h(n) unless they
  are terminal, and the terminal test is checked first.
- Nodes generated counts the child states created by applying a move; the root
  is not counted. Alpha-beta creates children one at a time, so pruned
  children are never generated.
- Heuristic runs are maximal lines of exactly 2 or 3 of the same mark, so a 3
  is not also counted as 2s. An end is open if the next cell in that direction
  is on the board and empty.

## Acknowledgment

The course permits AI tools for this assignment. The code and report were
developed with help from an AI coding assistant (Claude Code). The results
come from this program and are checked by its tests.
