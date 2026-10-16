"""Depth-limited minimax and depth-limited minimax with alpha-beta pruning.

Both searches share the same successor function, terminal test, terminal
utilities, heuristic, and move ordering (see game.py).  Conventions:

* "me" is the player to move at the root; every value is from me's
  point of view.  MAX nodes are me-to-move, MIN nodes are opponent-to-move.
* Depth d means a d-ply tree: the root is at ply 0 and nodes at ply d are
  cut off and scored with the heuristic (unless they are terminal).
* Terminal test (checked before the cutoff test): the last move made
  4-in-a-row -> +1000 if me made it, -1000 if the opponent did; a full
  board with no winner -> 0.
* Nodes generated: every child state created by applying a move, i.e. one
  per successor actually produced during the search.  The root is not
  counted.  Alpha-beta generates successors one at a time, so moves after a
  cutoff are never generated.
* Ties: children are explored in the given move order and the root keeps
  the first move that reaches the best value (a later move must be strictly
  better to replace it).  With NORMAL_ORDER this is exactly the assignment's
  tie-breaking rule (smaller column first, then smaller row).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .game import (
    EMPTY, LOSS_UTILITY, NORMAL_ORDER, TIE_UTILITY, WIN_UTILITY,
    heuristic, is_full, is_win_at, opponent, successors,
)

INF = float("inf")


@dataclass
class SearchResult:
    algorithm: str
    depth: int
    move: int | None          # flat board index of the selected move
    value: float              # minimax value of the root
    nodes: int                # nodes generated (root not counted)
    cpu_ms: float             # CPU time of the search, in milliseconds
    root_values: dict = field(default_factory=dict)  # move -> backed-up value


class _Search:
    """State shared by one search: the root player, the move order, and the
    node counter.  The board is modified in place (make / unmake move)."""

    def __init__(self, me, order):
        self.me = me
        self.opp = opponent(me)
        self.order = order
        self.nodes = 0

    # -- terminal test + cutoff test --------------------------------------

    def _static_value(self, board, last_move, depth_left):
        """Return the utility/heuristic if this node is terminal or at the
        cutoff depth, otherwise None (meaning: expand it)."""
        if last_move is not None:
            mover = board[last_move]
            if is_win_at(board, last_move, mover):
                return WIN_UTILITY if mover == self.me else LOSS_UTILITY
        if is_full(board):
            return TIE_UTILITY
        if depth_left == 0:
            return heuristic(board, self.me)
        return None

    # -- plain minimax ------------------------------------------------------

    def minimax_value(self, board, to_move, depth_left, last_move):
        static = self._static_value(board, last_move, depth_left)
        if static is not None:
            return static
        moves = successors(board, self.order)
        if not moves:                      # no legal move: game over, tie
            return TIE_UTILITY
        maximizing = to_move == self.me
        best = -INF if maximizing else INF
        nxt = opponent(to_move)
        for m in moves:
            board[m] = to_move             # generate child
            self.nodes += 1
            v = self.minimax_value(board, nxt, depth_left - 1, m)
            board[m] = EMPTY               # undo
            if maximizing:
                if v > best:
                    best = v
            elif v < best:
                best = v
        return best

    # -- alpha-beta -----------------------------------------------------------

    def alphabeta_value(self, board, to_move, depth_left, alpha, beta, last_move):
        static = self._static_value(board, last_move, depth_left)
        if static is not None:
            return static
        moves = successors(board, self.order)
        if not moves:
            return TIE_UTILITY
        nxt = opponent(to_move)
        if to_move == self.me:             # MAX node
            v = -INF
            for m in moves:
                board[m] = to_move
                self.nodes += 1
                v = max(v, self.alphabeta_value(board, nxt, depth_left - 1, alpha, beta, m))
                board[m] = EMPTY
                if v >= beta:              # MIN above will never allow this
                    return v
                alpha = max(alpha, v)
            return v
        else:                              # MIN node
            v = INF
            for m in moves:
                board[m] = to_move
                self.nodes += 1
                v = min(v, self.alphabeta_value(board, nxt, depth_left - 1, alpha, beta, m))
                board[m] = EMPTY
                if v <= alpha:             # MAX above will never allow this
                    return v
                beta = min(beta, v)
            return v


def _run(algorithm, board, me, depth, order):
    """Root of both searches: try each legal move for `me` and keep the best."""
    if depth < 1:
        raise ValueError("depth must be at least 1")
    board = list(board)                    # never mutate the caller's board
    s = _Search(me, order)
    start = time.process_time()

    best_move, best_value = None, -INF
    alpha, beta = -INF, INF
    root_values = {}
    for m in successors(board, order):
        board[m] = me
        s.nodes += 1
        if algorithm == "minimax":
            v = s.minimax_value(board, s.opp, depth - 1, m)
        else:
            v = s.alphabeta_value(board, s.opp, depth - 1, alpha, beta, m)
        board[m] = EMPTY
        root_values[m] = v
        if v > best_value:                 # strict: earlier move wins ties
            best_move, best_value = m, v
        if algorithm == "alphabeta":
            alpha = max(alpha, best_value)

    cpu_ms = (time.process_time() - start) * 1000.0
    return SearchResult(algorithm, depth, best_move, best_value, s.nodes, cpu_ms, root_values)


def minimax_search(board, me, depth, order=NORMAL_ORDER):
    """Depth-limited minimax search for the player `me` to move."""
    return _run("minimax", board, me, depth, order)


def alphabeta_search(board, me, depth, order=NORMAL_ORDER):
    """Depth-limited minimax search with alpha-beta pruning for `me` to move."""
    return _run("alphabeta", board, me, depth, order)


# ---------------------------------------------------------------------------
# Analysis helpers (used for the report, not part of the timed searches)
# ---------------------------------------------------------------------------

def exact_root_values(board, me, depth, order=NORMAL_ORDER):
    """Exact minimax value of every root move (each child searched with a
    full window, so no value is just a pruning bound)."""
    board = list(board)
    s = _Search(me, order)
    values = {}
    for m in successors(board, order):
        board[m] = me
        values[m] = s.alphabeta_value(board, s.opp, depth - 1, -INF, INF, m)
        board[m] = EMPTY
    return values


def principal_variation(board, me, depth, order=NORMAL_ORDER, first_move=None):
    """The line of play both sides would follow inside a depth-`depth`
    search for `me`: at each ply the side to move picks its best move
    (MAX for me, MIN for the opponent) under the remaining depth, using the
    same tie-breaking.  If `first_move` is given, me's first move is forced
    to it (to show why that root move gets its value).
    Returns (list of (player, move), value at the end of the line)."""
    board = list(board)
    s = _Search(me, order)
    line, to_move, last = [], me, None
    for depth_left in range(depth, 0, -1):
        if s._static_value(board, last, depth_left) is not None:
            break
        moves = successors(board, order)
        if not line and first_move is not None:
            moves = [first_move]
        if not moves:
            break
        best_m, best_v = None, None
        for m in moves:
            board[m] = to_move
            v = s.alphabeta_value(board, opponent(to_move), depth_left - 1, -INF, INF, m)
            board[m] = EMPTY
            better = best_v is None or (v > best_v if to_move == me else v < best_v)
            if better:
                best_m, best_v = m, v
        board[best_m] = to_move
        line.append((to_move, best_m))
        last, to_move = best_m, opponent(to_move)
    leaf = s._static_value(board, last, 0)
    return line, leaf
