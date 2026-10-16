"""Game rules for 4-in-a-row on a 6x6 board.

Board representation
--------------------
The board is a flat Python list of 36 characters: 'X', 'O' or '.' (empty).
Cell (row, col), with 1-based row and column numbers as in the assignment,
is stored at index (row - 1) * 6 + (col - 1).  The search places a mark,
recurses, and removes the mark again (make/unmake), so no board copies are
made during search.
"""

SIZE = 6
WIN_LENGTH = 4
EMPTY = "."
PLAYERS = ("X", "O")

# Terminal utilities, from the point of view of the player to move at the root.
WIN_UTILITY = 1000
LOSS_UTILITY = -1000
TIE_UTILITY = 0

# The four line directions as (d_row, d_col): horizontal, vertical,
# down-right diagonal, down-left (anti-)diagonal.
DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))


def index(row, col):
    """Flat index of 1-based (row, col)."""
    return (row - 1) * SIZE + (col - 1)


def coords(idx):
    """1-based (row, col) of a flat index."""
    return idx // SIZE + 1, idx % SIZE + 1


def opponent(player):
    return "O" if player == "X" else "X"


def format_move(idx):
    row, col = coords(idx)
    return f"(row {row}, col {col})"


# ---------------------------------------------------------------------------
# Precomputed tables (built once at import time)
# ---------------------------------------------------------------------------

def _in_bounds(row, col):
    return 1 <= row <= SIZE and 1 <= col <= SIZE


def _build_neighbors():
    """NEIGHBORS[i] = flat indices of the (up to 8) cells touching cell i."""
    table = []
    for i in range(SIZE * SIZE):
        row, col = coords(i)
        table.append(tuple(
            index(row + dr, col + dc)
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)
            if (dr or dc) and _in_bounds(row + dr, col + dc)
        ))
    return tuple(table)


def _build_lines():
    """Every maximal straight line on the board (rows, columns, diagonals)
    with at least 2 cells, as a tuple of flat indices in board order.
    A 2-in-a-row or 3-in-a-row always lies inside exactly one such line."""
    lines = []
    for d_row, d_col in DIRECTIONS:
        for row in range(1, SIZE + 1):
            for col in range(1, SIZE + 1):
                # Start a line only at its first cell (the previous cell
                # in this direction is off the board).
                if _in_bounds(row - d_row, col - d_col):
                    continue
                cells = []
                r, c = row, col
                while _in_bounds(r, c):
                    cells.append(index(r, c))
                    r, c = r + d_row, c + d_col
                if len(cells) >= 2:
                    lines.append(tuple(cells))
    return tuple(lines)


def _build_rays():
    """RAYS[i] = for each of the 4 directions, the pair (cells going forward
    from i, cells going backward from i), excluding i itself.  Used for a
    fast win test around the last move."""
    table = []
    for i in range(SIZE * SIZE):
        row, col = coords(i)
        per_dir = []
        for d_row, d_col in DIRECTIONS:
            fwd, back = [], []
            r, c = row + d_row, col + d_col
            while _in_bounds(r, c):
                fwd.append(index(r, c))
                r, c = r + d_row, c + d_col
            r, c = row - d_row, col - d_col
            while _in_bounds(r, c):
                back.append(index(r, c))
                r, c = r - d_row, c - d_col
            per_dir.append((tuple(fwd), tuple(back)))
        table.append(tuple(per_dir))
    return tuple(table)


NEIGHBORS = _build_neighbors()
LINES = _build_lines()
RAYS = _build_rays()

# Move-ordering priority: increasing column number first, then increasing
# row number.  This is the "normal" ordering and also the tie-breaking rule.
NORMAL_ORDER = tuple(sorted(range(SIZE * SIZE), key=lambda i: (coords(i)[1], coords(i)[0])))
REVERSE_ORDER = tuple(reversed(NORMAL_ORDER))


# ---------------------------------------------------------------------------
# Board helpers
# ---------------------------------------------------------------------------

def parse_board(text):
    """Build a board from 6 whitespace-separated rows of 6 symbols each
    (row 1 first).  Example row: '. O . . . .'"""
    rows = [line.split() for line in text.strip().splitlines() if line.strip()]
    if len(rows) != SIZE or any(len(r) != SIZE for r in rows):
        raise ValueError("board must have 6 rows of 6 cells")
    board = [cell.upper() for r in rows for cell in r]
    if any(cell not in ("X", "O", EMPTY) for cell in board):
        raise ValueError("cells must be X, O or .")
    return board


def board_to_string(board):
    header = "      " + " ".join(f"c{c}" for c in range(1, SIZE + 1))
    lines = [header]
    for row in range(1, SIZE + 1):
        cells = "  ".join(board[index(row, c)] for c in range(1, SIZE + 1))
        lines.append(f"  r{row}   {cells}")
    return "\n".join(lines)


def successors(board, order=NORMAL_ORDER):
    """Legal moves: empty cells horizontally, vertically or diagonally
    adjacent to any existing piece, listed in the given move order."""
    return [
        i for i in order
        if board[i] == EMPTY and any(board[n] != EMPTY for n in NEIGHBORS[i])
    ]


def is_win_at(board, idx, player):
    """True if the mark `player` at `idx` is part of 4 or more in a row.
    Only lines through the last move need checking."""
    for fwd, back in RAYS[idx]:
        count = 1
        for j in fwd:
            if board[j] != player:
                break
            count += 1
        for j in back:
            if board[j] != player:
                break
            count += 1
        if count >= WIN_LENGTH:
            return True
    return False


def is_full(board):
    return EMPTY not in board


def winner(board):
    """'X' or 'O' if that player has 4 in a row anywhere, else None."""
    for i, cell in enumerate(board):
        if cell != EMPTY and is_win_at(board, i, cell):
            return cell
    return None


# ---------------------------------------------------------------------------
# Heuristic evaluation
# ---------------------------------------------------------------------------

# Feature names, indexed the same way as the tuple from count_features().
FEATURES = ("two_open_3", "one_open_3", "two_open_2", "one_open_2")

# Weights for (me, opponent), in FEATURES order.
WEIGHTS_ME = (100, 30, 10, 3)
WEIGHTS_OPP = (-300, -80, -18, -7)


def count_features(board, player):
    """Count the open runs belonging to `player`.

    A run is a maximal sequence of the player's marks along one row, column
    or diagonal.  Only runs of exactly 2 or 3 are counted (a 3-in-a-row is
    not also counted as two 2-in-a-rows).  An end of a run is "open" if the
    next cell past that end, in the same line, is on the board and empty.

    Returns (two_open_3, one_open_3, two_open_2, one_open_2).
    """
    two3 = one3 = two2 = one2 = 0
    for line in LINES:
        n = len(line)
        k = 0
        while k < n:
            if board[line[k]] != player:
                k += 1
                continue
            start = k
            while k < n and board[line[k]] == player:
                k += 1
            length = k - start  # run occupies line[start:k]
            if length == 2 or length == 3:
                open_ends = 0
                if start > 0 and board[line[start - 1]] == EMPTY:
                    open_ends += 1
                if k < n and board[line[k]] == EMPTY:
                    open_ends += 1
                if length == 3:
                    if open_ends == 2:
                        two3 += 1
                    elif open_ends == 1:
                        one3 += 1
                else:
                    if open_ends == 2:
                        two2 += 1
                    elif open_ends == 1:
                        one2 += 1
    return two3, one3, two2, one2


def heuristic(board, me):
    """h(n) from the assignment, from the point of view of `me` (the player
    whose move is being selected at the root of the search)."""
    mine = count_features(board, me)
    theirs = count_features(board, opponent(me))
    return (sum(w * c for w, c in zip(WEIGHTS_ME, mine))
            + sum(w * c for w, c in zip(WEIGHTS_OPP, theirs)))
