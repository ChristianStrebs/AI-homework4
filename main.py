"""CS/ECE 4750/7750 HW4 -- minimax and alpha-beta on 6x6 four-in-a-row.

Usage:
    python main.py                 # run Task 2 and Task 3 (all report results)
    python main.py --task 2        # only Task 2
    python main.py --task 3        # only Task 3, plus the Q3 evidence lines
    python main.py --board FILE --player X --depth 4 [--reverse] [--minimax]
                                   # search any position (6 rows of 6 symbols)

Results are printed as tables.  CPU time is process CPU time
(time.process_time) and varies from machine to machine; the selected move,
minimax value and node counts are deterministic.
"""

import argparse

from fourinrow.game import (
    NORMAL_ORDER, REVERSE_ORDER, board_to_string, format_move, opponent, parse_board,
)
from fourinrow.search import (
    alphabeta_search, exact_root_values, minimax_search, principal_variation,
)

STATE_A = parse_board("""
    . O . . . .
    . . X . . .
    . O X X . .
    . . O . . .
    . . . . . .
    . . . . . .
""")

STATE_B = parse_board("""
    . . . . . .
    . . . . . .
    . . O . . .
    . X X O . .
    . . . . . .
    . . . . . .
""")


def print_table(title, first_col, rows):
    print(title)
    header = f"{first_col:<22}{'Selected Move':<18}{'Minimax Value':>14}{'Nodes Generated':>17}{'CPU Time (ms)':>15}"
    print(header)
    print("-" * len(header))
    for label, r in rows:
        print(f"{label:<22}{format_move(r.move):<18}{r.value:>14}{r.nodes:>17,}{r.cpu_ms:>15.1f}")
    print()


def format_line(line):
    return "  ->  ".join(f"{p}{format_move(m)}" for p, m in line)


def task2():
    print("=" * 86)
    print("TASK 2: Initial State A, X to move, depth 4")
    print(board_to_string(STATE_A))
    print()
    rows = [
        ("Minimax - normal", minimax_search(STATE_A, "X", 4, NORMAL_ORDER)),
        ("Alpha-beta - normal", alphabeta_search(STATE_A, "X", 4, NORMAL_ORDER)),
        ("Alpha-beta - reverse", alphabeta_search(STATE_A, "X", 4, REVERSE_ORDER)),
    ]
    print_table("Results", "Method", rows)
    line, value = principal_variation(STATE_A, "X", 4)
    print(f"Principal variation: {format_line(line)}   (leaf h = {value})")
    print()


def task3():
    print("=" * 86)
    print("TASK 3: Initial State B, X to move, alpha-beta with normal ordering")
    print(board_to_string(STATE_B))
    print()
    rows = [(str(d), alphabeta_search(STATE_B, "X", d, NORMAL_ORDER)) for d in (2, 3, 4)]
    print_table("Results", "Depth", rows)

    # Evidence for Q3: exact backed-up value of the top root moves at each
    # depth, and the line of play the search expects after each of them.
    print("Q3 evidence: exact values of the best root moves, and the expected line after each")
    for d in (2, 3, 4):
        values = exact_root_values(STATE_B, "X", d)
        ranked = sorted(values, key=lambda m: -values[m])   # stable: keeps normal order on ties
        top = ", ".join(f"{format_move(m)}={values[m]}" for m in ranked[:4])
        print(f"  depth {d}: {top}")
        for m in ranked[:2]:
            line, leaf = principal_variation(STATE_B, "X", d, first_move=m)
            print(f"      {format_line(line)}   (leaf = {leaf})")
    print()


def custom(args):
    with open(args.board, encoding="utf-8") as f:
        board = parse_board(f.read())
    order = REVERSE_ORDER if args.reverse else NORMAL_ORDER
    search = minimax_search if args.minimax else alphabeta_search
    r = search(board, args.player, args.depth, order)
    print(board_to_string(board))
    print()
    label = f"{r.algorithm} d={args.depth}" + (" rev" if args.reverse else "")
    print_table(f"{args.player} to move ({opponent(args.player)} is the opponent)", "Method", [(label, r)])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", choices=("2", "3"), help="run only this task")
    p.add_argument("--board", help="file with a custom 6x6 position")
    p.add_argument("--player", default="X", choices=("X", "O"), help="player to move (custom board)")
    p.add_argument("--depth", type=int, default=4, help="search depth in plies (custom board)")
    p.add_argument("--reverse", action="store_true", help="reverse move ordering (custom board)")
    p.add_argument("--minimax", action="store_true", help="plain minimax instead of alpha-beta (custom board)")
    args = p.parse_args()

    if args.board:
        custom(args)
        return
    if args.task in (None, "2"):
        task2()
    if args.task in (None, "3"):
        task3()


if __name__ == "__main__":
    main()
