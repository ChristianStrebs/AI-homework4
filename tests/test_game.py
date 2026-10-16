"""Tests for the game rules and the heuristic feature counts."""

import unittest

from fourinrow.game import (
    NORMAL_ORDER, REVERSE_ORDER, count_features, heuristic, index, is_win_at,
    parse_board, successors, winner,
)

EMPTY_ROWS = ". . . . . .\n" * 6


def board_with(**cells):
    """Empty board with marks placed, e.g. board_with(X=[(1, 1)], O=[(2, 2)])."""
    b = parse_board(EMPTY_ROWS)
    for player, positions in cells.items():
        for r, c in positions:
            b[index(r, c)] = player
    return b


class TestOrdering(unittest.TestCase):
    def test_normal_order_is_column_then_row(self):
        self.assertEqual(NORMAL_ORDER[:3], (index(1, 1), index(2, 1), index(3, 1)))
        self.assertEqual(NORMAL_ORDER[6], index(1, 2))
        self.assertEqual(NORMAL_ORDER[-1], index(6, 6))

    def test_reverse_order(self):
        self.assertEqual(REVERSE_ORDER, tuple(reversed(NORMAL_ORDER)))


class TestSuccessors(unittest.TestCase):
    def test_only_cells_next_to_a_piece(self):
        b = board_with(X=[(1, 1)])
        self.assertEqual(successors(b), [index(2, 1), index(1, 2), index(2, 2)])

    def test_includes_diagonal_neighbors_and_uses_order(self):
        b = board_with(O=[(3, 3)])
        moves = successors(b)
        self.assertEqual(len(moves), 8)
        self.assertEqual(moves[0], index(2, 2))      # column 2, smallest row first
        self.assertEqual(moves[-1], index(4, 4))
        self.assertEqual(successors(b, REVERSE_ORDER), moves[::-1])


class TestWin(unittest.TestCase):
    def test_four_in_each_direction(self):
        for cells in ([(2, 1), (2, 2), (2, 3), (2, 4)],     # horizontal
                      [(1, 6), (2, 6), (3, 6), (4, 6)],     # vertical
                      [(1, 1), (2, 2), (3, 3), (4, 4)],     # diagonal
                      [(1, 6), (2, 5), (3, 4), (4, 3)]):    # anti-diagonal
            b = board_with(X=cells)
            self.assertTrue(is_win_at(b, index(*cells[1]), "X"), cells)
            self.assertEqual(winner(b), "X")

    def test_three_is_not_a_win(self):
        b = board_with(O=[(5, 2), (5, 3), (5, 4)], X=[(5, 5)])
        self.assertFalse(is_win_at(b, index(5, 3), "O"))
        self.assertIsNone(winner(b))


class TestFeatures(unittest.TestCase):
    # count_features returns (two_open_3, one_open_3, two_open_2, one_open_2)

    def test_open_two(self):
        self.assertEqual(count_features(board_with(X=[(3, 3), (3, 4)]), "X"), (0, 0, 1, 0))

    def test_two_blocked_by_edge_and_by_opponent(self):
        self.assertEqual(count_features(board_with(X=[(1, 1), (1, 2)]), "X"), (0, 0, 0, 1))
        self.assertEqual(count_features(board_with(X=[(3, 3), (3, 4)], O=[(3, 2)]), "X"), (0, 0, 0, 1))
        self.assertEqual(count_features(board_with(X=[(3, 3), (3, 4)], O=[(3, 2), (3, 5)]), "X"), (0, 0, 0, 0))

    def test_three_counts_once_not_as_twos(self):
        self.assertEqual(count_features(board_with(O=[(2, 2), (3, 3), (4, 4)]), "O"), (1, 0, 0, 0))
        self.assertEqual(count_features(board_with(O=[(1, 4), (2, 4), (3, 4)]), "O"), (0, 1, 0, 0))

    def test_anti_diagonal(self):
        self.assertEqual(count_features(board_with(X=[(2, 5), (3, 4)]), "X"), (0, 0, 1, 0))

    def test_state_a_by_hand(self):
        a = parse_board("""
            . O . . . .
            . . X . . .
            . O X X . .
            . . O . . .
            . . . . . .
            . . . . . .""")
        # X: col 3, row 3 and diagonal (2,3)-(3,4) pairs are each blocked on one end.
        self.assertEqual(count_features(a, "X"), (0, 0, 0, 3))
        # O: diagonal (3,2)-(4,3) is open at (2,1) and (5,4).
        self.assertEqual(count_features(a, "O"), (0, 0, 1, 0))
        self.assertEqual(heuristic(a, "X"), 3 * 3 - 18)
        self.assertEqual(heuristic(a, "O"), 10 - 7 * 3)


if __name__ == "__main__":
    unittest.main()
