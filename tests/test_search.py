"""Tests for minimax and alpha-beta: agreement, pruning, tactics, and the
assignment's reported results."""

import random
import unittest

from fourinrow.game import (
    EMPTY, NORMAL_ORDER, REVERSE_ORDER, index, is_win_at, parse_board, successors,
)
from fourinrow.search import alphabeta_search, exact_root_values, minimax_search

STATE_A = parse_board("""
    . O . . . .
    . . X . . .
    . O X X . .
    . . O . . .
    . . . . . .
    . . . . . .""")

STATE_B = parse_board("""
    . . . . . .
    . . . . . .
    . . O . . .
    . X X O . .
    . . . . . .
    . . . . . .""")


def random_position(rng, n_moves):
    """A legal, non-terminal position reached by random play from one piece."""
    while True:
        b = [EMPTY] * 36
        b[index(rng.randint(2, 5), rng.randint(2, 5))] = "X"
        player, ok = "O", True
        for _ in range(n_moves):
            m = rng.choice(successors(b))
            b[m] = player
            if is_win_at(b, m, player):
                ok = False
                break
            player = "X" if player == "O" else "O"
        if ok:
            return b, player


class TestAgreement(unittest.TestCase):
    def test_alphabeta_matches_minimax_on_random_positions(self):
        rng = random.Random(4750)
        for _ in range(25):
            board, player = random_position(rng, rng.randint(2, 9))
            for depth in (1, 2, 3):
                mm = minimax_search(board, player, depth)
                ab = alphabeta_search(board, player, depth)
                ab_rev = alphabeta_search(board, player, depth, REVERSE_ORDER)
                self.assertEqual(ab.value, mm.value)
                self.assertEqual(ab.move, mm.move)          # same order, same tie-break
                self.assertEqual(ab_rev.value, mm.value)    # value never depends on order
                self.assertLessEqual(ab.nodes, mm.nodes)
                self.assertEqual(mm.root_values[ab_rev.move], mm.value)  # reverse picks a best move

    def test_search_does_not_modify_board(self):
        before = list(STATE_A)
        alphabeta_search(STATE_A, "X", 3)
        minimax_search(STATE_A, "X", 2)
        self.assertEqual(STATE_A, before)


class TestTactics(unittest.TestCase):
    def test_takes_immediate_win(self):
        b = parse_board("""
            . . . . . .
            . X X X . .
            . O O O . .
            . . . . . .
            . . . . . .
            . . . . . .""")
        for depth in (1, 2, 3):
            r = alphabeta_search(b, "X", depth)
            self.assertEqual(r.move, index(2, 1))           # (2,1) comes before (2,5)
            self.assertEqual(r.value, 1000)

    def test_blocks_immediate_loss(self):
        b = parse_board("""
            . . . . . .
            . . . . . .
            . O O O X .
            . . X . . .
            . . . X . .
            . . . . . .""")
        r = alphabeta_search(b, "X", 2)
        self.assertEqual(r.move, index(3, 1))
        self.assertGreater(r.value, -1000)


class TestReportedResults(unittest.TestCase):
    """Regression check: the numbers in the report."""

    def test_task2(self):
        mm = minimax_search(STATE_A, "X", 4, NORMAL_ORDER)
        ab = alphabeta_search(STATE_A, "X", 4, NORMAL_ORDER)
        rev = alphabeta_search(STATE_A, "X", 4, REVERSE_ORDER)
        for r in (mm, ab, rev):
            self.assertEqual(r.move, index(4, 5))
            self.assertEqual(r.value, -71)
        self.assertEqual((mm.nodes, ab.nodes, rev.nodes), (87190, 15746, 4476))

    def test_task3(self):
        expected = {2: (-77, 95), 3: (-1, 962), 4: (-208, 6746)}
        for depth, (value, nodes) in expected.items():
            r = alphabeta_search(STATE_B, "X", depth)
            self.assertEqual((r.move, r.value, r.nodes), (index(2, 2), value, nodes))

    def test_task3_depth2_is_a_tie_broken_by_column(self):
        values = exact_root_values(STATE_B, "X", 2)
        self.assertEqual(values[index(2, 2)], values[index(5, 5)])


if __name__ == "__main__":
    unittest.main()
