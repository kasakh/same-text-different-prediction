#!/usr/bin/env python3
"""Unit tests for pls.target_batch : pure CPU, no torch/GPU. Run:
    python -m unittest tests.test_target_batch  (or invoke directly)
"""
import unittest

from pls.target_batch import (
    select_peers, place_target, compute_padded_length, build_context,
    COMPOSITION_POLICIES,
)


def make_pool(n=20):
    # example_id "0".."n-1"; length = 10 + i; label = i % 3; margin = i/100
    return [{"example_id": str(i), "label": i % 3, "length": 10 + i,
             "reference_margin": i / 100.0} for i in range(n)]


class TestSelectPeers(unittest.TestCase):
    def setUp(self):
        self.pool = make_pool(20)
        self.target = self.pool[5]  # id "5", label 2, length 15

    def test_excludes_target_and_count(self):
        for pol in COMPOSITION_POLICIES:
            peers = select_peers(self.target, self.pool, pol, 4, seed=1)
            self.assertTrue(all(p["example_id"] != "5" for p in peers), pol)
            self.assertLessEqual(len(peers), 4)

    def test_short_returns_shortest(self):
        peers = select_peers(self.target, self.pool, "short", 3)
        lengths = [p["length"] for p in peers]
        self.assertEqual(lengths, sorted(lengths))
        self.assertEqual(min(p["length"] for p in self.pool if p["example_id"] != "5"),
                         peers[0]["length"])

    def test_long_returns_longest(self):
        peers = select_peers(self.target, self.pool, "long", 3)
        self.assertEqual(peers[0]["length"], 29)  # id 19

    def test_same_label(self):
        peers = select_peers(self.target, self.pool, "same_label", 3)
        self.assertTrue(all(p["label"] == self.target["label"] for p in peers))

    def test_different_label(self):
        peers = select_peers(self.target, self.pool, "different_label", 3)
        self.assertTrue(all(p["label"] != self.target["label"] for p in peers))

    def test_low_high_margin(self):
        lo = select_peers(self.target, self.pool, "low_margin", 3)
        hi = select_peers(self.target, self.pool, "high_margin", 3)
        self.assertLess(lo[0]["reference_margin"], hi[0]["reference_margin"])
        self.assertEqual(hi[0]["reference_margin"], 0.19)  # id 19

    def test_random_deterministic(self):
        a = select_peers(self.target, self.pool, "random", 4, seed=7)
        b = select_peers(self.target, self.pool, "random", 4, seed=7)
        c = select_peers(self.target, self.pool, "random", 4, seed=8)
        self.assertEqual([p["example_id"] for p in a], [p["example_id"] for p in b])
        self.assertNotEqual([p["example_id"] for p in a], [p["example_id"] for p in c])

    def test_margin_policy_requires_margin(self):
        pool = [{"example_id": str(i), "label": 0, "length": 5} for i in range(5)]
        with self.assertRaises(ValueError):
            select_peers(pool[0], pool, "low_margin", 2)

    def test_insufficient_peers_returns_available(self):
        pool = make_pool(3)
        peers = select_peers(pool[0], pool, "short", 10)
        self.assertEqual(len(peers), 2)  # only 2 non-target candidates


class TestPlaceTarget(unittest.TestCase):
    def test_positions(self):
        peers = ["a", "b", "c", "d"]
        m_first, i_first = place_target(peers, "T", "first")
        m_last, i_last = place_target(peers, "T", "last")
        m_mid, i_mid = place_target(peers, "T", "middle")
        self.assertEqual(i_first, 0)
        self.assertEqual(m_first[0], "T")
        self.assertEqual(i_last, 4)
        self.assertEqual(m_last[-1], "T")
        self.assertEqual(i_mid, 2)
        self.assertEqual(m_mid[2], "T")
        for m in (m_first, m_last, m_mid):
            self.assertEqual(len(m), 5)
            self.assertEqual(m.count("T"), 1)


class TestPadding(unittest.TestCase):
    def test_dynamic_longest(self):
        self.assertEqual(compute_padded_length([12, 30, 7], "dynamic_longest", max_length=128), 30)

    def test_static(self):
        self.assertEqual(compute_padded_length([12, 30, 7], "static_max_length", max_length=128), 128)

    def test_pad_to_multiple_8(self):
        self.assertEqual(compute_padded_length([30], "dynamic_pad_to_multiple_8", max_length=128), 32)
        self.assertEqual(compute_padded_length([32], "dynamic_pad_to_multiple_8", max_length=128), 32)

    def test_truncation_cap(self):
        self.assertEqual(compute_padded_length([500], "dynamic_longest", max_length=128), 128)

    def test_length_bucketed(self):
        self.assertEqual(compute_padded_length([40], "length_bucketed", max_length=128, length_bucket=32), 64)


class TestBuildContextControlled(unittest.TestCase):
    """The whole point: isolating ONE axis holds the others fixed."""
    def setUp(self):
        self.pool = make_pool(30)
        self.target = self.pool[10]
        self.lengths = {p["example_id"]: p["length"] for p in self.pool}

    def test_position_variants_share_members_and_padding(self):
        # same composition+seed -> same peer set -> position variants must share
        # members (as a set) and identical padded length (shape held fixed).
        ctxs = [build_context(self.target, self.pool, 6, "short", pos, "dynamic_longest",
                              self.lengths, seed=3, max_length=128)
                for pos in ("first", "middle", "last")]
        member_sets = [frozenset(c["member_ids"]) for c in ctxs]
        self.assertEqual(member_sets[0], member_sets[1])
        self.assertEqual(member_sets[1], member_sets[2])
        self.assertEqual(len({c["padded_sequence_length"] for c in ctxs}), 1)
        self.assertEqual([c["target_index"] for c in ctxs], [0, 2, 5])

    def test_composition_variants_hold_batch_and_position(self):
        ctxs = [build_context(self.target, self.pool, 6, comp, "first", "dynamic_longest",
                              self.lengths, seed=3)
                for comp in ("short", "long", "mixed_length")]
        self.assertTrue(all(c["realized_batch_size"] == 6 for c in ctxs))
        self.assertTrue(all(c["target_index"] == 0 for c in ctxs))
        # different compositions -> different peer sets
        self.assertNotEqual(set(ctxs[0]["peer_example_ids"]), set(ctxs[1]["peer_example_ids"]))

    def test_padding_variants_hold_members(self):
        base = dict(target=self.target, pool=self.pool, batch_size=6,
                    composition_policy="short", position="first",
                    id_to_length=self.lengths, seed=3)
        a = build_context(padding_policy="dynamic_longest", **base)
        b = build_context(padding_policy="static_max_length", **base)
        self.assertEqual(a["member_ids"], b["member_ids"])   # members identical
        self.assertNotEqual(a["padded_sequence_length"], b["padded_sequence_length"])


class TestCompositionShapeIsolation(unittest.TestCase):
    """Composition MUST be isolated from tensor shape: under static padding the
    padded length is constant across composition variants (pure composition);
    dynamic padding would confound composition with shape (documented below)."""
    def setUp(self):
        self.pool = make_pool(40)
        self.target = self.pool[10]
        self.lengths = {p["example_id"]: p["length"] for p in self.pool}

    def test_static_padding_holds_padded_length_across_composition(self):
        pads = set()
        for comp in ("random", "short", "long", "mixed_length"):
            c = build_context(self.target, self.pool, 6, comp, "first", "static_max_length",
                              self.lengths, seed=3, max_length=128)
            pads.add(c["padded_sequence_length"])
        self.assertEqual(len(pads), 1)          # identical shape -> pure composition
        self.assertEqual(pads.pop(), 128)

    def test_dynamic_padding_confounds_composition_with_shape(self):
        short = build_context(self.target, self.pool, 6, "short", "first", "dynamic_longest",
                              self.lengths, seed=3, max_length=128)
        long = build_context(self.target, self.pool, 6, "long", "first", "dynamic_longest",
                             self.lengths, seed=3, max_length=128)
        # documents WHY composition spokes use static: dynamic padded length moves with peers
        self.assertLess(short["padded_sequence_length"], long["padded_sequence_length"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
