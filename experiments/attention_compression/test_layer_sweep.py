# Unit tests for layer_sweep.py's pure-Python functions
# (candidate_layers, pick_best_layer) -- no GPU/model needed. This is
# only possible because compress_candidates()/evaluate_candidates() keep
# their transformers/llmlingua imports lazy (inside the function bodies)
# rather than at module level -- see layer_sweep.py's top-of-file
# comment. compress_candidates/evaluate_candidates themselves aren't
# covered here; they need a real model and get exercised for real when
# the sweep actually runs.
#
# Run directly: python test_layer_sweep.py
import unittest

from layer_sweep import candidate_layers, pick_best_layer


class CandidateLayersTest(unittest.TestCase):
    def test_matches_expected_depths_for_28_layers(self):
        # Qwen2.5-1.5B-Instruct-shaped: 50%=14th layer(idx13), 66%~=18th(idx17),
        # 75%=21st(idx20), 100%=28th/last(idx27).
        self.assertEqual(candidate_layers(28), [13, 17, 20, 27])

    def test_100_percent_is_the_last_layer_not_out_of_range(self):
        for n in [1, 2, 5, 16, 28, 32]:
            layers = candidate_layers(n)
            self.assertEqual(max(layers), n - 1, f"n={n}")
            self.assertTrue(all(0 <= l < n for l in layers), f"n={n} layers={layers}")

    def test_dedupes_when_small_model_collapses_fractions_onto_one_index(self):
        # n=1: every fraction rounds to the same single valid index (0).
        self.assertEqual(candidate_layers(1), [0])

    def test_order_is_shallow_to_deep_matching_fraction_order(self):
        layers = candidate_layers(28)
        self.assertEqual(layers, sorted(layers))  # increasing fractions -> increasing indices here


class PickBestLayerTest(unittest.TestCase):
    def test_picks_argmax_mean_em_per_model(self):
        records = [
            {"scorer_model_size": "1.5b", "layer": 10, "best_subspan_em": 0.0},
            {"scorer_model_size": "1.5b", "layer": 10, "best_subspan_em": 1.0},  # mean 0.5
            {"scorer_model_size": "1.5b", "layer": 20, "best_subspan_em": 1.0},
            {"scorer_model_size": "1.5b", "layer": 20, "best_subspan_em": 1.0},  # mean 1.0 -- wins
        ]
        result = pick_best_layer(records)
        self.assertEqual(result["1.5b"]["chosen_layer"], 20)
        self.assertAlmostEqual(result["1.5b"]["by_layer"][10], 0.5)
        self.assertAlmostEqual(result["1.5b"]["by_layer"][20], 1.0)

    def test_keeps_model_sizes_independent(self):
        records = [
            {"scorer_model_size": "1.5b", "layer": 10, "best_subspan_em": 1.0},  # 1.5b's best
            {"scorer_model_size": "1.5b", "layer": 20, "best_subspan_em": 0.0},
            {"scorer_model_size": "7b", "layer": 10, "best_subspan_em": 0.0},
            {"scorer_model_size": "7b", "layer": 20, "best_subspan_em": 1.0},  # 7b's best -- different layer
        ]
        result = pick_best_layer(records)
        self.assertEqual(result["1.5b"]["chosen_layer"], 10)
        self.assertEqual(result["7b"]["chosen_layer"], 20)

    def test_tie_favors_first_seen_layer(self):
        # Equal mean EM at two layers -- max() over a dict keeps the
        # first-inserted key on a tie, which is whichever layer's records
        # were appended first (candidate_layers' shallow-to-deep order).
        records = [
            {"scorer_model_size": "1.5b", "layer": 10, "best_subspan_em": 1.0},
            {"scorer_model_size": "1.5b", "layer": 20, "best_subspan_em": 1.0},
        ]
        result = pick_best_layer(records)
        self.assertEqual(result["1.5b"]["chosen_layer"], 10)


if __name__ == "__main__":
    unittest.main()
