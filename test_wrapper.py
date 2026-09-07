import json
from dataclasses import replace
import io
from contextlib import redirect_stdout
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np

from ranking_wrapper import IPFERanker, _borda_with_ties


class TestBordaRanks(unittest.TestCase):
    def test_known_ranks(self):
        for values, expected in [
            ([10, 20, 20, 30], [1, 3, 3, 4]),
            ([30, 10, 20, 20], [4, 1, 3, 3]),
            ([7, 7, 7], [3, 3, 3]),
            ([7], [1]),
            ([], []),
        ]:
            with self.subTest(values=values):
                np.testing.assert_array_equal(_borda_with_ties(values), expected)

    def test_invalid_values(self):
        for values in ([float('nan')], [float('inf')], [-float('inf')], [[1, 2]], 1):
            with self.subTest(values=values), self.assertRaises(ValueError):
                _borda_with_ties(values)


class TestIPFERanker(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).parent / 'dataset/resource-offers-ranking-format-sample.json'
        with path.open() as f:
            cls.data = json.load(f)
        cls.keys = list(cls.data['qos_priority'])
        cls.weights = [cls.data['qos_priority'][key] for key in cls.keys]
        cls.ranker = IPFERanker(cls.keys, cls.weights)

    def test_dataset_roundtrip(self):
        # Independent oracle: max-position rank equals count of values <= this value.
        columns = []
        for key in self.keys:
            values = np.asarray(self.data[key])
            if key != 'bandwidth':
                values = -values
            columns.append([np.count_nonzero(values <= value) for value in values])
        expected = np.column_stack(columns) @ np.round(np.asarray(self.weights) * 10).astype(int)
        order = np.argsort(-expected)
        np.testing.assert_array_equal(self.ranker.plaintext_scores(self.data), expected)
        np.testing.assert_array_equal(self.ranker.rank(self.data), order)
        ciphertexts = self.ranker.encrypt_offers(self.data)
        self.assertEqual(len(ciphertexts), len(expected))
        decrypted = self.ranker.decrypt_scores(ciphertexts)
        np.testing.assert_array_equal(decrypted, expected)
        np.testing.assert_array_equal(np.argsort(-decrypted), order)

    def test_bandwidth_only_roundtrip(self):
        ranker = IPFERanker(['bandwidth'], [1])
        data = {'bandwidth': [10, 20, 20, 30]}
        expected = [10, 30, 30, 40]
        np.testing.assert_array_equal(ranker.plaintext_scores(data), expected)
        np.testing.assert_array_equal(ranker.rank(data), np.argsort(-np.asarray(expected)))
        np.testing.assert_array_equal(ranker.decrypt_scores(ranker.encrypt_offers(data)), expected)

    def test_invalid_data(self):
        invalid = [
            {key: values for key, values in self.data.items() if key != self.keys[0]},
            dict(self.data, **{self.keys[0]: [1]}),
            dict(self.data, **{self.keys[0]: [float('nan')]}),
            dict(self.data, **{self.keys[0]: [float('inf')]}),
            dict(self.data, **{self.keys[0]: [[1, 2]]}),
        ]
        for index, data in enumerate(invalid):
            for method in (self.ranker.rank, self.ranker.plaintext_scores, self.ranker.encrypt_offers):
                with self.subTest(case=index, method=method.__name__), self.assertRaises(ValueError):
                    method(data)

    def test_empty_offers(self):
        data = {key: [] for key in self.keys}
        self.assertEqual(self.ranker.rank(data).size, 0)
        self.assertEqual(self.ranker.plaintext_scores(data).size, 0)
        self.assertEqual(len(self.ranker.encrypt_offers(data)), 0)
        self.assertEqual(self.ranker.decrypt_scores([]).size, 0)

    def test_invalid_configuration(self):
        for keys, weights in [([], []), (['bandwidth'], []), (['bandwidth'], [[1]]),
                              (['bandwidth'], [float('nan')])]:
            with self.subTest(keys=keys, weights=weights), self.assertRaises(ValueError):
                IPFERanker(keys, weights)

    def test_variable_metrics_and_order(self):
        for m in (1, 5, 6, 7, 12):
            keys = [f'metric_{i}' for i in range(m)]
            directions = {key: 'max' if i % 2 == 0 else 'min' for i, key in enumerate(keys)}
            data = {key: [10, 20, 20, 30] for key in keys}
            data['qos_priority'] = {key: (i + 1) / 10 for i, key in enumerate(keys)}
            data['qos_direction'] = directions
            expected = sum((i + 1) * np.array([1, 3, 3, 4] if i % 2 == 0 else [4, 3, 3, 1])
                           for i in range(m))
            for order in (keys, list(reversed(keys))):
                with self.subTest(metrics=m, order=order):
                    data['qos_priority'] = {key: data['qos_priority'][key] for key in order}
                    ranker = IPFERanker.from_dataset(data)
                    np.testing.assert_array_equal(ranker.plaintext_scores(data), expected)
                    np.testing.assert_array_equal(ranker.rank(data), np.argsort(-expected))
                    np.testing.assert_array_equal(ranker.decrypt_scores(ranker.encrypt_offers(data)), expected)

    def test_large_bound_and_slices(self):
        ranker = IPFERanker(['bandwidth'], [1], scale=20000)
        data = {'bandwidth': [1, 2, 3, 4]}
        batch = ranker.encrypt_offers(data)
        self.assertEqual(batch.score_max, 80000)
        np.testing.assert_array_equal(ranker.decrypt_scores(batch), [20000, 40000, 60000, 80000])
        # Slicing must retain N=4 even after another batch is encrypted.
        ranker.encrypt_offers({'bandwidth': [1]})
        self.assertEqual(batch[-1:].offer_count, 4)
        np.testing.assert_array_equal(ranker.decrypt_scores(batch[-1:]), [80000])

    def test_zero_weights(self):
        ranker = IPFERanker(['bandwidth'], [0.01])
        data = {'bandwidth': [1, 2]}
        np.testing.assert_array_equal(ranker.plaintext_scores(data), [0, 0])
        np.testing.assert_array_equal(ranker.decrypt_scores(ranker.encrypt_offers(data)), [0, 0])

    def test_batch_validation(self):
        ranker = IPFERanker(['bandwidth'], [1])
        batch = ranker.encrypt_offers({'bandwidth': [1, 2]})
        other = IPFERanker(['bandwidth'], [1])
        with self.assertRaisesRegex(ValueError, 'configuration or key'):
            other.decrypt_scores(batch)
        for invalid in (list(batch), replace(batch, config=()), replace(batch, score_max=1),
                        replace(batch, offer_count=0)):
            with self.subTest(invalid=type(invalid)), self.assertRaises(ValueError):
                ranker.decrypt_scores(invalid)
        from copy import copy
        bad_ciphertext = copy(batch[0])
        bad_ciphertext.c = bad_ciphertext.c[:-1]
        with self.assertRaisesRegex(ValueError, 'dimension'):
            ranker.decrypt_scores(replace(batch, ciphertexts=(bad_ciphertext,)))

    def test_configuration_validation(self):
        cases = [
            (['cpu'], [1], {}),
            (['cpu'], [1], {'directions': {'cpu': 'largest'}}),
            (['bandwidth'], [1], {'directions': {'cpu': 'max'}}),
            (['bandwidth', 'bandwidth'], [1, 1], {}),
            (['bandwidth'], [-1], {}),
            (['bandwidth'], [float('inf')], {}),
            (['bandwidth'], [1], {'scale': 0}),
            (['bandwidth'], [1], {'scale': -1}),
            (['bandwidth'], [1], {'scale': 1.5}),
            (['bandwidth'], [1], {'scale': True}),
            (['bandwidth'], [1e30], {}),
        ]
        for keys, weights, kwargs in cases:
            with self.subTest(kwargs=kwargs, weights=weights), self.assertRaises(ValueError):
                IPFERanker(keys, weights, **kwargs)
        ranker = IPFERanker(['bandwidth'], [1], scale=2**62)
        with self.assertRaisesRegex(ValueError, 'integer range'):
            ranker.plaintext_scores({'bandwidth': [1, 2]})

    def test_configuration_copied(self):
        keys, weights, directions = ['cpu'], [1], {'cpu': 'max'}
        ranker = IPFERanker(keys, weights, directions=directions)
        keys[0], weights[0], directions['cpu'] = 'other', 9, 'min'
        ranker.weights_scaled[0] = 999
        np.testing.assert_array_equal(ranker.plaintext_scores({'cpu': [1, 2]}), [10, 20])


class TestRankingCLI(unittest.TestCase):
    def test_import_has_no_side_effects(self):
        from ranking import argsort_with_ties
        self.assertIs(argsort_with_ties, _borda_with_ties)

    def test_seven_metric_cli_from_other_directory(self):
        root = Path(__file__).parent.resolve()
        result = subprocess.run(
            [sys.executable, str(root / 'ranking.py'), str(root / 'dataset/resource-offers-7-metrics.json')],
            cwd='/tmp', capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Number of QoS metrics: 7', result.stdout)
        self.assertIn('Plaintext top offer: 1', result.stdout)
        self.assertIn('IPFE top offer: 1', result.stdout)
        self.assertIn('"cpu": 8', result.stdout)

    def test_empty_input(self):
        from ranking import main
        data = {'qos_priority': {'bandwidth': 1}, 'bandwidth': []}
        with patch('ranking.json.load', return_value=data), redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main([]), 0)
        self.assertIn('No offers', output.getvalue())


if __name__ == '__main__':
    unittest.main(verbosity=2)
