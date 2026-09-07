"""Command-line resource ranking using the shared IPFE wrapper."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from ranking_wrapper import IPFERanker, _borda_with_ties

# Preserve the original helper import without duplicating ranking logic.
argsort_with_ties = _borda_with_ties
DEFAULT_DATASET = Path(__file__).parent / 'dataset/resource-offers-ranking-format-sample.json'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', nargs='?', type=Path, default=DEFAULT_DATASET)
    parser.add_argument('--scale', type=int, default=10, help='Positive integer weight scale (default: 10)')
    args = parser.parse_args(argv)
    try:
        with args.dataset.open() as file:
            data = json.load(file)
        if not isinstance(data, dict):
            raise ValueError('Dataset must be a JSON object')
        ranker = IPFERanker.from_dataset(data, scale=args.scale)
        start = time.perf_counter()
        scores_plain = ranker.plaintext_scores(data)
        ranking_plain = np.argsort(-scores_plain)
        plain_time = time.perf_counter() - start
        start = time.perf_counter()
        scores_ipfe = ranker.decrypt_scores(ranker.encrypt_offers(data))
        ranking_ipfe = np.argsort(-scores_ipfe)
        ipfe_time = time.perf_counter() - start
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))

    if not np.array_equal(scores_plain, scores_ipfe):
        raise RuntimeError('IPFE scores do not match plaintext scores')
    print(f'--> Number of offers: {len(scores_plain)}, Number of QoS metrics: {ranker.m}')
    if not len(scores_plain):
        print('--> No offers to rank.')
        return 0
    top = int(ranking_plain[0])
    print(f'--> Plaintext top offer: {top}, time: {plain_time:.4f}s')
    print(f'--> IPFE top offer: {ranking_ipfe[0]}, time: {ipfe_time:.4f}s')
    print(f'--> Top offer[{top}] = {json.dumps({qos: data[qos][top] for qos in ranker.qos_list})}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
