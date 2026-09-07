# ASRR-IPFE

A resource-ranking prototype using weighted Borda scores and inner-product functional encryption (IPFE) from `pymife`.

Both the command-line tool and Python wrapper support a configurable number of QoS metrics. They compute the same integer scores in plaintext and through IPFE. Borda ranks are computed from plaintext offers before encryption; the prototype retains the master key locally and does not implement separate trusted services or key storage.

## Project files

- [ranking.py](ranking.py): command-line entry point using the shared wrapper.
- [ranking_wrapper.py](ranking_wrapper.py): input validation, configurable Borda scoring, IPFE keys, and encrypted batches.
- [test_wrapper.py](test_wrapper.py): regression tests for scoring, encryption, configuration, and the CLI.
- [Original dataset](dataset/resource-offers-ranking-format-sample.json): 288 offers with five metrics.
- [Seven-metric dataset](dataset/resource-offers-7-metrics.json): three offers, adding CPU and memory with explicit preferences.

## Setup and tests

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -v
```

Dependencies: Python 3, `numpy`, and `pymife`. Tests use Python's built-in `unittest`.

Run commands from the project root. If `.venv` already exists with dependencies installed, run only the test command. The current suite reports `Ran 16 tests` followed by `OK`; failed checks produce a nonzero exit status. You can also run it directly with `.venv/bin/python test_wrapper.py`.

## Run the ranking tool

```bash
# Original 288-offer, five-metric dataset:
.venv/bin/python ranking.py

# Three offers with seven metrics:
.venv/bin/python ranking.py dataset/resource-offers-7-metrics.json

# Optional fixed-point weight precision:
.venv/bin/python ranking.py dataset/resource-offers-7-metrics.json --scale 100
```

`ranking.py` calls the shared wrapper, checks exact plaintext/IPFE score equality, and prints the top offer, all its configured metrics, and scoring times. Offer indices are zero-based. The default dataset path is relative to the script, so running the script from another directory also works. An empty offer set prints `No offers to rank.` Importing `ranking` does not run the demo.

With the default scale, the original dataset selects offer **236** and the seven-metric example selects offer **1** under both methods. Timing values vary per run. Reported times exclude key generation; the IPFE timing includes rank-vector construction, encryption, decryption, and sorting.

## Dataset configuration

Use one equal-length numeric array per metric. `qos_priority` selects metrics and provides nonnegative finite weights. `qos_direction` specifies whether larger (`max`) or smaller (`min`) values are preferred.

```json
{
  "qos_priority": {"price": 0.5, "cpu": 0.3, "memory": 0.2},
  "qos_direction": {"price": "min", "cpu": "max", "memory": "max"},
  "price": [0.1, 0.2, 0.05],
  "cpu": [4, 8, 2],
  "memory": [16, 32, 8]
}
```

Add or remove metrics through these mappings and their arrays. Extra top-level dataset fields are ignored, but `qos_direction` entries for metrics absent from `qos_priority` are rejected. All selected metrics must exist; arrays must be one-dimensional and finite, with the same number of offers. At least one metric is required. Weights do not need to sum to one and are not automatically normalized.

For compatibility, the original five metrics have default directions when omitted:

| Metric | Default |
| --- | --- |
| bandwidth | max |
| reliability | min |
| energy | min |
| latency | min |
| price | min |

Every new metric requires an explicit direction. Existing defaults can be overridden. In particular, set `"reliability": "max"` when reliability means a success rate; the legacy sample keeps its original behavior. The seven-metric example explicitly maximizes reliability, bandwidth, CPU, and memory.

### Extend a dataset to six, seven, or more metrics

1. Add each metric's array, keeping the same offer order and array length as the existing metrics.
2. Add its weight to `qos_priority` and its `min` or `max` preference to `qos_direction`.
3. Run `ranking.py` with the updated JSON path, or create a new ranker with `IPFERanker.from_dataset(data)`.

No Python source edits are needed. For example, adding `cpu` and `memory` to the original five metrics gives seven dimensions. To remove a metric, remove its priority and direction entries; its unused top-level array may also be removed.

`qos_priority` insertion order defines vector coordinates. The wrapper pairs each metric with its weight and direction by name, so reordering this mapping consistently does not change plaintext scores. Existing ciphertexts must still be evaluated with the ranker that created them.

## Python wrapper

```python
import json
import numpy as np
from ranking_wrapper import IPFERanker

with open('dataset/resource-offers-7-metrics.json') as file:
    data = json.load(file)

ranker = IPFERanker.from_dataset(data, scale=10)
scores = ranker.plaintext_scores(data)
order = ranker.rank(data)
batch = ranker.encrypt_offers(data)
encrypted_scores = ranker.decrypt_scores(batch)
np.testing.assert_array_equal(scores, encrypted_scores)
encrypted_order = np.argsort(-encrypted_scores)
np.testing.assert_array_equal(order, encrypted_order)

# A slice retains the ORIGINAL offer count, score bound, and configuration.
subset_scores = ranker.decrypt_scores(batch[1:3])
np.testing.assert_array_equal(subset_scores, scores[1:3])
```

Direct construction is also supported:

```python
ranker = IPFERanker(
    ['price', 'cpu'], [0.6, 0.4], scale=10,
    directions={'price': 'min', 'cpu': 'max'},
)
```

| Method | Result |
| --- | --- |
| `IPFERanker.from_dataset(data, scale=10)` | Ranker configured from JSON mappings |
| `ranker.plaintext_scores(data)` | Integer score array in original offer order |
| `ranker.rank(data)` | Offer indices sorted by descending plaintext score |
| `ranker.encrypt_offers(data)` | `EncryptedBatch` containing encrypted Borda vectors |
| `ranker.decrypt_scores(batch)` | Integer score array in the supplied batch order |

`rank()` performs plaintext scoring. For encrypted evaluation, decrypt a batch and sort the returned scores as shown above. `ranker.m` gives the number of metrics.

The constructor copies metric order, directions, and scaled weights. `qos_list` is a tuple, and `weights_scaled` returns a copy. Create a new ranker to change the configuration; regenerate ciphertexts under its new keys.

**Batch API change:** `encrypt_offers` returns an `EncryptedBatch`, not a plain list. Iteration, indexing, `len`, and slicing work. Pass a batch or batch slice to `decrypt_scores`; converting a nonempty batch to a list loses required metadata and is rejected. Use `batch[i:i+1]` to decrypt one offer. Empty lists remain accepted for compatibility.

Batches carry configuration and key identity to reject accidental cross-ranker use, plus the original offer count and maximum score. This is an in-memory API: metadata is not authenticated, and serialization or transport of untrusted ciphertexts is outside its scope.

Even a newly constructed ranker with identical settings has different keys and cannot decrypt an earlier ranker's batch. Keep the original ranker alive for its batches. Borda vectors depend on the offer set: adding or removing offers requires recomputing ranks and encrypting again. Slicing an existing batch evaluates scores from the original offer set rather than reranking the subset.

## Scoring and IPFE dimensions

For `N` offers and `m` metrics, the wrapper builds an `N × m` integer rank matrix `X`. For a maximized metric, a value's rank is the count of values less than or equal to it. Minimized metrics apply the same rule to negated values. Ties receive their maximum sorted position: `[10, 20, 20, 30]` becomes `[1, 3, 3, 4]`. These are maximum-position ranks, not dense ranks.

Weights use `w_prime = round(scale * weight)` with NumPy rounding; `scale` must be a positive integer. Scores are `X @ w_prime`, and higher scores win. Rounding can change rankings relative to the original fractional weights; very small weights can become zero. Plaintext and IPFE use the same rounded weights. All-zero scaled weights yield zero scores. Tied final scores use NumPy's default descending-score `argsort` without a separate tie-break policy.

`FeDDH.generate(m)` generates keys for exactly `m` dimensions. Each encrypted rank vector and functional-key weight vector uses the same metric order and length. Changing dimension requires new keys and new ciphertexts.

Each rank is between 1 and `N`, so nonnegative weights give the safe bound:

```text
S_max = N * sum(w_prime)
decryption search interval = (0, S_max + 1)
```

The upper endpoint includes the maximum score even where the library treats it as exclusive. Slices retain the original `N`; bounds never depend on the number of ciphertexts selected for decryption. Signed 64-bit weight/score overflow is rejected before scoring or encryption. There is no hardcoded metric-count cap, but more metrics increase encryption work and wider score intervals can increase decryption time.

Tests cover the legacy dataset, 1/5/6/7/12 metrics, mixed directions, reordered configurations, ties, invalid inputs, wrong batch/key/dimension metadata, zero weights, overflow, slices with scores above 50,000, and the CLI.

## Mathematical description

This section gives the ranking model in compact mathematical form. Let:

- `N` be number of offers;
- `m` be number of configured QoS metrics;
- `v_{ij}` be raw value of metric `j` for offer `i`;
- `w_j` be configured weight for metric `j`;
- `d_j` be its direction: `max` or `min`.

### 1. Rank each metric

For one metric, stably sort its values. A tie group receives the group's maximum sorted position. For example:

```text
[10, 20, 20, 30]  becomes  [1, 3, 3, 4]
```

For a metric where larger values are preferred, define:

```text
r_{ij} = rank(v_{1j}, ..., v_{Nj})_i
```

For a metric where smaller values are preferred, reverse the order before ranking:

```text
r_{ij} = rank(-v_{1j}, ..., -v_{Nj})_i
```

Thus every rank satisfies `1 ≤ r_{ij} ≤ N`, and equal raw values get equal ranks.

Collecting all ranks gives the integer matrix:

```text
X = [r_{ij}] ∈ Z^(N×m)
```

Row `X_i` is the rank vector for offer `i`. Metric order comes from insertion order in `qos_priority`; the same order is used for directions, weights, encryption, and decryption.

### 2. Convert weights to integers

IPFE evaluates integer vectors, so each configured weight is scaled and rounded:

```text
w'_j = round(scale · w_j)
```

The default `scale` is `10`. The scaled weight vector is:

```text
w' = (w'_1, ..., w'_m)
```

Weights must be finite and nonnegative. Rounding can turn a very small weight into zero. The wrapper does not normalize weights; multiplying all weights by the same positive factor preserves score ordering.

### 3. Plaintext score and ranking

The score for offer `i` is the weighted inner product:

```text
S_i = X_i · w'
    = Σ(j=1..m) X_{ij} w'_j
```

In matrix form:

```text
S = X w'
```

Offers are ordered by descending `S_i`. NumPy's stable input order determines the order of equal final scores.

### 4. IPFE evaluation

The wrapper creates an IPFE key for exactly `m` dimensions:

```text
key = FeDDH.generate(m)
sk  = FeDDH.keygen(w', key)
```

For each rank vector `X_i`, it creates a ciphertext:

```text
c_i = FeDDH.encrypt(X_i, key)
```

Decrypting with the functional key reveals only the requested inner product:

```text
FeDDH.decrypt(c_i, key.get_public_key(), sk) = X_i · w' = S_i
```

Therefore plaintext and IPFE produce the same integer score and the same ranking, provided they use the same offer set, metric order, directions, scaled weights, key, and ciphertext batch.

### 5. Decryption bound

Because `1 ≤ X_{ij} ≤ N` and weights are nonnegative:

```text
0 ≤ S_i ≤ S_max
S_max = N · Σ(j=1..m) w'_j
```

The wrapper gives the IPFE library the search interval:

```text
(0, S_max + 1)
```

`EncryptedBatch` stores the original `N` and `S_max`, so decrypting a slice still uses the bound for the original offer set. This matters because a slice contains fewer ciphertexts but its individual scores can still be as large as the original maximum. The wrapper rejects signed 64-bit overflow before scoring or encryption.

## References

- [pymife package](https://pypi.org/project/pymife/)
- [DDH-based inner-product functional encryption construction](https://eprint.iacr.org/2015/017.pdf)
