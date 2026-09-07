from mife.single.selective.ddh import FeDDH
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from numbers import Integral
from uuid import uuid4
import numpy as np
import logging

log = logging.getLogger('ipfe')


class PrinterLogger:
    def log(self, msg):
        print(msg)


def _borda_with_ties(values):
    """Borda ranks with ties assigned their maximum sorted position.
    Tied values get the *maximum position* in their sorted group (+1).
    e.g. [10, 20, 20, 30] -> [1, 3, 3, 4]
    """
    arr = np.array(values, dtype=float)
    if arr.ndim != 1 or not np.all(np.isfinite(arr)):
        raise ValueError('QoS values must be a one-dimensional array of finite numbers')
    idx = np.argsort(arr, kind='stable')
    sorted_arr = arr[idx]
    ranks = np.zeros(len(arr), dtype=int)
    i = 0
    while i < len(sorted_arr):
        j = i
        while j < len(sorted_arr) and sorted_arr[j] == sorted_arr[i]:
            j += 1
        max_rank = j
        for k in range(i, j):
            ranks[idx[k]] = max_rank
        i = j
    return ranks


@dataclass(frozen=True)
class EncryptedBatch(Sequence):
    """In-memory ciphertexts with original bounds; slicing preserves metadata.

    Metadata detects accidental misuse, but is not authenticated for transport.
    """

    ciphertexts: tuple
    config: tuple
    key_id: str
    offer_count: int
    score_max: int

    def __len__(self):
        return len(self.ciphertexts)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return replace(self, ciphertexts=self.ciphertexts[index])
        return self.ciphertexts[index]


class IPFERanker:
    DEFAULT_DIRECTIONS = {
        'reliability': 'min', 'energy': 'min', 'bandwidth': 'max',
        'latency': 'min', 'price': 'min',
    }

    def __init__(self, qos_list, weights, scale: int = 10, directions=None):
        self._qos_list = tuple(qos_list)
        if not self.m:
            raise ValueError('qos_list must contain at least one metric')
        if any(not isinstance(qos, str) or not qos for qos in self.qos_list):
            raise ValueError('QoS metric names must be nonempty strings')
        if len(set(self.qos_list)) != self.m:
            raise ValueError('QoS metric names must be unique')
        if isinstance(scale, bool) or not isinstance(scale, Integral) or scale <= 0:
            raise ValueError('scale must be a positive integer')
        directions = {} if directions is None else directions
        if not isinstance(directions, Mapping):
            raise ValueError('directions must map metric names to min or max')
        if set(directions) - set(self.qos_list):
            raise ValueError('directions contains unknown QoS metrics')
        resolved = tuple(directions.get(qos, self.DEFAULT_DIRECTIONS.get(qos))
                         for qos in self.qos_list)
        if any(direction not in ('min', 'max') for direction in resolved):
            raise ValueError('Each new QoS metric requires an explicit min or max direction')
        self._directions = resolved
        weights = np.asarray(weights, dtype=float)
        if weights.ndim != 1 or len(weights) != self.m:
            raise ValueError('weights must contain one value per QoS metric')
        if not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError('weights must be finite nonnegative numbers')
        try:
            with np.errstate(over='ignore', invalid='ignore'):
                scaled = np.round(weights * float(scale))
        except OverflowError as exc:
            raise ValueError('scale is too large') from exc
        if not np.all(np.isfinite(scaled)) or np.any(scaled >= 2**63):
            raise ValueError('Scaled weights exceed the supported integer range')
        self._weights_scaled = tuple(int(weight) for weight in scaled)
        self._config = (self.qos_list, self._directions, self._weights_scaled, int(scale))
        self._key_id = uuid4().hex
        self.key = FeDDH.generate(self.m)
        self.sk = FeDDH.keygen(self.weights_scaled.tolist(), self.key)

    @property
    def qos_list(self):
        return self._qos_list

    @property
    def m(self):
        return len(self._qos_list)

    @property
    def weights_scaled(self):
        return np.array(self._weights_scaled, dtype=np.int64)

    @classmethod
    def from_dataset(cls, data, scale: int = 10):
        priorities = data.get('qos_priority')
        if not isinstance(priorities, Mapping) or not priorities:
            raise ValueError('qos_priority must be a nonempty metric-to-weight mapping')
        return cls(priorities.keys(), list(priorities.values()), scale,
                   directions=data.get('qos_direction'))

    def _score_max(self, offer_count):
        score_max = offer_count * sum(self._weights_scaled)
        if score_max > np.iinfo(np.int64).max:
            raise ValueError('Maximum score exceeds the supported integer range')
        return score_max

    def rank(self, data):
        """Rank offers by weighted Borda. Return indices sorted descending."""
        X = self._borda_matrix(data)
        return np.argsort(-np.dot(X, self.weights_scaled))

    def encrypt_offers(self, data):
        """Encrypt rank vectors with original batch bounds and configuration."""
        X = self._borda_matrix(data)
        ciphertexts = tuple(FeDDH.encrypt(row.tolist(), self.key) for row in X)
        return EncryptedBatch(ciphertexts, self._config, self._key_id,
                              len(X), self._score_max(len(X)))

    def decrypt_scores(self, ciphertexts):
        """Decrypt a batch or batch slice using its original scoring bounds."""
        if not isinstance(ciphertexts, EncryptedBatch):
            if isinstance(ciphertexts, (list, tuple)) and not ciphertexts:
                return np.array([], dtype=np.int64)
            raise ValueError('Expected EncryptedBatch; keep metadata when selecting ciphertexts')
        if ciphertexts.config != self._config or ciphertexts.key_id != self._key_id:
            raise ValueError('Encrypted batch belongs to a different configuration or key')
        if (not isinstance(ciphertexts.offer_count, int) or ciphertexts.offer_count < len(ciphertexts)
                or ciphertexts.score_max != self._score_max(ciphertexts.offer_count)):
            raise ValueError('Invalid encrypted batch bounds')
        if any(len(c.c) != self.m for c in ciphertexts):
            raise ValueError('Ciphertext dimension does not match metric configuration')
        return np.array([
            FeDDH.decrypt(c, self.key.get_public_key(), self.sk, (0, ciphertexts.score_max + 1))
            for c in ciphertexts
        ], dtype=np.int64)

    def plaintext_scores(self, data):
        """Get raw Borda scores without encryption."""
        X = self._borda_matrix(data)
        return np.dot(X, self.weights_scaled)

    def _borda_matrix(self, data):
        """Build N x m Borda rank matrix from data dict.
        
        Directions select raw values (max) or negated values (min).
        """
        columns = []
        for qos in self.qos_list:
            if qos not in data:
                raise ValueError(f'Missing QoS metric: {qos}')
            raw = np.array(data[qos], dtype=float)
            if raw.ndim != 1 or not np.all(np.isfinite(raw)):
                raise ValueError(f'QoS metric {qos} must contain a one-dimensional array of finite numbers')
            columns.append(raw)
        N = len(columns[0])
        if any(len(raw) != N for raw in columns):
            raise ValueError('All QoS metrics must contain the same number of offers')
        self._score_max(N)
        X = np.zeros((N, self.m), dtype=int)
        for j, (direction, raw) in enumerate(zip(self._directions, columns)):
            if direction == 'max':
                X[:, j] = _borda_with_ties(raw)
            else:
                X[:, j] = _borda_with_ties(-raw)
        return X
