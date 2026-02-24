# ASRR-IPFE
A Simple Resource Ranking using Inner-Product Functional Encryption

This prototype ranks a set of resource offers using a **Borda-style scoring rule** and reproduces the same ranking using **inner-product functional encryption (IPFE)** based on a DDH-type scheme from `pymife`.

The goal is to show that:
- Plaintext weighted Borda ranking and
- IPFE-based inner-product evaluation

produce identical ranking results, while the IPFE variant can be used in a setting where feature vectors are encrypted.

---

## 1. Environment Setup

### 1.1. Requirements

You need Python 3 and `pip`:

```bash
python3 -V
pip -V
```

If you do not have them installed, check the official Python packaging guide on installing Python and `pip`.

Create and activate a virtual environment (recommended):

```bash
python3 -m venv venv
source venv/bin/activate
```


### 1.2. Install Dependencies

Install the required packages:

```bash
pip install -r requirements.txt
```

`requirements.txt` contains:

```text
numpy
pymife
```


---

## 2. Input Data Format

The script expects a JSON dataset at:

```text
dataset/resource-offers-ranking-format-sample.json
```

This file must define:

- Per-offer QoS arrays (all of length $N$):
    - `reliability`
    - `energy`
    - `bandwidth`
    - `latency`
    - `price`
- A QoS weight dictionary:

```json
"qos_priority": {
  "reliability": <float range in [0.0, 1.0]>,
  "energy": <float range in [0.0, 1.0]>,
  "bandwidth": <float range in [0.0, 1.0]>,
  "latency": <float range in [0.0, 1.0]>,
  "price": <float range in [0.0, 1.0]>
}
```


Here:

- $N$: number of offers (resources),
- `m=len(qos_priority)`, that is equal to the number of QoS objectives.

The script loads and normalizes the data as follows:

```python
data_path = 'dataset/resource-offers-ranking-format-sample.json'
keys_to_count = ['reliability', 'energy', 'bandwidth', 'latency', 'price']
weights_key = 'qos_priority'

with open(data_path, 'r') as f:
    data = json.load(f)

N = len(data[keys_to_count])           # number of offers
for k in keys_to_count:
    data[k] = np.array(data[k])[:N].tolist()

qos_list = list(data[weights_key].keys()) # QoS keys
m = len(qos_list)                         # number of QoS objectives
```


---

## 3. Borda Ranking with Tie Handling

### 3.1. `argsort_with_ties`

The core primitive is a Borda-style ranking function that maps raw QoS values to integer ranks $\{1,\dots,N\}$ with **dense ties**:

- Values are stably sorted.
- All equal values receive the same rank.
- That rank is derived from the maximum index in their sorted group (0-based), plus 1.
- The smallest value gets rank 1, and the largest value gets rank $N$ if unique.

Conceptually, for values $v \in \mathbb{R}^N$:

1. Compute stable sort indices:

$$
\text{idx} = \text{argsort}(v)
$$

2. Work over $v[\text{idx}]$; for a tie block at sorted positions $k, \dots, \ell$:
    - All elements in that block get rank $\ell + 1$.

Implementation:

```python
def argsort_with_ties(values)
```


### 3.2. Direction of Preference

The Borda mapping must reflect whether “higher is better” or “lower is better” for each QoS:

- For `bandwidth`: **higher is better**.
- For all other QoS (reliability, energy, latency, price): **lower is better**.

To achieve this:

- For higher-is-better:

$$
r_j = \texttt{argsort-with-ties}(v_j)
$$

- For lower-is-better:

$$
r_j = \texttt{argsort-with-ties}(-v_j)
$$

The script constructs an integer rank matrix $X \in \mathbb{Z}^{N \times m}$:

```python
x = np.zeros((N, m), dtype=int)
for j, qos in enumerate(qos_list):
    raw = np.array(data[qos])
    if qos == 'bandwidth':
        x[:, j] = argsort_with_ties(raw)     # higher is better
    else:
        x[:, j] = argsort_with_ties(-raw)    # lower is better
```

Each row $x_i$ (shape $(m,)$) is the Borda rank vector for offer $i$, i.e.:

$$
x_i = (r_1[i], \dots, r_m[i]) \in \{1,\dots,N\}^m
$$

---

## 4. Plaintext Weighted Borda Scoring

### 4.1. Weights and Fixed-Point Scaling

QoS weights are read from the JSON:

```python
weights = np.array([data[weights_key][qos] for qos in qos_list])
```

To make them integer-compatible with the DDH IPFE scheme, they are scaled by a fixed factor $P = 10$:

$$
w'_j = \text{round}(P \cdot w_j), \quad w' \in \mathbb{Z}^m
$$

Code:

```python
P = 10  # scaling factor for fixed-point representation
weights_scaled_int = np.round(
    [data[weights_key][qos] * P for qos in qos_list]
).astype(int)
```

This preserves ordering: scaling all weights by the same positive constant does not change which offers have larger scores.

### 4.2. Score Computation and Ranking

The plaintext Borda score for offer $i$ is the inner product:

$$
S_i^{\text{plain}} = \sum_{j=1}^m w'_j \cdot X[i,j].
$$

In matrix form:

$$
S^{\text{plain}} = X \cdot w', \quad S^{\text{plain}} \in \mathbb{Z}^N.
$$

Implementation:

```python
start = time.time()
scores_plain = np.dot(x, weights_scaled_int)  # shape (N,)
ranking_plain = np.argsort(-scores_plain)     # descending
plain_time = time.time() - start

top_plain = ranking_plain
print(f"Plaintext top offer: {top_plain}, time: {plain_time:.4f}s")
```

`ranking_plain[0]` is the index of the offer with the highest weighted Borda score.

---

## 5. IPFE-Based Ranking (DDH Scheme via `pymife`)

### 5.1. Scheme Initialization

The script uses the selective single-input DDH inner-product FE scheme from `pymife`:

```python
from mife.single.selective.ddh import FeDDH

key = FeDDH.generate(m)                         # master key (includes public and secret)
sk = FeDDH.keygen(weights_scaled_int.tolist(), key)  # functional key for w'
```

- `FeDDH.generate(m)` sets up parameters for vector length $m$.
- `FeDDH.keygen` derives a functional decryption key for the fixed integer weight vector $w'$.


### 5.2. Per-Offer Encryption and Decryption

For each offer $i$, the Borda rank vector $x_i \in \mathbb{Z}^m$ is encrypted:

$$
c_i = \text{FeDDH.Enc}_{\text{msk}}(x_i).
$$

Using the functional key for $w'$, the decryptor can recover only the inner product, not the raw vector:

$$
S_i^{\text{IPFE}} = \langle x_i, w' \rangle.
$$

This matches the plaintext score:

$$
S_i^{\text{IPFE}} = S_i^{\text{plain}}.
$$

Code:

```python
ipfe_scores = []
start = time.time()

for i in range(N):
    c = FeDDH.encrypt(x[i].tolist(), key)
    m_ipfe = FeDDH.decrypt(
        c,
        key.get_public_key(),
        sk,
        (0, 50000)  # bound on the score range
    )
    ipfe_scores.append(m_ipfe)

ranking_ipfe = np.argsort(-np.array(ipfe_scores))
ipfe_time = time.time() - start

print(f"IPFE top offer: {ranking_ipfe}, time: {ipfe_time:.4f}s")
```

The decryption bound `(0, 50000)` should cover the maximum possible score. A safe upper bound is:

$$
S_{\max} \approx m \cdot N \cdot \max(w'_j).
$$

### 5.3. Consistency with Plaintext Ranking

Both plaintext and IPFE compute the same integer expression:

$$
S_i^{\text{plain}} = S_i^{\text{IPFE}} = \sum_{j=1}^m w'_j X[i,j].
$$

Thus, `ranking_plain` and `ranking_ipfe` should coincide (up to ties from Borda ranks). The script prints the top-ranked offer under both methods and measures wall-clock times.

---

## 6. Logging and Debugging

Logging is controlled by a simple flag:

```python
# LOG = True
LOG = False

def print_log(*args):
    if LOG:
        print("[LOG]: ", *args)
```

Set `LOG = True` in `ranking.py` if you want to inspect:

- Raw QoS values per dimension,
- Borda ranks per dimension,
- Weights and scaled weights,
- Intermediate score vectors.

---

## 7. Running the Demo

Once the environment is prepared and the dataset is in place:

```bash
python3 ranking.py
```

Example output structure:

```text
Plaintext top offer: <index>, time: 0.0001s
IPFE top offer: <index>, time: 0.0XXXs
```

Where `<index>` is the 0-based index of the top-ranked offer.

---

## 8. References

This prototype relies on the `pymife` functional encryption library and its DDH-based inner-product scheme:

- `pymife` package: https://pypi.org/project/pymife/
- Selective secure DDH-based FE scheme (original construction):
https://eprint.iacr.org/2015/017.pdf

