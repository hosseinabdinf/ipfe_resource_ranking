import json
import numpy as np
from mife.single.selective.ddh import FeDDH
import time

# To enable/disable logging
# LOG = True
LOG = False
def print_log(*args):
    if LOG:
        print("[LOG]: ", *args)


data_path = 'dataset/resource-offers-ranking-format-sample.json'
keys_to_count = ['reliability', 'energy', 'bandwidth', 'latency', 'price']
weights_key = 'qos_priority'

# Load the resources file
with open(data_path, 'r') as f:
    data = json.load(f)
print_log("Data is loaded!")

N = len(data[keys_to_count[0]])  # Number of offers
print_log(f"Number of offers: {N}")

# Load all the offers metrics
for k in keys_to_count:
    data[k] = np.array(data[k])[:N].tolist()

# Load the weights for the QoS metrics
qos_list = list(data[weights_key].keys())
m = len(qos_list)


def argsort_with_ties(values):
    """
    Borda ranks with dense ties (no gaps)
    Tied values get the *maximum position* in their sorted group (+1).
    Example: [10, 20, 20, 30] → [1, 3, 3, 4]
    """

    sorted_indices = np.argsort(values, kind='stable')
    ranks = np.zeros(len(values), dtype=int)
    sorted_values = np.array(values)[sorted_indices]
    rank = 0
    while rank < len(sorted_values):
        same_value_indices = [i for i in range(rank, len(sorted_values)) if sorted_values[i] == sorted_values[rank]]
        max_rank = max(same_value_indices)
        for i in same_value_indices:
            ranks[sorted_indices[i]] = max_rank
        rank += len(same_value_indices)
    return ranks + 1


# Compute Borda ranks per QoS (x matrix: N x m)
x = np.zeros((N, m), dtype=int)
for j, qos in enumerate(qos_list):
    raw = np.array(data[qos])
    if qos == 'bandwidth':
        x[:, j] = argsort_with_ties(raw)
    else:
        x[:, j] = argsort_with_ties(-raw)
    
    print_log(f"Raw values for {qos}: \n {raw}")
    print_log(f"Borda ranks for {qos}: \n {x[:, j]}")

weights = np.array([data[weights_key][qos] for qos in qos_list])  # y vector (m,)
print_log(f"Weights: {weights}")

P = 10 # Scaling factor for fixed-point representation
weights_scaled_int = np.round([data[weights_key][qos] * P for qos in qos_list]).astype(int)
print_log(f"Scaled weights (w'): {weights_scaled_int}")

# PLAINTEXT calculation
start = time.time()
scores_plain = np.dot(x, weights_scaled_int)
ranking_plain = np.argsort(-scores_plain)  # descending
plain_time = time.time() - start
print_log(f"Ranking Plain: {ranking_plain}")

top_plain = ranking_plain[0]
print_log(f"Plain Scores: {scores_plain}")
print(f"Plaintext top offer: {top_plain}, time: {plain_time:.4f}s")

# IPFE (DDH)
key = FeDDH.generate(m)                     # Master key
c = FeDDH.encrypt(x[0].tolist(), key)       # Demo single; batch via loop in prod
sk = FeDDH.keygen(weights_scaled_int.tolist(), key)
# start = time.time()
# m_ipfe = FeDDH.decrypt(c, key.get_public_key(), sk, (0, 50000))  # Bound S_max~m*N*max_w
# ipfe_time = time.time() - start     
# print(f"IPFE score (offer 0): {m_ipfe}, time: {ipfe_time:.4f}s")

# Compute full IPFE ranking
ipfe_scores = []
start = time.time()
for i in range(N):
    c = FeDDH.encrypt(x[i].tolist(), key)
    m_ipfe = FeDDH.decrypt(c, key.get_public_key(), sk, (0, 50000))
    ipfe_scores.append(m_ipfe)
ranking_ipfe = np.argsort(-np.array(ipfe_scores))
ipfe_time = time.time() - start
print(f"IPFE top offer: {ranking_ipfe[0]}, time: {ipfe_time:.4f}s")

