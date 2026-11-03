import pennylane as qml
import torch
import numpy as np

n = 4
dim = 1 << n
beta = 0.02
feats = [0.5, 0.4, 0.3, 0.2]
edges = [(0, 1), (1, 2), (2, 3)]
# theta: 1 + 2*n = 9 params
theta = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# 1. PennyLane default.mixed
dev = qml.device('default.mixed', wires=n)

@qml.qnode(dev, interface='torch')
def circuit_pl(x, th):
    # Encoding
    for w in range(n):
        qml.RY(np.pi * x[w], wires=w)
        if beta > 0:
            qml.DepolarizingChannel(beta, wires=w)
    # Graph blocks
    for u, v in edges:
        qml.CNOT(wires=[u, v])
        if beta > 0:
            qml.DepolarizingChannel(beta, wires=v)
        qml.RZ(th[0], wires=v)
        if beta > 0:
            qml.DepolarizingChannel(beta, wires=v)
        qml.CNOT(wires=[u, v])
        if beta > 0:
            qml.DepolarizingChannel(beta, wires=v)
    # Trainable single-qubit
    for w in range(n):
        qml.RX(th[1 + w], wires=w)
        qml.RZ(th[1 + n + w], wires=w)
        if beta > 0:
            qml.DepolarizingChannel(beta, wires=w)
    return tuple(qml.expval(qml.PauliZ(w)) for w in range(n))

res_pl = [float(x.item()) for x in circuit_pl(feats, theta)]

# 2. Our simulator with PennyLane convention
# In PennyLane: 1q channel is:
# (1 - beta)*rho + (beta/3)*(X rho X + Y rho Y + Z rho Z)
X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)
Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)

def embed_1q(op, q):
    left = torch.eye(1 << q, dtype=torch.complex128) if q > 0 else None
    right = torch.eye(1 << (n - 1 - q), dtype=torch.complex128) if (n - 1 - q) > 0 else None
    res = op
    if left is not None: res = torch.kron(left, res)
    if right is not None: res = torch.kron(res, right)
    return res

paulis = {'X': [embed_1q(X, q) for q in range(n)],
          'Y': [embed_1q(Y, q) for q in range(n)],
          'Z': [embed_1q(Z, q) for q in range(n)]}

def ry_mat(t):
    c, s = np.cos(t/2), np.sin(t/2)
    return torch.tensor([[c, -s], [s, c]], dtype=torch.complex128)
def rx_mat(t):
    c, s = np.cos(t/2), np.sin(t/2)
    return torch.tensor([[c, -1j*s], [-1j*s, c]], dtype=torch.complex128)
def rz_mat(t):
    return torch.tensor([[np.exp(-1j*t/2), 0], [0, np.exp(1j*t/2)]], dtype=torch.complex128)

def apply_noise_1q_pl(rho, q, p):
    px, py, pz = paulis['X'][q], paulis['Y'][q], paulis['Z'][q]
    return (1.0 - p) * rho + (p / 3.0) * (px @ rho @ px + py @ rho @ py + pz @ rho @ pz)

cnot_perms = {}
for u in range(n):
    for v in range(n):
        if u != v:
            idx = torch.arange(dim)
            ctrl = (idx >> (n - 1 - u)) & 1
            cnot_perms[(u, v)] = idx ^ (ctrl << (n - 1 - v))

rho = torch.zeros(dim, dim, dtype=torch.complex128)
rho[0, 0] = 1.0

# 1. Encoding
for q in range(n):
    U = embed_1q(ry_mat(np.pi * feats[q]), q)
    rho = U @ rho @ U.conj().T
    if beta > 0:
        rho = apply_noise_1q_pl(rho, q, beta)

# 2. Entangling
for u, v in edges:
    p = cnot_perms[(u, v)]
    rho = rho[p, :][:, p]
    if beta > 0:
        rho = apply_noise_1q_pl(rho, v, beta)
    U = embed_1q(rz_mat(theta[0]), v)
    rho = U @ rho @ U.conj().T
    if beta > 0:
        rho = apply_noise_1q_pl(rho, v, beta)
    rho = rho[p, :][:, p]
    if beta > 0:
        rho = apply_noise_1q_pl(rho, v, beta)

# 3. RX then RZ
for q in range(n):
    U_rx = embed_1q(rx_mat(theta[1 + q]), q)
    rho = U_rx @ rho @ U_rx.conj().T
    U_rz = embed_1q(rz_mat(theta[1 + n + q]), q)
    rho = U_rz @ rho @ U_rz.conj().T
    if beta > 0:
        rho = apply_noise_1q_pl(rho, q, beta)

# Readout: individual <Z_w>
res_ours = []
for q in range(n):
    res_ours.append(float(torch.trace(paulis['Z'][q] @ rho).real.item()))

print('PennyLane outputs:', [round(x, 10) for x in res_pl])
print('Our simulator:    ', [round(x, 10) for x in res_ours])
diffs = [abs(a - b) for a, b in zip(res_pl, res_ours)]
max_diff = max(diffs)
print(f'Max diff: {max_diff:.2e}')
assert max_diff < 1e-12, 'Exceeded tolerance!'
print('EXACT MATCH TO 1e-12!')
