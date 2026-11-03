import qiskit
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
import torch
import numpy as np

n = 4
dim = 1 << n
beta = 0.02
feats = [0.5, 0.4, 0.3, 0.2]
edges = [(0, 1), (1, 2), (2, 3)]
theta = [0.1, 0.2, 0.3, 0.4]

# Qiskit Aer
qc = QuantumCircuit(n)
for q in range(n): qc.ry(float(feats[q]), q)
for u, v in edges:
    qc.cx(u, v)
    qc.rz(float(theta[v]), v)
    qc.cx(u, v)
for q in range(n): qc.rx(float(theta[q]), q)
obs = SparsePauliOp('Z' * n)
qc.save_expectation_value(obs, list(range(n)), label='expval')

nm = NoiseModel()
nm.add_all_qubit_quantum_error(depolarizing_error(beta, 1), ['ry', 'rz', 'rx'])
nm.add_all_qubit_quantum_error(depolarizing_error(beta, 2), ['cx'])
res_q = float(AerSimulator(method='density_matrix', noise_model=nm).run(qc).result().data(0)['expval'].real)

# Our simulator with exact Qiskit channel
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

def apply_noise_1q_qiskit(rho, q, p):
    px, py, pz = paulis['X'][q], paulis['Y'][q], paulis['Z'][q]
    return (1.0 - 0.75 * p) * rho + (p / 4.0) * (px @ rho @ px + py @ rho @ py + pz @ rho @ pz)

def apply_noise_2q_qiskit(rho, u, v, p):
    plist = ['X', 'Y', 'Z']
    pu = [paulis[x][u] for x in plist]
    pv = [paulis[x][v] for x in plist]
    out = (1.0 - (15.0 / 16.0) * p) * rho
    scale = p / 16.0
    rv_list = []
    for y in pv:
        rv = y @ rho @ y
        rv_list.append(rv)
        out = out + scale * rv
    for x in pu:
        out = out + scale * (x @ rho @ x)
    for rv in rv_list:
        for x in pu:
            out = out + scale * (x @ rv @ x)
    return out

cnot_perms = {}
for u in range(n):
    for v in range(n):
        if u != v:
            idx = torch.arange(dim)
            ctrl = (idx >> (n - 1 - u)) & 1
            cnot_perms[(u, v)] = idx ^ (ctrl << (n - 1 - v))

rho = torch.zeros(dim, dim, dtype=torch.complex128)
rho[0, 0] = 1.0

# 1. RY
for q in range(n):
    U = embed_1q(ry_mat(feats[q]), q)
    rho = U @ rho @ U.conj().T
    rho = apply_noise_1q_qiskit(rho, q, beta)

# 2. Entangling
for u, v in edges:
    p = cnot_perms[(u, v)]
    rho = rho[p, :][:, p]
    rho = apply_noise_2q_qiskit(rho, u, v, beta)
    U = embed_1q(rz_mat(theta[v]), v)
    rho = U @ rho @ U.conj().T
    rho = apply_noise_1q_qiskit(rho, v, beta)
    rho = rho[p, :][:, p]
    rho = apply_noise_2q_qiskit(rho, u, v, beta)

# 3. RX
for q in range(n):
    U = embed_1q(rx_mat(theta[q]), q)
    rho = U @ rho @ U.conj().T
    rho = apply_noise_1q_qiskit(rho, q, beta)

# Readout Z^n
Z_full = paulis['Z'][0]
for q in range(1, n):
    Z_full = Z_full @ paulis['Z'][q]
res_ours = torch.trace(Z_full @ rho).real.item()

print(f'Ours:   {res_ours:.14f}')
print(f'Qiskit: {res_q:.14f}')
diff = abs(res_ours - res_q)
print(f'Diff:   {diff:.2e}')
assert diff < 1e-12, 'Exceeded tolerance!'
print('EXACT MATCH TO 1e-12!')
