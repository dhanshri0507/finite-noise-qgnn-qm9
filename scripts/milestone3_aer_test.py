"""
Milestone 3 Final: Determine the actual root cause of gradient norm divergence.

Observations so far:
1. Forward pass at (alpha, theta2=0):
   - Aer=0.5728, custom=0.5311 at beta=0.1 (diff=0.0416)
   - The diff is CONSTANT across ALL theta2 values (0.0, 1e-10, 1e-6, 1e-3, 0.01, 0.1)
   => The divergence is NOT caused by identity-gate pruning (which would only 
      trigger at theta2=0, not at theta2=1e-6).

2. The diff is systematic and grows with beta:
   beta=0: diff=0
   beta=0.05: diff=0.022
   beta=0.10: diff=0.042
   beta=0.20: diff=0.072

HYPOTHESIS B: Different noise convention for the depolarizing channel:
  - Standard Qiskit: for 1-qubit depolarizing with parameter p:
    p_I = 1-p, p_X=p_Y=p_Z = p/3
    <Z>_shrink = (1-4p/3)
  - Our custom uses: p_error = beta
    rho -> (1-beta)*rho + (beta/3)*(X*rho*X + Y*rho*Y + Z*rho*Z)
    <Z>_shrink = (1-4*beta/3) ... same formula!
  
  But wait — after TWO gates (Ry + Rx):
    Aer (Rz/Rx are physical gates, noise applied after EACH physical gate):
      shrink = (1-4p/3)^2
    Our custom also applies twice. So same?

HYPOTHESIS C: Qiskit Aer uses DIFFERENT noise parameterization.
  Qiskit depolarizing_error(p, 1) with p=0.1:
    Actually implemented as: rho -> (1-p)*rho + p/3*(X+Y+Z pauli terms)
    This gives <Z>_shrink = 1 - 4p/3 = 1-0.133 = 0.867 per gate.
  
  For 2 gates: 0.867^2 = 0.751 (custom)
  For 1 gate:  0.867    = 0.867 (Aer -- if only 1 gate has noise)
  
  Let's check: at alpha=pi/4, <Z> = cos(pi/4) = 0.707
  Custom result: 0.531 / 0.707 = 0.751 -> 0.867^2 ✓ (2 noise applications)
  Aer result:   0.573 / 0.707 = 0.810 -> ???
  
  0.810 is NOT 0.867 (1 gate) = 0.867*0.933 two different shrinkages?
  Let's check: 0.810 ≈ (1-4*0.1/3) * (1-4*0.1/3) ... no, that's 0.751
  
  Actually 0.810 ≈ (1-0.1)^2 ... no, that's 0.81. YES!
  
  Hypothesis C confirmed: Aer applies (1-p)^N shrinkage (not (1-4p/3)^N).
  This corresponds to a DIFFERENT depolarizing convention where only the 
  "error happens" probability matters, not the per-Pauli probability.
  
  Standard: E(rho) = (1-p)rho + p*(mixed state) where mixed = I/2
    => <Z>_shrink = (1-p) - p*0 = (1-p)... wait
    E(rho) = (1-p)rho + p*I/2
    <Z> = (1-p)<Z> + p*Tr(Z*I/2) = (1-p)<Z> + 0 = (1-p)<Z>
    
  So Qiskit uses: E(rho) = (1-p)rho + p*I/2 form
    => shrinkage = (1-p) per gate
  
  Our custom uses: Pauli form: (1-p)rho + (p/3)(X*rho*X + Y*rho*Y + Z*rho*Z)
    => shrinkage = (1-4p/3) per gate
  
  At p=0.1:
    Aer: (1-0.1) = 0.9 per gate, (0.9)^2 = 0.81 for 2 gates ✓ matches 0.573/0.707=0.810
    Ours: (1-4*0.1/3) = 0.867 per gate, (0.867)^2 = 0.751 for 2 gates ✓ matches 0.531/0.707=0.751

CONCLUSION: The divergence is a NOISE CONVENTION MISMATCH:
  - Qiskit Aer: depolarizing_error(p) uses mixed-state form E(rho) = (1-p)rho + p*I/2
    => single-qubit shrinkage factor = (1-p)
  - Our custom: Pauli-sum form with (p/3) per Pauli term  
    => single-qubit shrinkage factor = (1-4p/3)
  
  At p=0.05: Aer=(1-0.05)=0.95, Ours=(1-4*0.05/3)=0.933 — DIFFERENT!
  
  For N gate noise applications (N depends on circuit depth and parameters):
    Aer gradient norm ~ (1-beta)^N
    Our gradient norm ~ (1-4*beta/3)^N
  
  The DEBUG_LOG must be REVISED: identity-gate pruning is NOT the cause.
  The cause is the different shrinkage factor per noisy gate:
    Aer: (1-beta) vs Ours: (1-4*beta/3)
  For beta>0: (1-beta) > (1-4*beta/3) => Aer shrinks LESS => Aer gradient norm is LARGER.
  Ratio: Aer/Our = [(1-beta)/(1-4*beta/3)]^N per gate.
"""

import sys
import os
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

ROOT_DIR = Path(__file__).resolve().parent.parent
OUT_FILE = ROOT_DIR / "results" / "raw" / "milestone3_aer_identity_test.json"
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)


def run_convention_test():
    print("=" * 70)
    print("Milestone 3 FINAL: Noise Convention Mismatch Hypothesis")
    print("=" * 70)

    results = {}

    try:
        from qiskit import QuantumCircuit
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel, depolarizing_error
        HAS_QISKIT = True
    except ImportError as e:
        print(f"Qiskit unavailable: {e}")
        return

    aer_sim = AerSimulator(method='density_matrix')

    def run_aer_1gate(theta1, beta):
        """1Q circuit: Ry(theta1) + depol -> <Z>"""
        qc = QuantumCircuit(1)
        noise_model = NoiseModel()
        if beta > 0:
            err1q = depolarizing_error(beta, 1)
            noise_model.add_all_qubit_quantum_error(err1q, ['ry'])
        qc.ry(theta1, 0)
        qc.save_density_matrix()
        job = aer_sim.run(qc, noise_model=noise_model, shots=None)
        dm = job.result().data()['density_matrix'].data
        return float(np.real(dm[0, 0] - dm[1, 1]))

    def run_custom_1gate_mixed(theta1, beta):
        """Custom: Ry + mixed-state depolarizing E(rho)=(1-p)rho + p*I/2"""
        rho = np.array([[1.0+0j, 0+0j], [0+0j, 0+0j]])
        c, s = np.cos(theta1/2), np.sin(theta1/2)
        Ry = np.array([[c, -s], [s, c]], dtype=complex)
        rho = Ry @ rho @ Ry.conj().T
        if beta > 0:
            rho = (1 - beta) * rho + beta * np.eye(2, dtype=complex) / 2
        return float(np.real(rho[0, 0] - rho[1, 1]))

    def run_custom_1gate_pauli(theta1, beta):
        """Custom: Ry + Pauli-sum depolarizing (1-beta)rho+(beta/3)(X+Y+Z terms)"""
        rho = np.array([[1.0+0j, 0+0j], [0+0j, 0+0j]])
        c, s = np.cos(theta1/2), np.sin(theta1/2)
        Ry = np.array([[c, -s], [s, c]], dtype=complex)
        rho = Ry @ rho @ Ry.conj().T
        if beta > 0:
            X = np.array([[0,1],[1,0]], dtype=complex)
            Y = np.array([[0,-1j],[1j,0]], dtype=complex)
            Z = np.array([[1,0],[0,-1]], dtype=complex)
            rho = (1-beta)*rho + (beta/3)*(X@rho@X + Y@rho@Y + Z@rho@Z)
        return float(np.real(rho[0, 0] - rho[1, 1]))

    alpha = np.pi / 4   # Ry(pi/4): <Z>=cos(pi/4)=0.70711 without noise
    betas = [0.0, 0.05, 0.10, 0.20]

    print(f"\n1-gate circuit: Ry(pi/4) + depol -> <Z>")
    print(f"Noiseless <Z> = cos(pi/4) = {np.cos(np.pi/4):.8f}")
    print()
    print(f"{'beta':>6} | {'f_aer':>12} | {'f_mixed':>12} | {'f_pauli':>12} | "
          f"{'shrink_aer':>11} | {'shrink_mixed':>13} | {'shrink_pauli':>13}")
    print("-" * 90)

    for beta in betas:
        fa = run_aer_1gate(alpha, beta)
        fm = run_custom_1gate_mixed(alpha, beta)
        fp = run_custom_1gate_pauli(alpha, beta)
        z0 = np.cos(alpha)
        sa = fa / z0
        sm = fm / z0
        sp = fp / z0
        theory_mixed = 1 - beta
        theory_pauli = 1 - 4*beta/3
        print(f"{beta:>6.3f} | {fa:>12.8f} | {fm:>12.8f} | {fp:>12.8f} | "
              f"{sa:>11.6f} | {sm:>13.6f} | {sp:>13.6f}")
        results[f"beta_{beta}"] = {
            "aer": fa, "mixed_form": fm, "pauli_form": fp,
            "shrink_aer": sa, "shrink_mixed": sm, "shrink_pauli": sp,
            "theory_mixed": theory_mixed, "theory_pauli": theory_pauli,
            "aer_matches_mixed": abs(sa - theory_mixed) < 0.01,
            "aer_matches_pauli": abs(sa - theory_pauli) < 0.01,
        }

    print()
    print("--- Theory check ---")
    print(f"{'beta':>6} | {'theory_(1-b)':>13} | {'theory_(1-4b/3)':>16} | {'aer_matches':>12}")
    print("-" * 55)
    for beta in betas:
        r = results[f"beta_{beta}"]
        m = "MIXED" if r["aer_matches_mixed"] else ("PAULI" if r["aer_matches_pauli"] else "NEITHER")
        print(f"{beta:>6.3f} | {1-beta:>13.8f} | {1-4*beta/3:>16.8f} | {m:>12}")

    # Conclusion
    print()
    aer_is_mixed = all(results[f"beta_{b}"]["aer_matches_mixed"] for b in betas if b > 0)
    aer_is_pauli = all(results[f"beta_{b}"]["aer_matches_pauli"] for b in betas if b > 0)

    if aer_is_mixed:
        conclusion = (
            "CONFIRMED: Qiskit Aer depolarizing_error(p) uses the MIXED-STATE form "
            "E(rho)=(1-p)*rho + p*I/2, giving per-gate shrinkage (1-p). "
            "Our custom simulator uses the PAULI-SUM form with shrinkage (1-4p/3). "
            "Since (1-p) > (1-4p/3) for 0<p<3/4, Aer shrinks the quantum state LESS, "
            "so Aer gradient norms are LARGER than ours at beta>0. "
            "The DEBUG_LOG Entry 005 must be REVISED: the cause is a noise CONVENTION "
            "MISMATCH (mixed-state vs Pauli-sum depolarizing), NOT identity-gate pruning. "
            "Per gate ratio: Aer/Our = (1-beta)/(1-4*beta/3). "
            "IMPORTANT: Both are valid depolarizing channels; they are equivalent up to "
            "a reparametrization p_eff = 4*beta/3. The 4/3 factor maps between them."
        )
    elif aer_is_pauli:
        conclusion = "CONFIRMED: Both use Pauli-sum form. The divergence has a different cause."
    else:
        conclusion = "NEITHER HYPOTHESIS CONFIRMED. Further investigation required."

    results["conclusion"] = conclusion
    print(f"CONCLUSION: {conclusion}")

    # Save complete results
    with open(OUT_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nFull results saved to {OUT_FILE}")
    return results


if __name__ == "__main__":
    run_convention_test()
