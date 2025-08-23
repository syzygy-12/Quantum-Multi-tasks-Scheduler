
from qiskit import QuantumCircuit
import qiskit.qpy as qpy
import os

# Create a 6-qubit Grover's algorithm circuit
n = 6
circuit = QuantumCircuit(n)

# --- Oracle for marked state |111111> ---
def oracle(qc, n):
    # Marks the state |111111>
    qc.mcp(3.14159265359, list(range(n-1)), n-1) # Multi-controlled-pi rotation (MCZ)

# --- Diffuser (Amplitude Amplification) ---
def diffuser(qc, n):
    # Apply H-gates
    qc.h(range(n))
    # Apply X-gates
    qc.x(range(n))
    # Do multi-controlled-Z gate
    qc.mcp(3.14159265359, list(range(n-1)), n-1)
    # Apply X-gates
    qc.x(range(n))
    # Apply H-gates
    qc.h(range(n))

# 1. Initialize state to uniform superposition
circuit.h(range(n))
circuit.barrier()

# 2. Apply Grover iterations
# Number of iterations is approx. (pi/4)*sqrt(2^n)
# For n=6, sqrt(64) = 8. (pi/4)*8 = 2*pi ~= 6 iterations.
iterations = 6
for _ in range(iterations):
    oracle(circuit, n)
    circuit.barrier()
    diffuser(circuit, n)
    circuit.barrier()

# --- Save the circuit to a .qpy file ---
output_dir = r'C:\Users\syzygy\Desktop\file\hack\schedule\circuits'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

file_path = os.path.join(output_dir, 'grover_6q.qpy')
with open(file_path, 'wb') as f:
    qpy.dump(circuit, f)

print(f"Circuit saved to {file_path}")
