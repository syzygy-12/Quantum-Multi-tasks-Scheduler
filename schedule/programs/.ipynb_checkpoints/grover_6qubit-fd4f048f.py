
from qiskit import QuantumCircuit
import os

# Create a 6-qubit quantum circuit
n = 6
qc = QuantumCircuit(n)

# Define the marked state (e.g., |111111>)
marked_state = '111111'

# Apply Hadamard gates to all qubits
for qubit in range(n):
    qc.h(qubit)

# --- Oracle ---
# Apply X gates to qubits corresponding to 0 in the marked state (none in this case)
# qc.x([i for i, bit in enumerate(marked_state) if bit == '0'])

# Apply a multi-controlled Z gate
qc.h(n-1)
qc.mct(list(range(n-1)), n-1)
qc.h(n-1)

# Apply X gates again to reverse the process (none in this case)
# qc.x([i for i, bit in enumerate(marked_state) if bit == '0'])

# --- Diffuser ---
for qubit in range(n):
    qc.h(qubit)
for qubit in range(n):
    qc.x(qubit)

qc.h(n-1)
qc.mct(list(range(n-1)), n-1)
qc.h(n-1)

for qubit in range(n):
    qc.x(qubit)
for qubit in range(n):
    qc.h(qubit)

# Save the circuit to a .qpy file
from qiskit import qpy
output_dir = r"C:\Users\syzygy\Desktop\file\hack\schedule\circuits"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
file_path = os.path.join(output_dir, "grover_6qubit.qpy")
with open(file_path, "wb") as f:
    qpy.dump(qc, f)

print(f"Circuit saved to {file_path}")
