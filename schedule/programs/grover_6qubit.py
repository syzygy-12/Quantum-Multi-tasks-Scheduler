from qiskit import QuantumCircuit
import qiskit.qpy as qpy

# Create a quantum circuit with 6 qubits
qc = QuantumCircuit(6)

# Initial Hadamard gates
qc.h(range(6))

# Oracle for the state |111111>
qc.mcx(list(range(5)), 5)

# Diffuser
qc.h(range(6))
qc.x(range(6))
qc.h(5)
qc.mcx(list(range(5)), 5)
qc.h(5)
qc.x(range(6))
qc.h(range(6))

# Save the circuit to a .qpy file
with open('C:/Users/syzygy/Desktop/file/hack/schedule/circuits/grover_6qubit.qpy', 'wb') as f:
    qpy.dump(qc, f)
