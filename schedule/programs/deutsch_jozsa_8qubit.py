from qiskit import QuantumCircuit
import qiskit.qpy as qpy

# n = 7, so we need n+1 = 8 qubits
n = 7
# Create a quantum circuit with n+1 qubits and n classical bits
qc = QuantumCircuit(n + 1, n)

# Prepare the ancilla qubit in state |->
qc.x(n)
qc.h(n)

# Apply Hadamard gates to the input qubits
qc.h(range(n))

# Oracle for a balanced function (f(x) = x_0)
# This oracle flips the ancilla qubit if the first input qubit is 1
qc.cx(0, n)

# Apply Hadamard gates to the input qubits again
qc.h(range(n))

# Measure the input qubits
qc.measure(range(n), range(n))

# Save the circuit to a .qpy file
with open('C:/Users/syzygy/Desktop/file/hack/schedule/circuits/deutsch_jozsa_8qubit.qpy', 'wb') as f:
    qpy.dump(qc, f)
