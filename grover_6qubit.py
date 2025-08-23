
from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram
import numpy as np

# --- Configuration ---
N_QUBITS = 6
# Let's choose a secret state to search for.
# For example, |110101> which corresponds to the integer 53.
# The oracle will be built to recognize this specific state.
SECRET_STATE = '110101'

# --- 1. Oracle Definition ---
# The oracle marks the SECRET_STATE by flipping its phase.
# For a state |x>, O|x> = (-1)^{f(x)}|x>
# where f(x) = 1 if x is the secret state, and f(x) = 0 otherwise.
# This is implemented using a multi-controlled Z gate (or X-CZ-X pattern).

oracle = QuantumCircuit(N_QUBITS, name='Oracle')

# To flip the phase of |110101>, we apply an X gate to qubits
# where the secret state has a '0' (qubits 0 and 2).
# This transforms the target state to |111111>.
for qubit, bit in enumerate(reversed(SECRET_STATE)):
    if bit == '0':
        oracle.x(qubit)

# Now, apply a multi-controlled Z gate on all qubits.
# This flips the phase of |111111>.
oracle.h(N_QUBITS - 1)
oracle.mcx(list(range(N_QUBITS - 1)), N_QUBITS - 1)
oracle.h(N_QUBITS - 1)

# Finally, undo the X gates to return to the original basis.
for qubit, bit in enumerate(reversed(SECRET_STATE)):
    if bit == '0':
        oracle.x(qubit)

# --- 2. Diffuser (Amplitude Amplification) Definition ---
# The diffuser amplifies the amplitude of the marked state.
# It performs a reflection about the average amplitude.
# D = 2|s><s| - I, where |s> is the equal superposition state.

diffuser = QuantumCircuit(N_QUBITS, name='Diffuser')

# Apply Hadamard gates to all qubits to move from |s> to |0...0>
diffuser.h(range(N_QUBITS))
# Apply X gates to all qubits to move from |0...0> to |1...1>
diffuser.x(range(N_QUBITS))

# Apply a multi-controlled Z gate
diffuser.h(N_QUBITS - 1)
diffuser.mcx(list(range(N_QUBITS - 1)), N_QUBITS - 1)
diffuser.h(N_QUBITS - 1)

# Undo the X and H gates
diffuser.x(range(N_QUBITS))
diffuser.h(range(N_QUBITS))


# --- 3. Build the Full Grover Circuit ---
# Determine the optimal number of iterations
# For N = 2^n items, optimal iterations is approx. (pi/4) * sqrt(N)
num_iterations = int(np.floor(np.pi / 4 * (2**(N_QUBITS/2))))

# Create the main circuit
grover_circuit = QuantumCircuit(N_QUBITS, N_QUBITS)

# a. Initialize state to uniform superposition |s>
grover_circuit.h(range(N_QUBITS))
grover_circuit.barrier()

# b. Apply Grover iterations
for _ in range(num_iterations):
    grover_circuit.append(oracle, range(N_QUBITS))
    grover_circuit.append(diffuser, range(N_QUBITS))
    grover_circuit.barrier()

# c. Measure the qubits
grover_circuit.measure(range(N_QUBITS), range(N_QUBITS))

# --- 4. Simulation ---
# Use the AerSimulator
simulator = Aer.get_backend('aer_simulator')
# Execute the circuit
result = execute(grover_circuit, simulator, shots=1024).result()
counts = result.get_counts(grover_circuit)

# --- 5. Display Results ---
print(f"Grover's Algorithm for {N_QUBITS} Qubits")
print(f"Secret State to find: |{SECRET_STATE}>")
print(f"Optimal number of iterations: {num_iterations}")
print("\\n--- Simulation Results ---")
print(counts)

# Find and print the most frequent result
most_frequent_state = max(counts, key=counts.get)
print(f"\\nMost frequent result is |{most_frequent_state}> with {counts[most_frequent_state]} counts.")

# You can also visualize the results
# plot_histogram(counts).show()

# To see the circuit diagram, uncomment the line below
# print(grover_circuit.draw())

