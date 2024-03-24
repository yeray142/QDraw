from typing import Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import EstimatorQNN


def quantum_feature_map(n: int) -> QuantumCircuit:
    """
    Create a custom quantum feature map
    :param n: number of qubits
    :return: Quantum Circuit
    """
    circuit = QuantumCircuit(n)
    features = ParameterVector("x", n)

    for i in range(n):
        circuit.rx(features[i], i)

    return circuit


def quantum_ansatz(n: int) -> QuantumCircuit:
    """
    Create a custom quantum ansatz
    :param n: number of qubits
    :return: Quantum Circuit
    """
    # Creamos un circuito cuántico con n qubits.
    circuit = QuantumCircuit(n)
    weights = ParameterVector("w", n * 3)

    # Asumiendo que 'weights' es una lista de valores de peso,
    # y la longitud es suficiente para aplicar Rz a cada qubit.
    for i in range(n):
        circuit.rz(weights[i], i)

    # Aplicar más rotaciones, que en la imagen parecen ser Ry
    for i in range(n):
        circuit.ry(weights[n + i], i)

    for i in range(n):
        circuit.rz(weights[2 * n + i], i)

    # Aplicar entrelazamiento con puertas CNOT
    for i in range(n - 1):
        circuit.cx(i, i + 1)
    circuit.cx(n - 1, 0)

    return circuit


def create_qnn(n: int) -> Tuple[EstimatorQNN, QuantumCircuit]:
    """
    Creates the Quantum Neural Network instance
    :param n: number of qubits
    :return: Tuple of EstimatorQNN and QuantumCircuit instances
    """
    feature_map = quantum_feature_map(n)
    ansatz = quantum_ansatz(n)

    qc = QuantumCircuit(n)
    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace=True)

    # REMEMBER TO SET input_gradients=True FOR ENABLING HYBRID GRADIENT BACKPROP
    qnn = EstimatorQNN(
        circuit=qc,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        input_gradients=True,
    )
    return qnn, qc