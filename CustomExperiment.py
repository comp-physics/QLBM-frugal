from numpy.random import default_rng, Generator
from qiskit import QuantumCircuit
from qiskit_experiments.framework import BaseExperiment
from CustomAnalysis import CustomAnalysis

class CustomExperiment(BaseExperiment):
    def __init__(
            self,
            circuit,
            label="Default",
            backend=None,
            num_samples=10_000,
            seed=None
    ): 
        physical_qubits = tuple(range(circuit.num_qubits))
        measured_qubits = tuple(range(circuit.num_qubits))

        analysis = CustomAnalysis()

        super().__init__(physical_qubits, analysis=analysis, backend=backend)

        self._circuit = circuit
        self._measured_qubits = measured_qubits

        valid_labels = ['vorticity', 'stream']
        if label in valid_labels:
            self._label = label
        else:
            raise ValueError(f"Invalid label type: {label}. Must be in {valid_labels}")

        self.set_experiment_options(num_samples=num_samples, seed=seed)

    def circuits(self):
        circuit = self._circuit
        label = self._label

        # apply a measurement at the end of the circuit
        circuit.measure_all()
        circuit.metadata['label'] = label

        circuits = []
        circuits.append(circuit)

    @classmethod
    def _default_experiment_options(cls):
        options = super()._default_experiment_options()
        options.num_samples = None
        options.seed = None
        return options