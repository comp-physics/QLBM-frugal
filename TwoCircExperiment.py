from qiskit.circuit import QuantumCircuit
from typing import List, Optional, Sequence
from qiskit.providers.backend import Backend
from qiskit_experiments.framework import BaseExperiment, Options

import numpy as np

from TwoCircuitUtils import vortCirc, streamCirc


M = 16 # lattices

dim = 2
dirs = 5

nlat = int(np.ceil(np.log2(M)))
nlinks = int(np.ceil(np.log2(dirs)))

vorticity = np.zeros((M,M))
streamfunction = np.zeros((M,M))

w = (2/6,1/6,1/6,1/6,1/6)
e = (0,-1,1,1,-1) 
cs = np.sqrt(3)   ##speed of sound 
U = 1
lambdas = [np.arccos(i) for i in w]#streamfunction lambdas, can use adv-dif

class StreamCircExperiment(BaseExperiment):
    """Custom experiment class template."""

    def __init__(self,
                 physical_qubits: Sequence[int],
                 analysis=None,
                 backend: Optional[Backend] = None):
        """Initialize the experiment."""
        super().__init__(physical_qubits,
                         backend = backend)

    def circuits(self) -> List[QuantumCircuit]:
        """Generate the list of circuits to be run."""

        streamfunction = np.zeros((M,M))
        vorticity = np.zeros((M,M))

        sCirc = streamCirc()
        
        circuits = []
        # Generate circuits and populate metadata here
        circuits.append(sCirc)
        return circuits

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """Set default experiment options here."""
        options = super()._default_experiment_options()
        options.update_options(
            dummy_option = None,
        )
        return options
    
class VortCircExperiment(BaseExperiment):
    """Custom experiment class template."""

    def __init__(self,
                 physical_qubits: Sequence[int],
                 analysis=None,
                 backend: Optional[Backend] = None):
        """Initialize the experiment."""
        super().__init__(physical_qubits,
                         backend = backend)

    def circuits(self) -> List[QuantumCircuit]:
        """Generate the list of circuits to be run."""

        streamfunction = np.zeros((M,M))
        vorticity = np.zeros((M,M))

        vCirc = vortCirc(streamfunction)
        
        circuits = []
        # Generate circuits and populate metadata here
        circuits.append(vCirc)
        return circuits

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """Set default experiment options here."""
        options = super()._default_experiment_options()
        options.update_options(
            dummy_option = None,
        )
        return options