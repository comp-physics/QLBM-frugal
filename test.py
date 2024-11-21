from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit, schedule, transpile
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService
# from qiskit_serverless import distribute_task

from qiskit_ibm_runtime.fake_provider import FakeBrisbane

# @distribute_task(target={
#     "cpu":1
# })
# def transpile_remote(circuit, optimization_level, backend):
#     pass_manager = generate_preset_pass_manager(
#         optimization_level=optimization_level,
#         backend=service.backend(backend)
#     )
#     isa_circuit = pass_manager.run(circuit)
#     return isa_circuit

def main():

    service = QiskitRuntimeService()
    optimization_level = 3

    # basic bell state
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    backend = AerSimulator.from_backend(
        FakeBrisbane(),
        device="CPU",
        blocking_enable=True,
    )
    # qc_opt = transpile_remote(
    #     qc,
    #     optimization_level,
    #     backend
    # )

    # print(type(qc_opt))
    pm = generate_preset_pass_manager(
        backend=backend, 
        optimization_level=optimization_level
    )
    qc_opt = pm.run(qc, num_processes = 4)

    # qc_transpiled = transpile(qc, backend)
    # print(type(qc_transpiled))

    print(qc_opt.depth())
    print(qc_opt.count_ops())


if __name__=='__main__':
    main()