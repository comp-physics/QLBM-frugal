import argparse
import twoCircuitTools
import pickle as pkl
import numpy as np

from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime.fake_provider import FakeBrisbane
from qiskit import schedule

parser = argparse.ArgumentParser(description="Simon's algorithm using Qiskit.")
parser.add_argument('--nlattice', type=int, default=16, help='Number of lattice points')
parser.add_argument('--outdir', type=str, default="data/", help="Path to output file")

def main(M):
    service = QiskitRuntimeService()
    
    backend = FakeBrisbane()

    vorticity = np.zeros((M,M))
    streamfunction = np.zeros((M,M))

    sCirc = streamCirc(M)
    vCirc = vortCirc(streamfunction, M)
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
    sCirc_opt, vCirc_opt = pm.run([sCirc, vCirc])

    sCirc_sched = schedule(sCirc_opt, backend)
    vCirc_sched = schedule(vCirc_opt, backend)

    out = {
        "stream": {
            "depth":sCirc_opt.depth(),
            "count_ops":sCirc_opt.count_ops(),
            "runtime":sCirc_sched.duration*backend.dt*1e6
        },
        "vorticity":{
            "depth":vCirc_opt.depth(),
            "count_ops":vCirc_opt.count_ops(),
            "runtime":vCirc_sched.duration*backend.dt*1e6
        }
    }

    return out

if __name__ == '__main__':
    args = parser.parse_args()
    num_lattice_points = args.nlattice
    path = args.outdir

    results = main(M=num_lattice_points)

    with open(f"{path}{M}.pkl", "wb") as f:
        pkl.dump(results, f)