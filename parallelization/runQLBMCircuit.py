import argparse
import pickle as pkl
import numpy as np

import math

from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime.fake_provider import FakeBrisbane
from qiskit import schedule

import numpy as np
import qiskit as q
from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister
from qiskit.quantum_info.operators import Operator
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import Diagonal
import qiskit.circuit.library as qlib
from qiskit import transpile

from qiskit_aer import AerSimulator

import time

# GLOBAL VARIABLES
dim = 2
dirs = 5

w = (2/6,1/6,1/6,1/6,1/6)
e = (0,-1,1,-1,1) #center, left, right, down, up
cs = np.sqrt(3)   ##speed of sound 
U = 1

lambdas = [np.arccos(i) for i in w]#streamfunction lambdas, can use adv-dif

# for boundaries
def top_half(A):
    def convert(val):
        return val+1j*np.sqrt(1-val**2)
    func = np.vectorize(convert)
    return func(A)
def bottom_half(A):
    def convert(val):
        return val-1j*np.sqrt(1-val**2)
    func = np.vectorize(convert)
    return func(A)
def createADiag(M):
    A_diag = np.ones((M,M))
    A_diag[0] = 0
    A_diag[-1] = 0
    A_diag[:,0] = 0
    A_diag[:,-1] = 0
    A_diag = A_diag.flatten()
    
    B1_diag = top_half(A_diag)
    B2_diag = bottom_half(A_diag)
    return B1_diag, B2_diag

#for collision gates
def createLambda(oper, M, isC1 = True):
    nlat = int(np.ceil(np.log2(M)))
    nlinks = int(np.ceil(np.log2(dirs)))
    n = nlinks+dim*nlat+1

    ctrl = oper
    if oper >= 8:
        oper -= 8

    s = bin(ctrl)[2:]
    s = '0'*int((4-len(s))) + s
    nlambda = lambdas[oper]
    
    if not isC1:
        nlambda*=-1

    qc = QuantumCircuit(2+nlinks)
    tmp = QuantumCircuit(1)

    tmp.p(nlambda, 0)
    tmp.x(0)
    tmp.p(nlambda, 0)
    tmp.x(0)
    
    gate = tmp.to_gate(label = "Lambda"+str(ctrl)).control(nlinks+1,ctrl_state = s)
    
    cbits = [i for i in range(1,nlinks+2)]
    cbits.append(0)
    qc.append(gate,cbits)
    return qc

#prop gates
def rshift(n):
    circ = QuantumCircuit(n)
    for i in range(n):
        if i == n-1:
            circ.x(i)
        else:
            circ.mcx(list(range(i+1,n)), i)
    return circ

def lshift(n):
    circ = QuantumCircuit(n)
    for i in reversed(range(n)):
        if i == n-1:
            circ.x(i)
        else:
            circ.mcx(list(range(i+1,n)), i)
    return circ

def streamCirc(M):
    nlat = int(np.ceil(np.log2(M)))
    nlinks = int(np.ceil(np.log2(dirs)))
    n = nlinks+dim*nlat+1

    q = QuantumRegister(nlinks+dim*nlat+1,'q')
    a = AncillaRegister(1,'a')

    setup = QuantumCircuit(q)
    setup.add_register(a)

    h1 = qlib.HGate().control(1,ctrl_state = 0)

    cb = [a,0]
    cb.extend([i for i in range(dim*nlat, dim*nlat+nlinks+1)])

    setup.h(a)
    for i in range(dirs):
        setup.append(createLambda(i, M, True).to_gate(label = 'Lambda'+str(i)).control(1,ctrl_state = 0),cb)
    for i in range(8,13):
        setup.append(createLambda(i, M, True).to_gate(label = 'Lambda'+str(i)).control(1,ctrl_state = 0),cb)
    for i in range(dirs):
        setup.append(createLambda(i, M, False).to_gate(label = 'Lambda'+str(i)).control(1,ctrl_state = 1),cb)
    for i in range(8,13):
        setup.append(createLambda(i, M, False).to_gate(label = 'Lambda'+str(i)).control(1,ctrl_state = 1),cb)

    setup.h(q[n - 2])
    setup.h(a)
    
    setup.barrier()
    L1 = lshift(nlat).to_gate(label = "L").control(nlinks+1,ctrl_state = '0001')
    R1 = rshift(nlat).to_gate(label = "R").control(nlinks+1,ctrl_state = '0010')
    L2 = lshift(nlat).to_gate(label = "L").control(nlinks+1,ctrl_state = '0011')
    R2 = rshift(nlat).to_gate(label = "R").control(nlinks+1,ctrl_state = '0100')
    
    log_M = round(math.log2(M))

    shift_controls = [n - 4, n - 3, n - 2]
    one = shift_controls + [a[0]] + [i for i in reversed(range(log_M))]
    two = shift_controls + [a[0]] + [i for i in reversed(range(log_M, 2*log_M))]

    setup.append(L1, one)
    setup.append(R1, one)
    setup.append(L2, two)
    setup.append(R2, two)
    setup.barrier()


    # setup.append(h1,[11,8])
    # setup.append(h1,[11,9])
    # setup.append(h1,[11,10])
    # GENERALIZED
    for i in shift_controls:
        setup.h(q[i])
    # setup.barrier()

    # B = QuantumCircuit(QuantumRegister(9))
    # B1d, B2d = createADiag(M)
    # B = Diagonal(list(np.concatenate((B1d,B2d))))
    # # B.diagonal(list(np.concatenate((B1d,B2d))), qubit=[0,1,2,3,4,5,6,7,8])
    # B = B.to_gate(label='B')

    # setup.append(B,[q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8]])

    return setup

#calculating derivs and diagonal array
def uv(streamfunc, M):
    # u = streamfunction_y, v = -streamfunction_x
    # streamfunc = np.reshape(streamfunc,(M,M))
    v = (streamfunc[:,1:]-streamfunc[:,:-1])
    u = streamfunc[1:,:]-streamfunc[:-1,:]
    
    v = np.append(v,np.reshape(v[:,-1],(M,1)),axis=1)
    u = np.append(u,u[-1:],axis=0)
    
    v*=-1
    streamfunc = np.reshape(streamfunc,(M*M))
    return u,v

def collision_f(u_vels, v_vels, M):
    nlat = int(np.ceil(np.log2(M)))
    nlinks = int(np.ceil(np.log2(dirs)))
    n = nlinks+dim*nlat+1

    # for f, weights have to be calculated by advection diffusion eq, but u and v must be calculated
    def coef(link,vel):
        return w[link]*(1+e[link]*vel/cs**2)
    func = np.vectorize(coef)
    
    ret = np.zeros((5,M,M))
    ret[0] = func(0,np.zeros((M,M)))
    ret[1] = func(1,u_vels)
    ret[2] = func(2,u_vels)
    ret[3] = func(3,v_vels)
    ret[4] = func(4,v_vels)
    return ret.flatten()

def vortCirc(stream, M):
    nlat = int(np.ceil(np.log2(M)))
    nlinks = int(np.ceil(np.log2(dirs)))
    n = nlinks+dim*nlat+1

    q = QuantumRegister(nlinks+dim*nlat+2,'q')
    # a = AncillaRegister(1,'a')

    setup = QuantumCircuit(q)
    
    h1 = qlib.HGate().control(1,ctrl_state = 0)
    h3 = qlib.HGate().control(2,ctrl_state = '00')
    
    uvel,vvel = uv(stream, M)
    A_diag = collision_f(uvel,vvel, M)
    zeros = np.zeros(M*M)
    A_diag = np.concatenate((A_diag, zeros, zeros, zeros))
    B1_diag = top_half(A_diag)
    B2_diag = bottom_half(A_diag)
    
    setup.h(n - 1)

    Col1_diag = Diagonal(list(B1_diag))
    Col1 = Col1_diag.to_gate(label='c1')
    
    Col2_diag = Diagonal(list(B2_diag))
    Col2 = Col2_diag.to_gate(label='c2')
    
    # GENERALIZED
    indices = [n - 1] + [i for i in range(n - 1)]
    setup.append(Col1.control(1,ctrl_state = '0'), indices)
    setup.append(Col2.control(1,ctrl_state = '1'), indices)

    # GENERALIZED
    setup.h(n - 1)



    setup.barrier()
    L1 = lshift(nlat).to_gate(label = "L").control(nlinks,ctrl_state = '001')
    R1 = rshift(nlat).to_gate(label = "R").control(nlinks,ctrl_state = '010')
    L2 = lshift(nlat).to_gate(label = "L").control(nlinks,ctrl_state = '011')
    R2 = rshift(nlat).to_gate(label = "R").control(nlinks,ctrl_state = '100')
    
    # GENERALIZED
    log_M = round(math.log2(M))
    one = [n - 4, n - 3, n - 2] + [i for i in reversed(range(log_M))]
    two = [n - 4, n - 3, n - 2] + [i for i in reversed(range(log_M, 2*log_M))]
    setup.append(L1, one)
    setup.append(R1, one)
    setup.append(L2, two)
    setup.append(R2, two)
    setup.barrier()

    # GENERALIZED
    setup.h(n - 4)
    setup.h(n - 3)
    setup.h(n - 2)

    return setup

def calcBounds(streamfunction, M):
    arr = np.zeros((M,M))
    arr[-1] = -2*streamfunction[-2]
    arr[:,0] = -2*streamfunction[:,1]
    arr[:,-1] = -2*streamfunction[:,-2]
    arr[0] = -2*streamfunction[1]-2*U
    return arr

def Vtimestep(vort, stream, M):
    bounds = calcBounds(stream)
    zeros = np.zeros((M,M))
    vort = np.concatenate((vort,vort,vort,vort,vort,zeros,zeros,zeros)).flatten()
    vortSV = Statevector(vort).expand([1,0]).evolve(vortCirc(stream))
    vortAr = np.reshape(np.array(vortSV)[:M*M],(M,M))
    vortAr = np.real(vortAr)*2**(3/2)
    vortAr[0] = bounds[0]
    vortAr[-1] = bounds[-1]
    vortAr[:,0] = bounds[:,0]
    vortAr[:,-1] = bounds[:,-1]
    return np.reshape(vortAr,(M,M)), vort

def Stimestep(stream, source, M):
    zeros = np.zeros((M,M))
    stream = np.concatenate((stream,stream,stream,stream,stream,zeros,zeros,zeros,source,source,source,source,source,zeros,zeros,zeros)).flatten()
    streamSV = Statevector(stream).expand([1,0]).evolve(streamCirc())
    streamAr = np.reshape(np.array(streamSV)[:M*M],(M,M))
    streamAr = np.real(streamAr)*2**(4/2)
    streamAr[0] = 0
    streamAr[-1] = 0
    streamAr[:,0] = 0
    streamAr[:,-1] = 0
    return np.reshape(streamAr,(M,M)), stream

# JUMP
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