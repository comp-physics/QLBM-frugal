import numpy as np
import math
from qiskit.quantum_info import Statevector

def decimal_to_binary(n):
    out = "" 
    if n == 0:
        return ""
    
    return str(decimal_to_binary(int(n/2))) + str(n % 2)

def sort_counts(counts):
    # sort counts
    keys = list(counts.keys())
    keys.sort()
    counts = {i: counts[i] for i in keys}
    return counts

def correct_counts(counts, power):
    for i in range(pow(2, power)):
        binary_string = decimal_to_binary(i)
        # pad value
        binary_string = binary_string.zfill(power)

        if binary_string not in list(counts.keys()):
            counts[binary_string] = 0
        
    counts = sort_counts(counts)

    return counts

def counts_to_statevector(counts):
    if len(list(counts.keys())) == 0:
        raise Exception("Can not convert empty dictionary to Statevector")
    
    if len(list(counts.keys())) != pow(2, len(list(counts.keys())[0])):
        counts = correct_counts(counts)

    if list(counts.keys()) != list(counts.keys()).sort():
        counts = sort_counts(counts)

    vals = np.array(list(counts.values()))
    total_counts = 0

    for v in vals:
        total_counts = total_counts + v

    if total_counts == 0:
        raise Exception("Can not convert 0 counts to Statevector")

    for i in list(counts.keys()):
        counts[i] = math.sqrt(counts[i] / total_counts)
    
    # convert to SV
    sv = Statevector(list(counts.values()))

    return sv