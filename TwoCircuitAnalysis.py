from qiskit_experiments.framework import BaseAnalysis, AnalysisResultData

class Analysis(BaseAnalysis):
    def _run_analysis(self, experiment_data):
    
        combined_counts = {}
        for datum in experiment_data.data():
            # Get counts
            counts = datum["counts"]
            num_bits = len(next(iter(counts)))

            # Get metadata
            metadata = datum["metadata"]
            clbits = metadata["rm_bits"]
            sig = metadata["rm_sig"]

            # Construct full signature
            full_sig = num_bits * [0]
            for bit, val in zip(clbits, sig):
                full_sig[bit] = val

            # Combine dicts
            for key, val in counts.items():
                bitstring = self._swap_bitstring(key, full_sig)
                if bitstring in combined_counts:
                    combined_counts[bitstring] += val
                else:
                    combined_counts[bitstring] = val

        result = AnalysisResultData("counts", combined_counts)
        return [result], []
    
    # Helper dict to swap a clbit value
    _swap_bit = {"0": "1", "1": "0"}
    
    @classmethod
    def _swap_bitstring(cls, bitstring, sig):
        """Swap a bitstring based signature to flip bits at."""
        # This is very inefficient but demonstrates the basic idea
        return "".join(reversed(
            [cls._swap_bit[b] if sig[- 1 - i] else b for i, b in enumerate(bitstring)]
        ))