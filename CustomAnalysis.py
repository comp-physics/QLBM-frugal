from qiskit_experiments.framework import BaseAnalysis, AnalysisResultData
from qiskit_experiments.framework import Options

class CustomAnalysis(BaseAnalysis):
    def _run_analysis(self, experiment_data):
        combined_counts = {}

        # get counts
        for datum in experiment_data:
            counts = datum["counts"]

            for key, val in counts.items():
                if key in combined_counts:
                    combined_counts[key] += val
                else:
                    combined_counts[key] = val

        result = AnalysisResultData("counts", combined_counts)

        return [result], []

    @classmethod
    def _default_options(cls) -> Options:
        options = super()._default_options()
        options.dummy_analysis_option = None
        options.plot = True
        options.ax = None
        return options