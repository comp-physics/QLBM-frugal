## A resource frugal quantum lattice Boltzmann method

<p align="center"> 
<a href="https://lbesson.mit-license.org/">
  <img src="https://img.shields.io/badge/License-MIT-blue.svg" />
</a>
<a href="http://doi.org/10.48550/arXiv.2401.12248">
  <img src="http://img.shields.io/badge/DOI-10.48550/arXiv.2401.12248-B31B1B.svg" />
</a>
</p>

Code for reproducing results from preprint:

__A two-circuit approach to reducing quantum resources for the quantum lattice Boltzmann method__  
_Authors: Sriharsha Kocherla, Austin Adams, Zhixin Song, Alexander Alexeev, Spencer H. Bryngelson_  
Georgia Institute of Technology, Atlanta, GA USA 30332

It can be cited as
```bibtex
@article{kocherla24_2,
  author = {Kocherla, S. and Adams, A. and Song, Z. and Alexeev, A. and Bryngelson, S. H.},
  title = {A two-circuit approach to reducing quantum resources for the quantum lattice {B}oltzmann method},
  journal = {arXiv preprint arXiv 2401.12248},
  doi = {10.48550/arXiv.2401.12248},
  year = {2024}
}
```

### Files
 * `advectionDiffusion/`: Contains code for solving the advection-diffusion equation
   * `QuantumD1Q2.ipynb`: D1Q2 QLBM advection-diffusion simulation
   * `QuantumD1Q3.ipynb`: D1Q3 QLBM advection-diffusion simulation
   * `QuantumD2Q5.ipynb`: D2Q5 QLBM advection-diffusion simulation
 * `twoCircuitLBM.ipynb`: Two-circuit QLBM 2D lid driven cavity flow simulation
 * `oneCircuitLBM.ipynb`: Single-circuit QLBM 2D-lid driven cavity flow simulation
 * `classicalLBM.ipynb`: Classical LBM 2D lid driven cavity flow simulation
 * `visualizations/`: Contains code for graphing and visualizing results
   * `errorAnalysis.ipynb`: Graphs error plots from output files
   * `graphics.ipynb`: Graphs results from output files

### License

MIT
