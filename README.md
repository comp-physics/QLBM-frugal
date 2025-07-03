## A resource frugal quantum lattice Boltzmann method

<p align="center"> 
<a href="https://lbesson.mit-license.org/">
  <img src="https://img.shields.io/badge/License-MIT-blue.svg" />
</a>
<a href="https://doi.org/10.1016/j.future.2025.107975">
  <img src="https://img.shields.io/badge/DOI-10.1016/j.future.2025.107975-blue.svg" />
</a>
</p>

Code for reproducing results from the paper:

__A multiple-circuit approach to quantum resource reduction with application to the quantum lattice Boltzmann method__  
_Authors: Melody Lee, Zhixin Song, Sriharsha Kocherla, Austin Adams, Alexander Alexeev, Spencer H. Bryngelson_  
Future Generation Computing Systems, _174_ 107975 (2026)  
Georgia Institute of Technology, Atlanta, GA USA 30332

It can be cited as
```bibtex
@article{lee25,
  author = {Lee, M. and Song, Z. and Kocherla, S. and Adams, A. and Alexeev, A. and Bryngelson, S. H.},
  title = {A multiple-circuit approach to quantum resource reduction with application to the quantum lattice {B}oltzmann method},
  doi = {10.1016/j.future.2025.107975},
  journal = {Future Generation Computing Systems},
  pages = {107975},
  volume = {174},
  year = {2026}
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
