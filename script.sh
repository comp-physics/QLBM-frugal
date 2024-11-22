#SBATCH -ECR-counts
#SBATCH --ntasks=7
#SBATCH -N1 --cpus-per-task=4  
#SBATCH --mem-per-cpu=8G  
#SBATCH -t 10:00:00  
#SBATCH -p rg-nextgen-hpc  
#SBATCH -p rg-nextgen-hpc                        # Partition Name
#SBATCH -o ./slurm-arm.out
#SBATCH -C aarch64,ampereq8030                   # Request an ARM64 node
#SBATCH -W                                       # Do not exit until the submitted job terminates.


cd parallelization
echo "Running job"

source /nethome/mlee769/miniconda3/bin/activate qiskit1

srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK echo 64; python runQLBMCircuit.py --nlattice 64 --outdir "data/" &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK echo 128; python runQLBMCircuit.py --nlattice 128 --outdir "data/" &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK echo 256; python runQLBMCircuit.py --nlattice 256 --outdir "data/" &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK echo 512; python runQLBMCircuit.py --nlattice 512 --outdir "data/" &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK echo 1024; python runQLBMCircuit.py --nlattice 1024 --outdir "data/" &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK echo 2048; python runQLBMCircuit.py --nlattice 2048 --outdir "data/" &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK echo 4096; python runQLBMCircuit.py --nlattice 4096 --outdir "data/"

echo "DONE"
