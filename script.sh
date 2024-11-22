#SBATCH -ECR-counts
#SBATCH -N1 --cpus-per-task=4  
#SBATCH --mem-per-cpu=4G  
#SBATCH -t 10:00:00  
#SBATCH -p rg-nextgen-hpc  
#SBATCH -p rg-nextgen-hpc                        # Partition Name
#SBATCH -o ./slurm-arm.out
#SBATCH -C aarch64,ampereq8030                   # Request an ARM64 node
#SBATCH -W                                       # Do not exit until the submitted job terminates.


cd parallelization
echo "Running job"

source /nethome/mlee769/miniconda3/bin/activate qiskit1

for n in 64 128 256 512 1024 2048 4096;
do
    echo $n
    python runQLBMCircuit.py --nlattice $n --outdir "data/"
done
