# install the packages used in our simulation
using Pkg;
Pkg.add("Distributed");
Pkg.add("ProgressBars");
Pkg.add("DelimitedFiles");
Pkg.add("ArgParse");
Pkg.add("Bloqade");
Pkg.add("StatsBase");
Pkg.add("BitBasis");
Pkg.add("NPZ");
Pkg.add("ProgressBars");

## If you want to use GPUs for simulation, uncomment the following line
# Pkg.add("CUDA");
# Pkg.add("Adapt");

Pkg.update();