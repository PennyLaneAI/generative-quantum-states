import argparse
import numpy as np
import os
from tqdm import tqdm, trange
from argparse import ArgumentParser

gpu_idx = 0
ny = 1
dim = 1


itr_step = .05
itr_init, itr_final = 1., 2.95



itr_final = round(itr_final, 2)
itr_init = round(itr_init, 2)

n_threads = 0
time_steps = 40 # number of steps for measurements in a single adiabatic evolution process
prefix = f"CUDA_VISIBLE_DEVICES={gpu_idx} "
script = "rydberg_evolution.jl"
main_cmd = f"julia {script} "
use_cuda = True
# solver = "VCABM"  # "Vern8"
solver = "Vern8"
detuning_start_measure = -2
args_cmd = f"--dim {dim} --itr_init {itr_init} --itr_final {itr_final} --ny {ny} --n_threads {n_threads} --itr_step {itr_step} --solver {solver} --time_steps {time_steps} --detuning_start_measure {detuning_start_measure} "
if use_cuda:
    args_cmd += " --use_cuda"

total_time = 15
blockade_subspace = True
if not blockade_subspace:
    args_cmd += " --blockade_subspace false"

n_qubits = np.arange(13,34,2)

for nx in tqdm(n_qubits, desc='n_qubits'):
    cmd = prefix + main_cmd + args_cmd + \
        f" --total_time {total_time} --nx {nx}"
    print("Command: {cmd}".format(cmd=cmd))
    os.system(cmd)
