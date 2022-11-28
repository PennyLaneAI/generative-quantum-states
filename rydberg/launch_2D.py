import argparse
from curses import use_default_colors
import numpy as np
import os
from tqdm import tqdm, trange
from argparse import ArgumentParser

gpu_idx = 0
nx = 5
ny = 5
dim = 2
itr_step = .025
itr_init, itr_final = 1.025, 2.5
n_threads = 0
time_steps = 32 # number of steps for measurements in a single adiabatic evolution process
prefix = f"CUDA_VISIBLE_DEVICES={gpu_idx} "
script = "rydberg_evolute.jl"

main_cmd = f"julia {script} "
use_cuda = False
# solver = "VCABM"  # "Vern8"
solver = "Vern8"
detuning_start_measure = -2.
args_cmd = f"--dim {dim} --itr_init {itr_init} --itr_final {itr_final} --nx {nx} --ny {ny} --n_threads {n_threads} --itr_step {itr_step} --solver {solver} --time_steps {time_steps} --detuning_start_measure {detuning_start_measure}"
if use_cuda:
    args_cmd += " --use_cuda"

total_times = np.arange(0.4, 3.05, 0.2)

for total_time in tqdm(total_times, desc='T'):
    total_time = np.round(total_time, 3)
    cmd = prefix + main_cmd + args_cmd + f" --total_time {total_time}"
    print("Command: {cmd}".format(cmd=cmd))
    os.system(cmd)
