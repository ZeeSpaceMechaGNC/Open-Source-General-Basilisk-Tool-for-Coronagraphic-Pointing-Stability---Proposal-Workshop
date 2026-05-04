import sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')
from run_sim import run_montecarlo, print_vv, run_single, save_fig
import matplotlib
matplotlib.use('Agg')

print("running MC 500 trials...", flush=True)
mc = run_montecarlo(method='hinf')
print("done mc", flush=True)

baseline = run_single(method='hinf', show_plot=False)
print("done baseline", flush=True)

# copy mc plot to results
import shutil
shutil.copy('Monte Carlo.png', '../progression/results/src_mc.png')
print("copied mc plot", flush=True)

print_vv(baseline, mc)
print("done", flush=True)
