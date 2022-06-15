'''
This script provides the solution
for Question 9 of Assignment 3
'''
import sys
from pathlib import Path

# Python imports
import numpy as np

# Add the Utils Module
p = Path(__file__).resolve().parents[1]
sys.path.append(str(p))

# Utils module imports
from Utils.mc_routines import metropolis, rng
from Utils.colors import red, blue, magenta
from Utils.plt_creator import plt_creator

# Python imports
from scipy import stats

# proposal pdf (walker)
def proposal(theta):
    # theta_p = rng.normal(loc = theta, scale=3)
    theta_p = rng.uniform(1., 9.)
    return np.asarray([theta_p])
    # return theta_p

# target distribution
def target(theta):
    pdf = 0.25*((theta > 3) & (theta < 7))
    return pdf

# params
ndim = 1
npoints = 20000
theta_i = np.ones(ndim)*5.0

# sample list
sample = [theta_i]

# Markov chain vars
bad_steps, good_steps = [], [0]
bad_points = []

# sampler
for i in range(npoints):
    theta, prop_arr = metropolis(sample[-1], proposal, target, ndim)
    sample.append(theta)
    bad_points += prop_arr[:-1]
    bad_steps += list(range(good_steps[-1] + 1, len(prop_arr) + good_steps[-1]))
    good_steps.append(good_steps[-1] + len(prop_arr)) 

# KS Test
print(stats.ks_1samp(sample, stats.uniform.cdf, args=(3.0,4.0)))

# Plot Markov chain
plt = plt_creator(
    title="Markov Chain", 
    xLabel="steps", 
    yLabel=r'$\theta$',
    xMargin=0.02, 
    yMargin=0.02
)

plt.scatter(bad_steps, [A[0] for A in bad_points], color=magenta, s=10, label="Reject")
plt.plot(good_steps, [A[0] for A in sample], color=blue, lw=0.5, label="Chain")
plt.legend(loc="upper right")
plt.savefig("Results/MC_plot.png")

# Initialise plot
plt = plt_creator(
    title="Uniform (Metropolis)", 
    xLabel="x", 
    yLabel="Entries",
    xMargin=0.02, 
    yMargin=0.02
)

# Plot params
Range = (1, 9)
nbins = 40
x_space = np.linspace(Range[0], Range[1], nbins)

# Uniform histogram
uniform = np.linspace(3, 7, len(sample))
plt.hist(uniform, range=Range, bins=nbins, alpha=0.5, color=red, label="True Uniform")

# Metropolis Hostogram
plt.hist([A[0] for A in sample], range=Range, bins=nbins, alpha=0.5, color=blue, label="Metropolis")

plt.legend(loc="upper right")
plt.savefig("Results/Metropolis.png")
