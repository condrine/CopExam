'''
This script provides the solution 
for Question 2 of End Term Exams
'''

import sys
from pathlib import Path

# Add path to the Utils Module
p = Path(__file__).resolve().parents[1]
sys.path.append(str(p))

# Utils module imports
from Utils.plt_creator import plt_creator

# Python imports
import pandas as pd
import numpy as np
from scipy.fft import fft, fftfreq, fftshift, ifft

# read file and store data
file = "dimuon-yield"
E_arr =[]
Count_arr = []
with open(file) as f:
    line = f.readline()
    line = f.readline()
    while line:
        E, Count = line.split()
        # create a new entry for a new t
        E_arr.append(float(E))
        Count_arr.append(float(Count))
        line = f.readline()

Count_arr = np.asarray(Count_arr)
E_arr = np.asarray(E_arr)


# Plot signal before denoising
plt = plt_creator(
    title=r'Counts vs E', 
    xLabel="E (GeV)", 
    yLabel=r'Counts',
    xMargin=0.02, 
    yMargin=0.02
)
plt.plot(E_arr, Count_arr)
plt.savefig("Results/CvsE_before.png")
plt.close()

# Get the fft and the frequencies of the given data
fft_Count = fft(Count_arr)
freq = fftfreq(len(E_arr), E_arr)

# Plot signal before denoising in freq domain
plt = plt_creator(
    title=r'Amplitude squared vs freq.', 
    xLabel="freq", 
    yLabel=r'Amplitude squared',
    xMargin=0.02, 
    yMargin=0.02
)
plt.plot(fftshift(freq), np.abs(fftshift(fft_Count))**2)
plt.savefig("Results/AvsF_before.png")
plt.close()

# Define FFT of Gaussian
sigma = 2.3
omega = 2*np.pi*freq
FFTGauss = np.exp(-(omega**2)*(sigma**2)/2)

# Denoise
Clean = fft_Count/FFTGauss
Counts_clean = ifft(Clean)

# Plot signal after denoising
plt = plt_creator(
    title=r'Counts vs E', 
    xLabel="E (GeV)", 
    yLabel=r'Counts',
    xMargin=0.02, 
    yMargin=0.02
)
plt.plot(E_arr, np.abs(Counts_clean))
plt.savefig("Results/CvsE_after.png")
plt.close()