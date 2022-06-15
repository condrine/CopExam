'''
This script provides the solution 
for Question 2 of End Term Exams
'''

import sys
from pathlib import Path

# Add path to the Utils Module
p = Path(__file__).resolve().parents[2]
sys.path.append(str(p))

# Python imports
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift, ifft
import numpy as np

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
Cf = fft(Count_arr)
xf = fftfreq(len(E_arr), E_arr)

sigma = 2.3
FFTGauss = np.exp(-(xf**2)*(sigma**2)/2)
Clean = Cf/FFTGauss
Sig = ifft(Clean)
sig_b = ifft(Cf/FFTGauss)
plt.plot(E_arr, Sig)
plt.show()
# print(FFTGauss)