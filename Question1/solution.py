'''
This script provides the solution 
for Question 1 of End Term Exams
'''

import sys
from pathlib import Path

# Add path to the Utils Module
p = Path(__file__).resolve().parents[1]
sys.path.append(str(p))

# Utils module imports
from Utils.plt_creator import plt_creator
from Utils.diag_form import diagonal_form

# Python imports
from scipy.sparse import coo_matrix
import numpy as np
from scipy.linalg import solve_banded as sb

# Matrix
def Matrix(xi, xf, yi, yf, N):
    h = (xf - xi)/N
    X = np.linspace(xi, xf, N, endpoint=False)[1:]

    # calculate diagonal (i,i) elements
    diag = -2/(h*h) - 2/(X*X)
    diag_row = np.arange(0, N-1)
    diag_col = diag_row.copy()

    # calculate subdiagonal (i, i+1) elements
    subdiag_1 = 1/(h*h) + 1/(h*X[:-1])
    subdiag_1_row = np.arange(0, N-2)
    subdiag_1_col = subdiag_1_row.copy() + 1

    # calculate subdiagonal (i+1, i) elements
    subdiag_2 = 1/(h*h) - 1/(h*X[1:])
    subdiag_2_row = np.arange(1, N-1)
    subdiag_2_col = subdiag_2_row.copy() - 1

    # concatenate all the arrays to form the sparse matrix
    row = np.concatenate((diag_row, subdiag_1_row, subdiag_2_row))
    col = np.concatenate((diag_col, subdiag_1_col, subdiag_2_col))
    val = np.concatenate((diag, subdiag_1, subdiag_2))

    # create the main Matrix
    M = coo_matrix((val, (row, col)), shape=(N-1, N-1)).toarray()

    # create the RHS Vector
    b = np.sin(np.log(X))/(X*X)

    # apply the boundary conditions at (xi,yi) and (xf, yf)
    b[0] += -yi*(1/(h**2) - 1/(h*xi))
    b[N-2] += -yf*(1/(h**2) + 1/(xf*h))

    return M, b




# Params
xi, yi = [1,1]
xf, yf = [2,2]
N = 10000

# Get the Matrix and the RHS
M, b = Matrix(xi, xf, yi, yf, N)

# Solve M*X = b linear algebra Eqn.
X = sb((1, 1), diagonal_form(M), b)

# Plotter
plt = plt_creator(
    title=r'y vs x', 
    xLabel="x", 
    yLabel=r'y',
    xMargin=0.02, 
    yMargin=0.02
)
plt.plot(np.linspace(xi, xf, N, endpoint=False)[1:], X)
plt.savefig("Results/yvsx.png")
plt.show()




