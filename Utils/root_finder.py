# Functions for finding the roots.

def secant(v1, v2, x1, x2, xf):
    return v1 + (xf - x1)*(v1 - v2)/(x1 - x2)