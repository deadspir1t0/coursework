from numpy import exp


def initial(x):
    return exp(-(x-1)**2/2) - exp(-(x+1)**2/2)


def left(t):
    return 0


def right(t):
    return 0
