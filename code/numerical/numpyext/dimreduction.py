import numpy as np


def PCAreduction(Ycentered, Q):
    eigenval, eigenvec = np.linalg.eigh(np.dot(Ycentered.T, Ycentered))
    eigenvecscaled = eigenvec / np.sqrt(eigenval)  # whitening
    basis = eigenvecscaled[:, -Q:]
    mapped = np.dot(Ycentered, basis)
    return mapped
