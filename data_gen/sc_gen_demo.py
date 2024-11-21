import numpy as np


def get_faces(lst):
    """Compute all the possible faces by iteratively deleting vertices"""
    return [lst[:i] + lst[i+1:] for i in range(len(lst))]


def get_coeff(simplex, faces):
    """
    If simplex is not in the list of faces, return 0.
    If it is, return index parity.
    """
    if simplex in faces:
        idx = faces.index(simplex)
        return 1 if idx % 2 == 0 else -1
    else:
        return 0


def boundary(complex):
    """
    Given an abstract simplicial complex specified as a list of lists of vertices, return a
    list of boundary operators in matrix form.
    """
    # Get maximal simplex dimension
    maxdim = len(max(complex, key=len))
    # Group simplices by (ascending) dimension and sort them lexicographically
    simplices = [sorted([spx for spx in complex if len(spx) == i]) for i in range(1, maxdim+1)]

    # Iterate over consecutive groups (dim k and k+1)
    bnd = []
    for spx_k, spx_kp1 in zip(simplices, simplices[1:]):
        mtx = []
        for sigma in spx_kp1:
            faces = get_faces(sigma)
            mtx.append([get_coeff(spx, faces) for spx in spx_k])
        bnd.append(np.array(mtx).T)

    return bnd


X = ["A", "B", "C", "D", "AB", "AC", "BC", "BD", "CD", "BCD"]
print(boundary(X)[0])
print(boundary(X)[1])
