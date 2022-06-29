"""
    Material properties. For each atom in the primitive cell specify N, S, L, and L^i S^j
    values.

    Note: the order of the parameters must match the phonopy output. For a given material run phonopy
    once and examine the output to get this list

    TODO: automate this process, specify values for unique atoms in cell and fill these vectors with the
    appropriate values.
"""

material = 'LiF'
supercell_dim = [2, 2, 2]
