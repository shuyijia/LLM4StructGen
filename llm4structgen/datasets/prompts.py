Z_MATRIX_GENERATION_PROMPT_HEADER = """
Below is a description of a bulk material where each atom is described by its \
element type and three attributes: 1. distance to the previous atom, 2. angle \
to the previous two atoms, 3. dihedral angle to the previous three atoms. The \
first three Fm atoms are dummies that help define the rest of the material. \
<attributes> Generate a description of the lengths and angles of the lattice \
vectors and the three dummy Fm atoms, followed by the element type and the \
three attributes for each atom within the lattice:
"""

CARTESIAN_GENERATION_PROMPT_HEADER = """
Below is a description of a bulk material. <attributes> Generate a description \
of the lengths and angles of the lattice vectors and then the element type and \
coordinates for each atom within the lattice:
"""