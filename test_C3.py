# test_C15.py

from solver import NonogramSolver
from solver import NonogramSolverV2

row_clues = [
    [1],
[],
    [],
]
col_clues = [
    [],
[],
    [1],
]

MySolver = NonogramSolverV2(
    row=3,
    col=3,
    row_clues=row_clues,
    col_clues=col_clues,
)
MySolver.solve()
# 该样例下有输出BUG，先不管
MySolver.print_solution()
