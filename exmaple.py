# exmaple.py

from solver import NonogramSolver
from solver import NonogramSolverV2

row_clues = [
    [6],
    [1,1],
    [1, 1],
    [1, 1,1],
    [1, 6],
    [1, 1, 1],
    [1,5],
    [1],
    [7],
    [1],
]
col_clues = [
    [7,1],
    [1,1],
    [1,1,1],
    [6,1],
    [1,1,1],
    [1,7],
    [1,1,1],
    [1,1],
    [1],
    [1],
]

# ----try dfs-----
MySolver = NonogramSolver(
    row=10,
    col=10,
    row_clues=row_clues,
    col_clues=col_clues,
)

# ---- or try smarter method-----
# MySolver = NonogramSolverV2(
#     row=10,
#     col=10,
#     row_clues=row_clues,
#     col_clues=col_clues,
# )

MySolver.solve()
MySolver.print_solution()
