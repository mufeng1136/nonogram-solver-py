from itertools import chain
from itertools import groupby
import logging

# Module logger
logger = logging.getLogger(__name__)
def integer_partitions(
    n: int, k: int, current: list[int] | None = None
) -> list[list[int]]:
    """Generate all ordered compositions of integer n into k parts.

    Args:
        n: The integer to partition
        k: The number of parts to partition into
        current: Current partition being built (used internally for recursion)

    Returns:
        A list of all possible compositions, where each composition is a list of k positive integers
        that sum to n. Different orderings are treated as different compositions.

    Example:
        >>> integer_partitions(5, 2)
        [[1, 4], [2, 3], [3, 2], [4, 1]]
        >>> integer_partitions(4, 3)
        [[1, 1, 2], [1, 2, 1], [2, 1, 1]]
    """
    if current is None:
        current = []

    if k == 1:
        return [current + [n]]

    if k > n:
        return []

    result = []

    for i in range(1, n - k + 2):
        result.extend(integer_partitions(n - i, k - 1, current + [i]))

    return result

class NonogramSolver:
    def __init__(
        self,
        row: int,
        col: int,
        row_clues: list[list[int]] | None = None,
        col_clues: list[list[int]] | None = None,
    ):
        self.row_clues: list = row_clues
        self.col_clues: list = col_clues
        self.row: int = row
        self.col: int = col
        self.rows_possible: list[list[list[int]]] = []
        self.cols_current: list[list[int]] = [[0] for _ in range(col)]
        self.solved: bool = False
        self.ready_to_solve: bool = False
        self.rows_possible_index: list[int] = [0 for _ in range(row)]

    def row_clues(self, row_clues: list[list[int]]) -> None:
        self.row_clues = row_clues
        self.ready_to_solve = False

    def col_clues(self, col_clues: list[list[int]]) -> None:
        self.col_clues = col_clues
        self.ready_to_solve = False

    def check_ready(self) -> bool:
        """Check if the solver is ready to solve and validate clues.

        Validates:
        - Clues are not None
        - Number of clues matches grid dimensions
        - Each row clue sum doesn't exceed column count
        - Each column clue sum doesn't exceed row count
        - Clue values are positive integers
        """
        if self.row_clues is None or self.col_clues is None:
            self.ready_to_solve = False
            return False

        if len(self.row_clues) != self.row or len(self.col_clues) != self.col:
            self.ready_to_solve = False
            return False

        # Validate row clues
        for i, clue in enumerate(self.row_clues):
            # Check for empty or invalid clues
            if not isinstance(clue, list):
                self.ready_to_solve = False
                return False

            # Check all values are positive integers
            if any(val <= 0 for val in clue):
                self.ready_to_solve = False
                return False

            # Check if clue can fit in the row
            # Minimum cells needed: sum of clues + gaps between them
            min_cells_needed = sum(clue) + len(clue) - 1 if clue else 0
            if min_cells_needed > self.col:
                self.ready_to_solve = False
                return False

        # Validate column clues
        for i, clue in enumerate(self.col_clues):
            # Check for empty or invalid clues
            if not isinstance(clue, list):
                self.ready_to_solve = False
                return False

            # Check all values are positive integers
            if any(val <= 0 for val in clue):
                self.ready_to_solve = False
                return False

            # Check if clue can fit in the column
            min_cells_needed = sum(clue) + len(clue) - 1 if clue else 0
            if min_cells_needed > self.row:
                self.ready_to_solve = False
                return False
            
        col_clue_sums = sum(sum(clue) for clue in self.col_clues)
        row_clue_sums = sum(sum(clue) for clue in self.row_clues)
        if col_clue_sums != row_clue_sums:
            return False

        self.ready_to_solve = True
        return True

    @staticmethod
    def _generate_one_possibility(clue: list[int], partition: list[int]) -> list[int]:
        parts = list(
            chain.from_iterable(
                chain([0] * partition[i], [1] * c) for i, c in enumerate(clue)
            )
        )
        parts.extend([0] * partition[-1])
        return parts[1:-1]

    def _generate_possibilities_for_clue(self,
        clue: list[int], length: int
    ) -> list[list[int]]:
        partitions = integer_partitions(
            length - sum(clue) + 2, len(clue) + 1
        )
        possibilities = [
            self._generate_one_possibility(clue, partition)
            for partition in partitions
        ]
        return possibilities

    def _generate_possibilities(self) -> None:
        for clue in self.row_clues:
            self.rows_possible.append(self._generate_possibilities_for_clue(clue, self.col))
        pass

    @staticmethod
    def _check_one_column(column_current: list[int], clue: list[int]) -> bool:
        ans: bool = True

        new_cc = [
            sum(1 for _ in grp) for val, grp in groupby(column_current) if val == 1
        ]
        len_new_cc: int = len(new_cc)
        len_clue: int = len(clue)
        for i, c in enumerate(new_cc):
            if i >= len_clue:
                logger.debug(
                    "no match for reason 1: runs=%s, column=%s, clue=%s",
                    new_cc,
                    column_current,
                    clue,
                )
                return False
            elif i == len_new_cc - 1 and c > clue[i]:
                logger.debug(
                    "no match for reason 2: runs=%s, column=%s, clue=%s",
                    new_cc,
                    column_current,
                    clue,
                )
                return False
            elif i < len_new_cc - 1 and c != clue[i]:
                logger.debug(
                    "no match for reason 3: runs=%s, column=%s, clue=%s",
                    new_cc,
                    column_current,
                    clue,
                )
                ans = False

        return True

    def _check_columns_match_clues(self) -> bool:
        for i, col in enumerate(self.cols_current):
            if not self._check_one_column(col, self.col_clues[i]):
                return False
        return True

    def _columns_step_ahead(self, row: list[int]) -> None:
        for i, val in enumerate(row):
            self.cols_current[i].append(val)

    def _columns_step_back(self) -> None:
        for i in range(self.col):
            self.cols_current[i].pop()

    def _recursive_solve(self, row_idx: int) -> bool:
        """Recursively solve the nonogram using backtracking.

        Args:
            row_idx: Current row index to process

        Returns:
            True if solution found, False otherwise
        """
        # Base case: all rows processed
        if self.solved:
            return False
        if row_idx >= self.row:
            # Check if all columns match their clues
            logger.debug("Reached base case")
            logger.debug("cols_current=%s", self.cols_current)
            flag = self._check_columns_match_clues()
            if flag:
                self.solved = True
                logger.info("Solved inside recursive")
                logger.debug("Final cols_current: %s", self.cols_current)
                logger.debug("Final rows_possible_index: %s", self.rows_possible_index)
                return self._check_columns_match_clues()
            else:
                return False

        # Try each possible configuration for the current row
        for i, possible_row in enumerate(self.rows_possible[row_idx]):

            # Update column states with this row
            self._columns_step_ahead(possible_row)
            self.rows_possible_index[row_idx] = i
            logger.debug("Trying row %s possibility %s: %s", row_idx, i, possible_row)
            logger.debug("Current cols_current: %s", self.cols_current)
            # Check if columns are still valid
            if self._check_columns_match_clues():
                # Recursively try next row
                if self._recursive_solve(row_idx + 1):
                    self.solved = True
                    return True

            # Backtrack: undo column changes
            else:
                self._columns_step_back()
        self._columns_step_back()
        return False

    def solve(self) -> None:
        """Solve the nonogram puzzle."""
        self.check_ready()
        if not self.ready_to_solve:
            raise RuntimeError("Solver is not ready. Please check clues and grid size.")

        # Generate all possible row configurations
        self._generate_possibilities()

        # Initialize column state tracking
        self.cols_current = [[0] for _ in range(self.col)]

        # Start recursive search from row 0
        self.solved = self._recursive_solve(0)

        if not self.solved:
            raise RuntimeError("No solution found for the given clues.")

    def get_solution(self) -> list[list[int]]:
        """Retrieve the solved nonogram grid.

        Returns:
            A 2D list representing the solved grid, where 1 indicates filled cells and 0 indicates empty cells.

        Raises:
            RuntimeError: If the puzzle has not been solved yet.
        """
        if not self.solved:
            raise RuntimeError("Puzzle not solved yet. Call solve() first.")

        solution = [
            self.rows_possible[i][self.rows_possible_index[i]] for i in range(self.row)
        ]
        return solution

    def print_solution(
        self,
        as_chars: bool = True,
        cell_width: int = 3,
        show_row_clues: bool = True,
        show_col_clues: bool = True,
        show_row_numbers: bool = True,
    ) -> None:
        """Print the solved nonogram to stdout in a nicely formatted way.

        Shows column clues vertically above the grid, row clues to the left,
        optional row numbers, and ensures equal column width.

        Args:
            as_chars: If True, prints using `#` for filled and `.` for empty.
                      If False, prints numeric 1/0.
            cell_width: width in characters for each cell (default 3).
            show_row_clues: whether to print row clues at left.
            show_col_clues: whether to print column clues above.
            show_row_numbers: whether to print row numbers at left.
        """
        if not self.solved:
            raise RuntimeError("Puzzle not solved yet. Call solve() first.")

        solution = self.get_solution()

        # Prepare row clues and column clues
        row_clues = self.row_clues or [[] for _ in range(self.row)]
        col_clues = self.col_clues or [[] for _ in range(self.col)]

        # Compute widths for left area
        rownum_width = len(str(self.row)) if show_row_numbers else 0
        row_clue_texts = [" ".join(map(str, rc)) for rc in row_clues]
        row_clue_width = (
            max((len(t) for t in row_clue_texts), default=0) if show_row_clues else 0
        )
        left_pad = 0
        if show_row_numbers:
            left_pad += rownum_width + 1  # number + space
        if show_row_clues:
            left_pad += row_clue_width + 1  # clues + space

        # Prepare column clues matrix (top to bottom)
        max_col_clue_h = max((len(c) for c in col_clues), default=0)
        col_clue_matrix = []
        for c in col_clues:
            padded = [""] * (max_col_clue_h - len(c)) + [str(x) for x in c]
            col_clue_matrix.append(padded)

        def format_cell(v: int) -> str:
            if as_chars:
                ch = "#" if v == 1 else "."
            else:
                ch = str(v)
            # center in cell_width
            return ch.center(cell_width)

        # Print column clues
        if show_col_clues and max_col_clue_h > 0:
            for r in range(max_col_clue_h):
                line = " " * left_pad
                parts = []
                for j in range(self.col):
                    val = col_clue_matrix[j][r]
                    parts.append(val.center(cell_width))
                line += "".join(parts)
                print(line)

        # Print grid with row numbers and row clues
        for idx, row in enumerate(solution, start=1):
            parts = []
            for v in row:
                parts.append(format_cell(v))

            left = ""
            if show_row_numbers:
                left += str(idx).rjust(rownum_width) + " "
            if show_row_clues:
                left += row_clue_texts[idx - 1].rjust(row_clue_width) + " "

            print(left + "".join(parts))

class NonogramSolverV2(NonogramSolver):
    """No dfs ,More Intelligent"""
    def __init__(
            self,
            row: int,
            col: int,
            row_clues: list[list[int]] | None = None,
            col_clues: list[list[int]] | None = None,
    ):
        super().__init__( row, col, row_clues, col_clues)

        # Solution grid where: 1 = filled cell, 0 = empty cell, -1 = uncertain
        # The first layer is rows, and the second layer is cells within rows
        self.solve_result: list[list[int]] = [[-1] * col for _ in range(row)]

        self.cols_possible: list[list[list[int]]] = []

        del self.cols_current

    def solve(self) -> None:
        """Solve the nonogram puzzle."""
        self.check_ready()
        if not self.ready_to_solve:
            raise RuntimeError("Solver is not ready. Please check clues and grid size.")

        # Generate all possible row configurations
        self._generate_possibilities()

        # Start overlapping states solve
        self.solved = self._overlapping_states_solve()

        if not self.solved:
            raise RuntimeError("No solution found for the given clues.")

    def _generate_possibilities(self) -> None:
        for clue in self.row_clues:
            self.rows_possible.append(self._generate_possibilities_for_clue(clue, self.col))
        for clue in self.col_clues:
            self.cols_possible.append(self._generate_possibilities_for_clue(clue, self.row))
        pass

    @staticmethod
    def _reasoning_solve_result(current_state: list[int], possible: list[list[int]]) -> None:
        """
        Perform logical deduction for a single row/column and update state in-place.

        This method filters the possible patterns based on the current state and then
        deduces which cells must be filled (1) or empty (0) in all valid patterns.
        The results are directly applied to the input parameters.

        Args:
            current_state: Current state of the row/column where:
                          -1 = uncertain, 0 = empty, 1 = filled (modified in-place)
            possible: List of all possible patterns for this row/column (modified in-place)

        Raises:
            ValueError: If no valid patterns match the current state (contradiction detected)
        """
        length = len(current_state)
        if length == 0:
            return  # No cells to process

        # Filter possible patterns to only those consistent with current state
        filtered_possible = []
        for pattern in possible:
            valid = True
            for i in range(length):
                # Skip uncertain positions, but check determined positions
                if current_state[i] != -1 and current_state[i] != pattern[i]:
                    valid = False
                    break
            if valid:
                filtered_possible.append(pattern)

        # Update the possible list in place to reflect the filtered results
        possible.clear()
        possible.extend(filtered_possible)

        # If no valid patterns remain, raise an exception
        if not possible:
            raise ValueError("Contradiction detected: no valid patterns match the current state")

        # For each position, check if all valid patterns agree on the value
        for i in range(length):
            all_ones = True
            all_zeros = True

            for pattern in possible:
                if pattern[i] == 0:
                    all_ones = False
                else:  # pattern[i] == 1
                    all_zeros = False

                # Early exit if both become False
                if not all_ones and not all_zeros:
                    break

            # Update current_state in-place based on consensus
            if all_ones:
                current_state[i] = 1
            elif all_zeros:
                current_state[i] = 0
            # Otherwise remains unchanged (could be -1 or previously determined value)

    def _overlapping_states_solve(self) -> bool:
        """
        Solve the Nonogram puzzle using iterative row and column reasoning.

        This method alternates between reasoning on rows and columns. In each iteration,
        it first processes all rows using the `_reasoning_solve_result` method to update
        the solve_result based on row clues and possible patterns. Then it processes all
        columns similarly. The process repeats until no more changes are made in an
        entire iteration or the puzzle is solved.

        Returns:
            bool: True if the puzzle is successfully solved (no uncertain cells remain),
                  False if no further progress can be made but uncertain cells still exist.

        Raises:
            ValueError: If a contradiction is found during reasoning (e.g., no valid
                       patterns match the current state for a row or column).
        """
        changed = True
        iterations = 0
        max_iterations = self.row * self.col * 2  # Prevent infinite loops

        # Continue until no changes occur or puzzle is solved
        while changed and iterations < max_iterations:
            changed = False
            iterations += 1

            # Phase 1: Process all rows
            for i in range(self.row):
                # Get current row state and possible patterns
                current_row_state = self.solve_result[i]
                row_possibilities = self.rows_possible[i]

                # Skip if row is already determined
                if -1 not in current_row_state:
                    continue

                # Save original state for comparison
                original_state = current_row_state.copy()

                try:
                    # Apply reasoning to this row
                    self._reasoning_solve_result(
                         current_row_state, row_possibilities
                    )
                except ValueError as e:
                    # Contradiction found in row reasoning
                    raise ValueError(f"Contradiction in row {i}: {e}")

                # Check if row state was updated
                if current_row_state != original_state:
                    changed = True
                    # Update the solve_result in place (current_row_state is a reference)
                    # No need to explicitly assign as it's modified in place

            # Phase 2: Process all columns
            for j in range(self.col):
                # Build current column state
                current_col_state = [self.solve_result[i][j] for i in range(self.row)]
                col_possibilities = self.cols_possible[j]

                # Skip if column is already determined
                if -1 not in current_col_state:
                    continue

                # Save original state for comparison
                original_state = current_col_state.copy()

                try:
                    # Apply reasoning to this column
                    self._reasoning_solve_result(
                         current_col_state, col_possibilities
                    )
                except ValueError as e:
                    # Contradiction found in column reasoning
                    raise ValueError(f"Contradiction in column {j}: {e}")

                # Check if column state was updated and update solve_result
                if current_col_state != original_state:
                    changed = True
                    # Write back the updated column state to solve_result
                    for i in range(self.row):
                        self.solve_result[i][j] = current_col_state[i]

            # Check if puzzle is solved (no uncertain cells remaining)
            if all(-1 not in row for row in self.solve_result):
                return True

        # Check why we exited the loop
        if any(-1 in row for row in self.solve_result):
            # Puzzle not solved but no more progress can be made
            return False
        else:
            # Puzzle solved
            return True

    def get_solution(self) -> list[list[int]]:
        return self.solve_result

class NonogramSolverV3(NonogramSolverV2):
    def __init__(
            self,
            row: int,
            col: int,
            row_clues: list[list[int]] | None = None,
            col_clues: list[list[int]] | None = None,
    ):
        super().__init__(row, col, row_clues, col_clues)

        # Solution grid where: 1 = filled cell, 0 = empty cell, -1 = uncertain
        # The first layer is rows, and the second layer is cells within rows
        self.solve_result: list[list[int]] = [[-1] * col for _ in range(row)]

        # self.cols_possible: list[list[list[int]]] = []

        # del self.cols_current
        del self.cols_possible
        del self.rows_possible
        del self.rows_possible_index

    def solve(self) -> None:
        # 该方法的重要假设为：能够确定的格子的状态空间数量远小于无法确定状态的格子。
        # 因此在计算可能性时，不罗列所有情况，而是直接判断格子是否能被确定
        # 在单行/列穷举时，遇到不合理情况批量跳过
        # 为此需要：
        # ①一个函数，能够根据该行（列）的线索和已经确定的格子判断新确定格子
        #       判断方法：针对某格子，查找其能确定和不能确定的样例。各找到一个说明无法确定，找不到说明该格子颜色已确定
        # ②为了实现①，需要函数，能够遍历格子排布。遍历方法：将所有可能性看成一个链，从前往后遍历

        # ===========创新点==============
        # 你提出的是一个非常高级的剪枝优化策略，用于在
        # _has_possible_state
        # 中跳过大量无效的
        # inter
        # 组合，而不是简单地逐一枚举。
        #
        # 这种技术称为 “冲突驱动的跳跃”（conflict - driven
        # skipping），在
        # Nonogram
        # 求解器中属于高效推理的核心技巧。

        self.check_ready()
        if not self.ready_to_solve:
            raise RuntimeError("Solver is not ready. Please check clues and grid size.")

        # Start solve
        self.solved = self._solve_by_iterative_line_reasoning()

        if not self.solved:
            raise RuntimeError("No solution found for the given clues.")

        return

    def _solve_by_iterative_line_reasoning(self) -> bool:
        """
        Solve the nonogram by iteratively applying line-wise logical deduction.

        Procedure:
            - Repeatedly refresh all rows, then all columns.
            - Stop when a full iteration (rows + cols) makes no changes.
            - Return True if fully solved (no -1 remains), False otherwise.

        Raises:
            ValueError: If contradiction is found in any line.
        """
        changed = True
        iteration_count = 0
        max_iterations = self.row * self.col + 10  # safety bound

        while changed and iteration_count < max_iterations:
            changed = False
            iteration_count += 1

            # --- Phase 1: Refresh all rows ---
            for i in range(self.row):
                original_row = self.solve_result[i].copy()
                clue = self.row_clues[i]
                new_row = self._refresh_line_solution(
                    line_solution=original_row,
                    grid_num=self.col,
                    clue=clue
                )
                # Update in-place if changed
                if new_row != original_row:
                    self.solve_result[i] = new_row
                    changed = True

            # --- Phase 2: Refresh all columns ---
            for j in range(self.col):
                # Extract current column
                original_col = [self.solve_result[i][j] for i in range(self.row)]
                clue = self.col_clues[j]
                new_col = self._refresh_line_solution(
                    line_solution=original_col,
                    grid_num=self.row,
                    clue=clue
                )
                # Write back if changed
                if new_col != original_col:
                    for i in range(self.row):
                        self.solve_result[i][j] = new_col[i]
                    changed = True

            # Early exit if fully solved
            if all(cell != -1 for row in self.solve_result for cell in row):
                return True

        # After loop, check final state
        fully_solved = all(cell != -1 for row in self.solve_result for cell in row)
        return fully_solved

    @staticmethod
    def _refresh_line_solution(line_solution: list[int], grid_num: int, clue: list[int]) -> list[int] :
        # 更新solution状态
        solution_copy = line_solution.copy()

        # 逐个检查格子，能否找到两个可行解
        for i_grid in range(grid_num):
            if solution_copy[i_grid] != -1:
                continue
            # 该格子状态不确定

            # 查找其为空是否有解
            solution_copy[i_grid] = 0
            is_empty_valid = NonogramSolverV3._has_possible_state(solution_copy,grid_num,clue)

            # 查找其为实是否有解
            solution_copy[i_grid] = 1
            is_filled_valid = NonogramSolverV3._has_possible_state(solution_copy,grid_num,clue)

            # 结论
            if is_empty_valid and is_filled_valid:
                solution_copy[i_grid]= -1
            elif is_empty_valid:
                solution_copy[i_grid] = 0
            elif is_filled_valid:
                solution_copy[i_grid] = 1
            else:
                raise ValueError("Single grid iteration error")

        return solution_copy

    @staticmethod
    def _has_possible_state(solution: list[int], grid_num: int, clue: list[int]) -> bool:
        """
        Check if there exists at least one valid state matching the clue
        that is compatible with the current partial solution.

        In `solution`:
            - 1 = filled (must be 1 in candidate)
            - 0 = empty (must be 0 in candidate)
            - -1 = unknown (can be 0 or 1)
        """
        if len(solution) != grid_num:
            raise ValueError("solution length must equal grid_num")

        clue_num = len(clue)

        # Case: no clues → must be all empty
        if clue_num == 0:
            return all(cell != 1 for cell in solution)

        min_required = sum(clue) + (clue_num - 1)
        if min_required > grid_num:
            return False  # impossible to fit

        total_extra_max = grid_num - min_required  # >=0
        k = clue_num

        # Start with inter = [0] * k
        inter = [0] * k

        while True:
            # Only consider if sum(inter) <= total_extra_max
            if sum(inter) <= total_extra_max:

                expected_val = None

                try:
                    state = NonogramSolverV3._get_state_from_clue_and_inter(clue, inter, grid_num)
                    # Check compatibility
                    compatible = True
                    for i_grid in range(grid_num):
                        if solution[i_grid] != -1 and solution[i_grid] != state[i_grid]:
                            compatible = False
                            conflict_index = i_grid
                            expected_val = solution[i_grid]
                            break
                    if compatible:
                        return True
                except Exception:
                    pass  # skip invalid

                # --- Apply intelligent skipping based on conflict ---
                if expected_val == 0:
                    # Constraint is EMPTY, but state has FILLED at conflict_index
                    # Find which block contains conflict_index
                    # Reconstruct block positions from inter and clue
                    pos = NonogramSolverV3._inter2pos(clue, inter)

                    # Find the rightmost block that ends <= conflict_index
                    idx_clue = next((i for i in reversed(range(len(pos))) if pos[i] <= conflict_index), -1)

                    # Move this block to the right so conflict_index becomes empty
                    right_move_grid = conflict_index + 1 - pos[idx_clue]
                    inter[idx_clue] += right_move_grid

                    # 其他块间距为0
                    inter[idx_clue+1:] = [0] * (len(inter) - (idx_clue+1) )

                    # 如果移动后inter总和已经超过最大值，则可以判断不存在合理state。
                    if sum(inter)>total_extra_max:
                        return False

                elif expected_val == 1:
                    # Constraint is FILLED, but state has EMPTY at conflict_index
                    # Compute block end positions
                    pos = NonogramSolverV3._inter2pos(clue,inter)

                    # Find the rightmost block that ends < conflict_index
                    idx_clue = next((i for i in reversed(range(len(pos))) if pos[i] < conflict_index), -1)

                    # 如果i_grid左侧已没有填充块，则直接人为不存在合理state
                    if idx_clue < 0:
                        return False

                    # 左侧右移，直至右端符合条件
                    right_move_grid = conflict_index - (pos[idx_clue]+clue[idx_clue]-1)
                    inter[idx_clue] += right_move_grid

                    # 其他块间距为0
                    inter[idx_clue+1:] = [0] * (len(inter) - (idx_clue+1) )

                    # 如果移动后inter总和已经超过最大值，则可以判断不存在合理state。
                    if sum(inter)>total_extra_max:
                        return False
            else:
                #============ simple +1 =================
                # Generate next inter: increment like a counter from right
                pos = k - 1
                while pos >= 0:
                    inter[pos] += 1
                    # If after increment, total sum already exceeds, we need to carry
                    if sum(inter) <= total_extra_max:
                        break  # valid, stop incrementing
                    else:
                        # Overflow: reset this digit to 0 and carry to left
                        inter[pos] = 0
                        pos -= 1
                else:
                    # pos < 0: all digits overflowed → done
                    break



        return False

    @staticmethod
    def _inter2pos(_clue: list[int], _inter: list[int]) -> list[int]:
        """
        Compute the starting index of each block given clue and inter-gap increments.

        Args:
            _clue: List of block lengths, e.g., [2, 1, 3]
            _inter: List of extra spaces for the first len(_clue) gaps:
                    - _inter[0]: extra spaces before the first block
                    - _inter[i] for i>=1: extra spaces in the gap between block (i-1) and block i

        Returns:
            A list of starting indices for each block.
            Example: _clue=[2,1], _inter=[1,0] → blocks start at [1, 1+2+1+0 = 4] → [1, 4]
        """
        if not _clue:
            return []

        k = len(_clue)
        if len(_inter) != k:
            raise ValueError(f"_inter must have length {k} (same as clue), got {len(_inter)}")

        starts = []
        current_pos = _inter[0]  # leading extra spaces

        for i in range(k):
            starts.append(current_pos)
            # Add current block length
            current_pos += _clue[i]
            # Add mandatory gap (1 zero) + extra gap (if not last block)
            if i < k - 1:
                current_pos += 1 + _inter[i + 1]

        return starts


    @staticmethod
    def _get_state_from_clue_and_inter(_clue: list[int], _inter: list[int], _grid_num: int) -> list[int]:
        """
        Generate a row/column state from a clue and extra spaces in the first k gaps.

        The last gap's extra space is inferred from the total length.

        Args:
            _clue: List of block lengths (e.g., [1, 2, 3])
            _inter: List of extra spaces for the first len(_clue) gaps.
                    - _inter[0]: extra before first block
                    - _inter[i] for i>=1: extra in the gap after block i (between block i and i+1)
                    Length must be len(_clue).
            _grid_num: Total length of the row/column.

        Returns:
            A list of 0s and 1s of length _grid_num.
        """
        if not _clue:
            return [0] * _grid_num

        k = len(_clue)
        if len(_inter) != k:
            raise ValueError(f"_inter must have length {k} (same as clue), got {len(_inter)}")
        if any(x < 0 for x in _inter):
            raise ValueError("_inter must contain non-negative integers")

        min_required = sum(_clue) + (k - 1)  # k-1 mandatory single zeros between blocks
        if min_required > _grid_num:
            raise ValueError("Clue cannot fit in given grid length")

        total_extra = _grid_num - min_required
        used_extra = sum(_inter)
        if used_extra > total_extra:
            raise ValueError(f"Sum of _inter ({used_extra}) exceeds available extra spaces ({total_extra})")

        last_extra = total_extra - used_extra  # extra spaces after last block

        # Build the pattern
        parts = [[0] * _inter[0]]
        # Gap before first block: only extra (no mandatory zero)
        # Blocks and intermediate gaps
        for i in range(k):
            parts.append([1] * _clue[i])
            # After block i, if not last block: add mandatory 1 zero + extra from _inter[i+1]
            if i < k - 1:
                parts.append([0] * (1 + _inter[i + 1]))
        # Final gap after last block
        parts.append([0] * last_extra)

        result = [cell for part in parts for cell in part]
        assert len(result) == _grid_num, f"Generated length {len(result)} != {_grid_num}"
        return result

    def _get_col_result(self, col_num: int) -> list[int]:
        """Extract the current state of a column from solve_result.

        Args:
            col_num: Column index (0-based)

        Returns:
            A list representing the column state, where each element is -1 (uncertain), 0 (empty), or 1 (filled).
        """
        return [self.solve_result[row][col_num] for row in range(self.row)]

    def _get_row_result(self, row_num: int) -> list[int]:
        """Extract the current state of a row from solve_result.

        Args:
            row_num: Row index (0-based)

        Returns:
            A list representing the row state, where each element is -1 (uncertain), 0 (empty), or 1 (filled).
        """
        return self.solve_result[row_num]

    # def _states_chain(self):

