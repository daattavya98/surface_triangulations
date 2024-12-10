from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from beartype import beartype
from scipy.spatial import Delaunay


@beartype
def triangulation_to_adjacency(
    triangulations: Delaunay, total_no_of_vertices: int
) -> np.ndarray:
    """
    This function converts a Delaunay triangulation to an adjacency matrix.

    Parameters
    ----------
    triangulations : Delaunay
        The inputted Delaunay triangulation.
    total_no_of_vertices : int
        The total number of unindentified vertices in the triangulation

    Returns
    -------
    np.ndarray
        The adjacency matrix of the triangulation.
    """

    n = total_no_of_vertices
    adj_matrix = np.zeros((n, n))

    for simplex in triangulations.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                p1, p2 = simplex[i], simplex[j]
                adj_matrix[p1, p2] = 1
                adj_matrix[p2, p1] = 1

    return adj_matrix


@beartype
def adjacency_to_quotiented_adjacency_column_addition(
    adj_matrix: np.ndarray,
    n_cycle_1: int,
    n_cycle_2: int,
    n_cycle_3: None | int = None,
    n_cycle_4: None | int = None,
    adj_matrix_second_square: None | np.ndarray = None,
    genus_0_tri: bool = False,
    genus_1_tri: bool = True,
) -> np.ndarray:
    """
    This function converts an adjacency matrix to a quotiented adjacency matrix.
    The ordering of the vertices is as follows:
    - 4 corners
    - n_cycle_1 left-right samples
    - n_cycle_2 top-bottom samples

    Parameters
    ----------
    adj_matrix : np.ndarray
        The inputted adjacency matrix.

    n_cycle_1 : int
        The number of left-right cycle samples.

    n_cycle_2 : int
        The number of top-bottom cycle samples.

    vertices_to_identify : np.ndarray
        The vertices to be identified.

    Returns
    -------
    quotiented_matrix : np.ndarray
        The quotiented adjacency matrix.
    """
    if genus_1_tri:
        n_cycle_1_starts = 4
        n_cycle_2_starts = 4 + 2 * n_cycle_1
        adj_matrix[0, :] += adj_matrix[1, :] + adj_matrix[2, :] + adj_matrix[3, :]
        adj_matrix[:, 0] += adj_matrix[:, 1] + adj_matrix[:, 2] + adj_matrix[:, 3]
        indices_to_delete = [1, 2, 3]

    elif genus_0_tri:
        if adj_matrix_second_square is None:
            raise ValueError(
                "adj_matrix_second_square must be provided for genus 0 triangulations"
            )

        adj_matrix[0, :] += adj_matrix_second_square[0, :]
        adj_matrix[:, 0] += adj_matrix_second_square[:, 0]
        adj_matrix[1, :] += adj_matrix_second_square[1, :]
        adj_matrix[:, 1] += adj_matrix_second_square[:, 1]
        adj_matrix[2, :] += adj_matrix_second_square[2, :]
        adj_matrix[:, 2] += adj_matrix_second_square[:, 2]
        adj_matrix[3, :] += adj_matrix_second_square[3, :]
        adj_matrix[:, 3] += adj_matrix_second_square[:, 3]

        n_cycle_1_starts = 4
        n_cycle_2_starts = 4 + n_cycle_1
        n_cycle_3_starts = n_cycle_2_starts + n_cycle_2
        n_cycle_4_starts = n_cycle_3_starts + n_cycle_3

    if genus_1_tri:
        # Initial addition
        for i in range(n_cycle_1):
            adj_matrix[n_cycle_1_starts + i, :] += adj_matrix[
                n_cycle_1_starts + n_cycle_1 + i, :
            ]
            adj_matrix[:, n_cycle_1_starts + i] += adj_matrix[
                :, n_cycle_1_starts + n_cycle_1 + i
            ]
        for i in range(n_cycle_2):
            adj_matrix[n_cycle_2_starts + i, :] += adj_matrix[
                n_cycle_2_starts + n_cycle_2 + i, :
            ]
            adj_matrix[:, n_cycle_2_starts + i] += adj_matrix[
                :, n_cycle_2_starts + n_cycle_2 + i
            ]

        adj_matrix[adj_matrix > 1] = 1

        for i in range(n_cycle_1):
            indices_to_delete.append(n_cycle_1_starts + n_cycle_1 + i)
        for i in range(n_cycle_2):
            indices_to_delete.append(n_cycle_2_starts + n_cycle_2 + i)

        adj_matrix = np.delete(adj_matrix, indices_to_delete, 0)
        adj_matrix = np.delete(adj_matrix, indices_to_delete, 1)

        quotiented_matrix = adj_matrix

    elif genus_0_tri:
        for i in range(n_cycle_1):
            adj_matrix[n_cycle_1_starts + i, 0] += adj_matrix_second_square[
                n_cycle_1_starts + i, 1
            ]
            adj_matrix[:, n_cycle_1_starts + i] += adj_matrix_second_square[
                :, n_cycle_1_starts + 1
            ]
        for i in range(n_cycle_2):
            adj_matrix[n_cycle_2_starts + i, 0] += adj_matrix_second_square[
                n_cycle_2_starts + i, 1
            ]
            adj_matrix[:, n_cycle_2_starts + i] += adj_matrix_second_square[
                :, n_cycle_2_starts + 1
            ]
        for i in range(n_cycle_3):
            adj_matrix[n_cycle_3_starts + i, 0] += adj_matrix_second_square[
                n_cycle_3_starts + i, 1
            ]
            adj_matrix[:, n_cycle_3_starts + i] += adj_matrix_second_square[
                :, n_cycle_3_starts + 1
            ]
        for i in range(n_cycle_4):
            adj_matrix[n_cycle_4_starts + i, 0] += adj_matrix_second_square[
                n_cycle_4_starts + i, 1
            ]
            adj_matrix[:, n_cycle_4_starts + i] += adj_matrix_second_square[
                :, n_cycle_4_starts + 1
            ]

        adj_matrix[adj_matrix > 1] = 1
        quotiented_matrix = adj_matrix

    return quotiented_matrix


@beartype
def adjacency_to_quotiented_adjacency(
    adj_matrix: np.ndarray,
    n_cycle_1: int,
    n_cycle_2: int,
    n_interior: int,
    genus_0_tri: bool = False,
    genus_1_tri: bool = True,
) -> np.ndarray:
    """
    This function converts an adjacency matrix to a quotiented adjacency matrix.
    The ordering of the vertices is as follows:
    - 4 corners
    - n_cycle_1 left-right samples
    - n_cycle_2 top-bottom samples
    - n_interior samples

    Parameters
    ----------
    adj_matrix : np.ndarray
        The inputted adjacency matrix.

    n_cycle_1 : int
        The number of left-right cycle samples.

    n_cycle_2 : int
        The number of top-bottom cycle samples.

    n_interior : int
        The number of interior samples.

    vertices_to_identify : np.ndarray
        The vertices to be identified.

    Returns
    -------
    quotiented_matrix : np.ndarray
        The quotiented adjacency matrix.
    """

    interior_starts = 4 + 2 * n_cycle_1 + 2 * n_cycle_2
    n_cycle_1_starts = 4
    n_cycle_2_starts = 4 + 2 * n_cycle_1

    if genus_1_tri:
        adj_matrix[0, :] += adj_matrix[1, :] + adj_matrix[2, :] + adj_matrix[3, :]
        indices_to_delete = [1, 2, 3]
    elif genus_0_tri:
        adj_matrix[0, :] += adj_matrix[2, :]
        indices_to_delete = [2]

    # Initial addition
    for i in range(n_cycle_1):
        adj_matrix[n_cycle_1_starts + i, :] += adj_matrix[
            n_cycle_1_starts + n_cycle_1 + i, :
        ]
    for i in range(n_cycle_2):
        adj_matrix[n_cycle_2_starts + i, :] += adj_matrix[
            n_cycle_2_starts + n_cycle_2 + i, :
        ]

    # Fix symmetry issues for edge-edge cases. In the case where edge-edge adjacency addition is not symmetric
    for i in range(n_cycle_1):
        for j in range(n_cycle_2):
            adj_matrix[n_cycle_1_starts + i, n_cycle_2_starts + j] += adj_matrix[
                n_cycle_1_starts + i, n_cycle_2_starts + n_cycle_2 + j
            ]

    for i in range(n_cycle_2):
        for j in range(n_cycle_1):
            adj_matrix[n_cycle_2_starts + i, n_cycle_1_starts + j] += adj_matrix[
                n_cycle_2_starts + i, n_cycle_1_starts + n_cycle_1 + j
            ]

    # Fix symmetry issues for edge-corner cases. There are two separate cases to consider: genus 0 and genus 1
    # This is becasue the columns for all corners that remain after quotienting need to be updated.
    # TODO: Add genus 0 case
    if genus_1_tri:
        for i in range(n_cycle_1):
            adj_matrix[n_cycle_1_starts + i, 0] += np.sum(
                adj_matrix[n_cycle_1_starts + i, indices_to_delete]
            )
        for i in range(n_cycle_2):
            adj_matrix[n_cycle_2_starts + i, 0] += np.sum(
                adj_matrix[n_cycle_2_starts + i, indices_to_delete]
            )

    # Perform interior addition
    for i in range(n_interior):
        for j in range(n_cycle_1):
            adj_matrix[interior_starts + i, n_cycle_1_starts] += adj_matrix[
                interior_starts + i, n_cycle_1_starts + n_cycle_1 + j
            ]
        for j in range(n_cycle_2):
            adj_matrix[interior_starts + i, n_cycle_2_starts] += adj_matrix[
                interior_starts + i, n_cycle_2_starts + n_cycle_2 + j
            ]

    # Fix symmetry issues for interior-edge cases
    for i in range(n_interior):
        for j in range(n_cycle_1):
            adj_matrix[interior_starts + i, n_cycle_1_starts + j] += adj_matrix[
                interior_starts + i, n_cycle_1_starts + n_cycle_1 + j
            ]
        for j in range(n_cycle_2):
            adj_matrix[interior_starts + i, n_cycle_2_starts + j] += adj_matrix[
                interior_starts + i, n_cycle_2_starts + n_cycle_2 + j
            ]

    # Fix symmetry issues for interior-corner cases. Again, two separate cases: genus 0 and genus 1
    # TODO: Add genus 0 case
    if genus_1_tri:
        for i in range(n_interior):
            adj_matrix[interior_starts + i, 0] += np.sum(
                adj_matrix[interior_starts + i, indices_to_delete]
            )

    adj_matrix[adj_matrix > 1] = 1

    for i in range(n_cycle_1):
        indices_to_delete.append(n_cycle_1_starts + n_cycle_1 + i)
    for i in range(n_cycle_2):
        indices_to_delete.append(n_cycle_2_starts + n_cycle_2 + i)

    adj_matrix = np.delete(adj_matrix, indices_to_delete, 0)
    adj_matrix = np.delete(adj_matrix, indices_to_delete, 1)

    quotiented_matrix = adj_matrix

    return quotiented_matrix


# Sample vertices from uniform distribution
@beartype
def sample_random_vertices(
    n_cycle_1: int,
    n_cycle_2: int,
    n_interior: int,
    genus: int = 1,
    n_second_square_interior: None | int = None,
    n_cycle_3: None | int = None,
    n_cycle_4: None | int = None,
) -> np.ndarray | List[np.ndarray]:
    """
    This function samples random vertices from a uniform distribution over a square

    Parameters
    ----------
    n_cycle_1 : int
        The number of vertices on the left and right edges.
    n_cycle_2 : int
        The number of vertices on the top and bottom edges.
    n_interior : int
        The number of vertices in the interior.
    genus : int
        The genus of the surface.
    n_second_square_interior : None | int
        The number of vertices in the interior of the second square if genus is 0, none otherwise.
    n_cycle_3 : None | int
        The number of vertices on the edge for identification type 3 if genus is 0, none otherwise.
    n_cycle_4 : None | int
        The number of vertices on the edge for identification type 4 if genus is 0, none otherwise.

    Returns
    -------
    np.ndarray
        The generated vertices.
    """

    if genus == 1:
        # Square vertices
        default_vertices = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        interior = np.random.rand(n_interior, 2)
        # Generate vertices
        left_right_vertices = np.random.rand(n_cycle_1)

        cycle_1_left = np.zeros((n_cycle_1, 2))
        cycle_1_left[:, 1] = left_right_vertices
        cycle_1_left = np.sort(cycle_1_left, axis=0)
        cycle_1_right = np.zeros((n_cycle_1, 2))
        cycle_1_right[:, 0] = 1
        cycle_1_right[:, 1] = left_right_vertices
        cycle_1_right = np.sort(cycle_1_right, axis=0)

        bottom_top_vertices = np.random.rand(n_cycle_2)

        cycle_2_bot = np.zeros((n_cycle_2, 2))
        cycle_2_top = np.zeros((n_cycle_2, 2))
        cycle_2_bot[:, 0] = bottom_top_vertices
        cycle_2_top[:, 0] = bottom_top_vertices
        cycle_2_top[:, 1] = 1
        cycle_2_top = np.sort(cycle_2_top, axis=0)
        cycle_2_bot = np.sort(cycle_2_bot, axis=0)

        return np.concatenate(
            [
                default_vertices,
                cycle_1_left,
                cycle_1_right,
                cycle_2_bot,
                cycle_2_top,
                interior,
            ]
        )

    elif genus == 0:
        if not n_cycle_3 or not n_cycle_4 or not n_second_square_interior:
            raise ValueError(
                "n_cycle_3, n_cycle_4, and n_second_square_interior must be provided for genus 0 \
                             triangulations"
            )

        # Double square vertices
        default_vertices = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        # Sample interior vertices
        n_interior = np.random.rand(n_interior, 2)
        n_second_square_interior = np.random.rand(n_second_square_interior, 2)

        # Generate vertices
        left_edge_square_1_left_edge_square_2 = np.random.rand(n_cycle_1)
        right_edge_square_1_right_edge_square_2 = np.random.rand(n_cycle_2)
        top_edge_square_1_top_edge_square_2 = np.random.rand(n_cycle_3)
        bottom_edge_square_1_bottom_edge_square_2 = np.random.rand(n_cycle_4)

        # Join left edge of square 1 to left edge of square 2
        left_edge_square_1_vertices = np.zeros((n_cycle_1, 2))
        left_edge_square_1_vertices[:, 1] = left_edge_square_1_left_edge_square_2
        left_edge_square_1_vertices = np.sort(left_edge_square_1_vertices, axis=0)
        left_edge_square_2_vertices = left_edge_square_1_vertices

        # Join right edge of square 1 to right edge of square 2
        right_edge_square_1_vertices = np.zeros((n_cycle_2, 2))
        right_edge_square_1_vertices[:, 0] = 1
        right_edge_square_1_vertices[:, 1] = right_edge_square_1_right_edge_square_2
        right_edge_square_1_vertices = np.sort(right_edge_square_1_vertices, axis=0)
        right_edge_square_2_vertices = right_edge_square_1_vertices

        # Join top edge of square 1 to top edge of square 2
        top_edge_square_1_vertices = np.zeros((n_cycle_3, 2))
        top_edge_square_1_vertices[:, 0] = top_edge_square_1_top_edge_square_2
        top_edge_square_1_vertices[:, 1] = 1
        top_edge_square_1_vertices = np.sort(top_edge_square_1_vertices, axis=0)
        top_edge_square_2_vertices = top_edge_square_1_vertices

        # Join bottom edge of square 1 to bottom edge of square 2
        bottom_edge_square_1_vertices = np.zeros((n_cycle_4, 2))
        bottom_edge_square_1_vertices[:, 0] = bottom_edge_square_1_bottom_edge_square_2
        bottom_edge_square_1_vertices = np.sort(bottom_edge_square_1_vertices, axis=0)
        bottom_edge_square_2_vertices = bottom_edge_square_1_vertices

        return [
            np.concatenate(
                [
                    default_vertices,
                    left_edge_square_1_vertices,
                    right_edge_square_1_vertices,
                    top_edge_square_1_vertices,
                    bottom_edge_square_1_vertices,
                    n_interior,
                ]
            ),
            np.concatenate(
                [
                    default_vertices,
                    left_edge_square_2_vertices,
                    right_edge_square_2_vertices,
                    top_edge_square_2_vertices,
                    bottom_edge_square_2_vertices,
                    n_second_square_interior,
                ]
            ),
        ]


# Generate genus 0 triangulations
@beartype
def generate_genus_0_triangulations(
    n_cycle_1: int,
    n_cycle_2: int,
    n_interior: int,
    n_cycle_3: int,
    n_cycle_4: int,
    n_second_square_interior: int,
) -> Tuple[
    np.ndarray, np.ndarray, Delaunay, Delaunay, np.ndarray, np.ndarray, np.ndarray
]:
    """
    Generate genus 0 triangulations

    Parameters
    ----------
    n_cycle_1 : int
        The number of vertices on left edge.
    n_cycle_2 : int
        The number of vertices on the right edge.
    n_cycle_3 : int
        The number of vertices on the top edge.
    n_cycle_4 : int
        The number of vertices on the bottom edge.
    n_interior : int
        The number of vertices in the interior of square 1.
    n_second_square_interior : int
        The number of vertices in the interior of square 2.

    Returns
    -------
    Tuple[np.ndarray, Delaunay, np.ndarray, np.ndarray]
        The generated points, triangulation, adjacency matrix, and quotiented adjacency
    """

    points = sample_random_vertices(
        n_cycle_1=n_cycle_1,
        n_cycle_2=n_cycle_2,
        n_interior=n_interior,
        genus=0,
        n_cycle_3=n_cycle_3,
        n_cycle_4=n_cycle_4,
        n_second_square_interior=n_second_square_interior,
    )
    points_square1 = points[0]
    points_square2 = points[1]

    tri1 = Delaunay(points_square1)
    tri2 = Delaunay(points_square2)

    adj_matrix = triangulation_to_adjacency(tri1, len(points_square1))
    adj_matrix_second_square = triangulation_to_adjacency(tri2, len(points_square2))

    quotiented_adj_matrix = adjacency_to_quotiented_adjacency_column_addition(
        adj_matrix=adj_matrix.copy(),
        adj_matrix_second_square=adj_matrix_second_square.copy(),
        n_cycle_1=n_cycle_1,
        n_cycle_2=n_cycle_2,
        n_cycle_3=n_cycle_3,
        n_cycle_4=n_cycle_4,
        genus_0_tri=True,
        genus_1_tri=False,
    )

    return (
        points_square1,
        points_square2,
        tri1,
        tri2,
        adj_matrix,
        adj_matrix_second_square,
        quotiented_adj_matrix,
    )


# Generate genus 1 triangulations
@beartype
def generate_genus_1_triangulations(
    n_cycle_1: int,
    n_cycle_2: int,
    n_interior: int,
) -> Tuple[np.ndarray, Delaunay, np.ndarray, np.ndarray]:
    """
    Generate genus 1 triangulations

    Parameters
    ----------
    n_cycle_1 : int
        The number of cycles of type 1.
    n_cycle_2 : int
        The number of cycles of type 2.
    n_interior : int
        The number of interior points.

    Returns
    -------
    Tuple[np.ndarray, Delaunay, np.ndarray, np.ndarray]
        The generated points, triangulation, adjacency matrix, and quotiented adjacency
    """

    points = sample_random_vertices(
        n_cycle_1=n_cycle_1, n_cycle_2=n_cycle_2, n_interior=n_interior, genus=1
    )
    tri = Delaunay(points)

    adj_matrix = triangulation_to_adjacency(tri, len(points))
    quotiented_adj_matrix = adjacency_to_quotiented_adjacency_column_addition(
        n_cycle_1=n_cycle_1,
        n_cycle_2=n_cycle_2,
        n_interior=n_interior,
        adj_matrix=adj_matrix.copy(),
    )

    return points, tri, adj_matrix, quotiented_adj_matrix


@beartype
def main() -> None:
    """
    The main function
    """

    n_cycle_1, n_cycle_2, n_cycle_3, n_cycle_4, n_interior, n_interior_second_square = (
        2,
        2,
        2,
        2,
        2,
        1,
    )

    (
        points_square1,
        points_square2,
        tri1,
        tri2,
        adj_matrix,
        adj_matrix_second_square,
        quotiented_adj_matrix,
    ) = generate_genus_0_triangulations(
        n_cycle_1=n_cycle_1,
        n_cycle_2=n_cycle_2,
        n_interior=n_interior,
        n_cycle_3=n_cycle_3,
        n_cycle_4=n_cycle_4,
        n_second_square_interior=n_interior_second_square,
    )

    print(f"Sampled points: {points_square1}, {points_square2}")
    print(f"Adjacency matrix: {adj_matrix}, {adj_matrix_second_square}")
    print(f"Quotiented adjacency matrix: {quotiented_adj_matrix}")

    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 2)

    # Plot the first triangulation in the first subplot
    axs[0].triplot(points_square1[:, 0], points_square1[:, 1], tri1.simplices)
    axs[0].plot(points_square1[:, 0], points_square1[:, 1], "o")
    axs[0].set_title("Triangulation 1")

    # Plot the second triangulation in the second subplot
    axs[1].triplot(points_square2[:, 0], points_square2[:, 1], tri2.simplices)
    axs[1].plot(points_square2[:, 0], points_square2[:, 1], "o")
    axs[1].set_title("Triangulation 2")

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Show the figure
    plt.show()


if __name__ == "__main__":
    main()
