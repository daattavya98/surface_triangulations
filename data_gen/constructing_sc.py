from typing import Tuple
!pip install beartype
!pip install scipy.spatial
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
    init_adj_matrix : np.ndarray
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
    n_cycle_1: int, n_cycle_2: int, n_interior: int
) -> np.ndarray:
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

    Returns
    -------
    np.ndarray
        The generated vertices.
    """

    # Square vertices
    default_vertices = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])

    # Generate vertices
    left_right_vertices = np.random.rand(n_cycle_1)

    cycle_1_left = np.zeros((n_cycle_1, 2))
    cycle_1_left[:, 1] = left_right_vertices
    cycle_1_left = np.sort(cycle_1_left, axis=0)
    cycle_1_right = np.zeros((n_cycle_1, 2))
    cycle_1_right[:, 0] = 1
    cycle_1_right[:, 1] = left_right_vertices
    cycle_1_right = np.sort(cycle_1_right, axis=0)

    top_bottom_vertices = np.random.rand(n_cycle_2)

    cycle_2_bot = np.zeros((n_cycle_2, 2))
    cycle_2_top = np.zeros((n_cycle_2, 2))
    cycle_2_bot[:, 0] = top_bottom_vertices
    cycle_2_top[:, 0] = top_bottom_vertices
    cycle_2_top[:, 1] = 1
    cycle_2_top = np.sort(cycle_2_top, axis=0)
    cycle_2_bot = np.sort(cycle_2_bot, axis=0)

    interior = np.random.rand(n_interior, 2)

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


# Generate genus 0 triangulations
@beartype
def generate_genus_0_triangulations(
    n_cycle_1: int,
    n_cycle_2: int,
    n_interior: int,
) -> Tuple[np.ndarray, Delaunay, np.ndarray, np.ndarray]:
    """
    Generate genus 0 triangulations

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
        n_cycle_1=n_cycle_1, n_cycle_2=n_cycle_2, n_interior=n_interior
    )
    tri = Delaunay(points)

    adj_matrix = triangulation_to_adjacency(tri, len(points))
    quotiented_adj_matrix = adjacency_to_quotiented_adjacency(
        adj_matrix=adj_matrix.copy(),
        n_cycle_1=n_cycle_1,
        n_cycle_2=n_cycle_2,
        n_interior=n_interior,
        genus_0_tri=True,
        genus_1_tri=False,
    )

    return points, tri, adj_matrix, quotiented_adj_matrix


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
        n_cycle_1=n_cycle_1, n_cycle_2=n_cycle_2, n_interior=n_interior
    )
    tri = Delaunay(points)

    adj_matrix = triangulation_to_adjacency(tri, len(points))
    quotiented_adj_matrix = adjacency_to_quotiented_adjacency(
        adj_matrix=adj_matrix.copy(),
        n_cycle_1=n_cycle_1,
        n_cycle_2=n_cycle_2,
        n_interior=n_interior,
    )

    return points, tri, adj_matrix, quotiented_adj_matrix


@beartype
def main() -> None:
    """
    The main function
    """

    n_cycle_1, n_cycle_2, n_interior = 2, 2, 2

    points, tri, adj_matrix, quotiented_adj_matrix = generate_genus_1_triangulations(
        n_cycle_1=n_cycle_1, n_cycle_2=n_cycle_2, n_interior=n_interior
    )

    print(f"Sampled points: {points}")
    print(f"Adjacency matrix: {adj_matrix}")
    print(f"Quotiented adjacency matrix: {quotiented_adj_matrix}")

    # Plot the triangulation
    plt.triplot(points[:, 0], points[:, 1], tri.simplices)
    plt.plot(points[:, 0], points[:, 1], "o")
    plt.show()


if __name__ == "__main__":
    main()