import os
from itertools import combinations
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import toponetx as tnx
from numpy.linalg import matrix_rank
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import Delaunay


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


def sample_random_vertices(
    n_cycle_1: int,
    n_cycle_2: int,
    n_interior: int,
    genus: int = 1,
    n_second_square_interior: None | int = 0,
    n_cycle_3: None | int = 0,
    n_cycle_4: None | int = 0,
    n_diagonal_1_square_1: None | int = 0,
    n_diagonal_2_square_1: None | int = 0,
    n_diagonal_1_square_2: None | int = 0,
    n_diagonal_2_square_2: None | int = 0,
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

    if genus == 1:

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

    elif genus == 0:

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

        # Create points along the diagonal_1 of square 1
        diagonal_1_square_1 = np.zeros((n_diagonal_1_square_1, 2))
        corner_diag_point1 = np.random.uniform(0, 0.05)
        corner_diag_point2 = np.random.uniform(0.95, 1)
        corner_diagonal_point1 = np.array([corner_diag_point1, corner_diag_point1])
        corner_diagonal_point1 = corner_diagonal_point1.reshape(1, -1)
        corner_diagonal_point2 = np.array([corner_diag_point2, corner_diag_point2])
        corner_diagonal_point2 = corner_diagonal_point2.reshape(1, -1)
        for i in range(n_diagonal_1_square_1):
            diag_point = np.random.uniform(0, 1)
            diagonal_1_square_1[i] = np.array([diag_point, diag_point])
        diagonal_1_square_1 = np.concatenate(
            [corner_diagonal_point1, diagonal_1_square_1]
        )
        diagonal_1_square_1 = np.concatenate(
            [diagonal_1_square_1, corner_diagonal_point2]
        )

        # Create points along the diagonal_2 of square 1
        diagonal_2_square_1 = np.zeros((n_diagonal_2_square_1, 2))
        corner_diag_point1 = np.random.uniform(0, 0.05)
        corner_diag_point2 = np.random.uniform(0.95, 1)
        corner_diagonal_point1 = np.array([corner_diag_point1, 1 - corner_diag_point1])
        corner_diagonal_point1 = corner_diagonal_point1.reshape(1, -1)
        corner_diagonal_point2 = np.array([corner_diag_point2, 1 - corner_diag_point2])
        corner_diagonal_point2 = corner_diagonal_point2.reshape(1, -1)
        for i in range(n_diagonal_2_square_1):
            diag_point = np.random.uniform(0, 1)
            diagonal_2_square_1[i] = np.array([diag_point, 1 - diag_point])
        diagonal_2_square_1 = np.concatenate(
            [corner_diagonal_point1, diagonal_2_square_1]
        )
        diagonal_2_square_1 = np.concatenate(
            [diagonal_2_square_1, corner_diagonal_point2]
        )

        # Create points along the diagonal_1 of square 2
        diagonal_1_square_2 = np.zeros((n_diagonal_1_square_2, 2))
        corner_diag_point1 = np.random.uniform(0, 0.05)
        corner_diag_point2 = np.random.uniform(0.95, 1)
        corner_diagonal_point1 = np.array([corner_diag_point1, corner_diag_point1])
        corner_diagonal_point1 = corner_diagonal_point1.reshape(1, -1)
        corner_diagonal_point2 = np.array([corner_diag_point2, corner_diag_point2])
        corner_diagonal_point2 = corner_diagonal_point2.reshape(1, -1)
        for i in range(n_diagonal_1_square_2):
            diag_point = np.random.uniform(0, 1)
            diagonal_1_square_2[i] = np.array([diag_point, diag_point])
        diagonal_1_square_2 = np.concatenate(
            [corner_diagonal_point1, diagonal_1_square_2]
        )
        diagonal_1_square_2 = np.concatenate(
            [diagonal_1_square_2, corner_diagonal_point2]
        )

        # Create points along the diagonal_2 of square 2
        diagonal_2_square_2 = np.zeros((n_diagonal_2_square_2, 2))
        corner_diag_point1 = np.random.uniform(0, 0.05)
        corner_diag_point2 = np.random.uniform(0.95, 1)
        corner_diagonal_point1 = np.array([corner_diag_point1, 1 - corner_diag_point1])
        corner_diagonal_point1 = corner_diagonal_point1.reshape(1, -1)
        corner_diagonal_point2 = np.array([corner_diag_point2, 1 - corner_diag_point2])
        corner_diagonal_point2 = corner_diagonal_point2.reshape(1, -1)
        for i in range(n_diagonal_2_square_2):
            diag_point = np.random.uniform(0, 1)
            diagonal_2_square_2[i] = np.array([diag_point, 1 - diag_point])
        diagonal_2_square_2 = np.concatenate(
            [corner_diagonal_point1, diagonal_2_square_2]
        )
        diagonal_2_square_2 = np.concatenate(
            [diagonal_2_square_2, corner_diagonal_point2]
        )

        return [
            np.concatenate(
                [
                    default_vertices,
                    left_edge_square_1_vertices,
                    right_edge_square_1_vertices,
                    top_edge_square_1_vertices,
                    bottom_edge_square_1_vertices,
                    n_interior,
                    diagonal_1_square_1,
                    diagonal_2_square_1,
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
                    diagonal_1_square_2,
                    diagonal_2_square_2,
                ]
            ),
        ]


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
        n_cycle_1=n_cycle_1, n_cycle_2=n_cycle_2, n_interior=n_interior, genus=1
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


# def fix_boundary_edges_genus_0(previous_tri_list: List, square_simplices: np.ndarray, new_label_counter: int):
#     """
#     Fix the gluing of the triangulation for the boundary edges in genus 0.

#     Parameters
#     ----------
#     square_simplices : np.ndarray
#         Second square simplices
#     starting_new_label_counter : int
#         Starting label for new vertices
#     """
#     triangles_to_remove_from_square_simplices = []
#     previous_label_counter = new_label_counter
#     previous_triangle = []
#     for corner_el in range(4):
#         for triangle in square_simplices:
#             if corner_el in triangle:
#                 remaining_indices = [other_el for other_el in triangle if other_el != corner_el]
#                 new_triangle1 = [corner_el, remaining_indices[0], new_label_counter]
#                 new_triangle2 = [corner_el, remaining_indices[1], new_label_counter]
#                 previous_tri_list.append(new_triangle1)
#                 previous_tri_list.append(new_triangle2)
#                 previous_label_counter = new_label_counter
#                 previous_triangle = triangle
#                 new_label_counter += 1
#                 triangles_to_remove_from_square_simplices.append(triangle)
#             else:


def construct_simplicial_complex(
    quotiented_adj_matrix: np.ndarray,
    simplices: np.ndarray,
    n_cycle_1: int,
    n_cycle_2: int,
    n_interior: int,
) -> tnx.SimplicialComplex:
    """
    Construct a simplicial complex from a quotiented adjacency matrix that respects the original Delaunay triangulation.

    Parameters
    ----------
    quotiented_adj_matrix : np.ndarray
        The inputted quotiented adjacency matrix.
    triangulation : Delaunay
        The original Delaunay triangulation.

    Returns
    -------
    tnx.SimplicialComplex
        The constructed simplicial complex.
    """

    # one_simplex = []
    # for i in range(quotiented_adj_matrix.shape[0]):
    #     for j in range(quotiented_adj_matrix.shape[1]):
    #         if quotiented_adj_matrix[i, j] == 1:
    #             if (i, j) not in one_simplex and (j, i) not in one_simplex:
    #                 one_simplex.append((i, j))
    # one_simplex = list(set(one_simplex))
    # one_simplex = np.array(one_simplex)

    two_simplex = []
    tri_list = []

    for triangle in simplices:
        for el in triangle:
            if el in [1, 2, 3]:
                triangle[triangle == el] = 0
            for i in range(4 + n_cycle_1, 4 + 2 * n_cycle_1):
                if el == i:
                    triangle[triangle == el] = i - n_cycle_1
            for i in range(
                4 + 2 * n_cycle_1 + n_cycle_2, 4 + 2 * n_cycle_1 + 2 * n_cycle_2
            ):
                if el == i:
                    triangle[triangle == el] = i - n_cycle_2
        tri_list.append(triangle)

    tri_list = list(set(tuple(triangle) for triangle in tri_list))

    two_simplex = np.array(tri_list)

    sc = tnx.SimplicialComplex(two_simplex)
    return sc


def construct_simplicial_complex_genus_0(
    n_cycle_1: int,
    n_cycle_2: int,
    n_cycle_3: int,
    n_cycle_4: int,
    n_interior: int,
    n_interior_square_2: int,
    n_diagonal_1_square_1: int,
    n_diagonal_2_square_1: int,
    n_diagonal_1_square_2: int,
    n_diagonal_2_square_2: int,
) -> tnx.SimplicialComplex:

    points = sample_random_vertices(
        n_cycle_1=n_cycle_1,
        n_cycle_2=n_cycle_2,
        n_interior=n_interior,
        n_second_square_interior=n_interior_square_2,
        n_cycle_3=n_cycle_3,
        n_cycle_4=n_cycle_4,
        genus=0,
        n_diagonal_1_square_1=n_diagonal_1_square_1,
        n_diagonal_2_square_1=n_diagonal_2_square_1,
        n_diagonal_1_square_2=n_diagonal_1_square_2,
        n_diagonal_2_square_2=n_diagonal_2_square_2,
    )

    points_square_1 = points[0]
    points_square_2 = points[1]

    tri_square_1 = Delaunay(points_square_1)
    tri_square_2 = Delaunay(points_square_2)

    two_simplex = []
    tri_list = tri_square_1.simplices.tolist()

    # second_square_interior_start_index = (
    #     4 + n_cycle_1 + n_cycle_2 + n_cycle_3 + n_cycle_4 + n_interior
    # )

    extra_index_counter = (
        4 + n_cycle_1 + n_cycle_2 + n_cycle_3 + n_cycle_4 + n_interior + 4
    )
    +n_diagonal_1_square_1 + n_diagonal_2_square_1
    # # Check the double corner-boundary triangles and update the labelling to deal with this
    # for tri1 in tri_square_1.simplices:
    #     if any(el in [0, 1 , 2, 3] for el in tri1):
    #         remaining_indices = [el for el in tri1 if el not in [0, 1, 2, 3]]
    #         if not any(
    #             el in range(4 + n_cycle_1 + n_cycle_2 + n_cycle_3 + n_cycle_4,
    #                         4 + n_cycle_1 + n_cycle_2 + n_cycle_3 + n_cycle_4 + n_interior) for el in
    #                   remaining_indices):

    #             for triangle_2 in tri_square_2.simplices:
    #                 if np.array_equal(tri1, triangle_2):
    #                     corner_index = np.min(triangle_2)
    #                     remaining_indices = [el for el in triangle_2 if el != corner_index]
    #                     new_triangle_1 = [corner_index, remaining_indices[0], extra_index_counter]
    #                     new_triangle_2 = [corner_index, remaining_indices[1], extra_index_counter]
    #                     extra_index_counter += 1
    #                     tri_list.append(new_triangle_1)
    #                     tri_list.append(new_triangle_2)
    #                     # Use boolean indexing to filter out the target array
    #                     tri_square_2.simplices = np.array([row for row in tri_square_2.simplices
    #                                                        if not np.array_equal(row, triangle_2)])

    # for tri1 in tri_square_1.simplices:
    #     for el in tri1:
    #         if el + 1 in tri1:
    #             # Deal with boundary-boundary edge case
    #             # Check if such a triangle exists in the second square
    #             for tri2 in tri_square_2.simplices:
    #                 # if tri2 == tri1:
    #                 pass

    # else:
    #     boundary_range = range(4, 4 + n_cycle_1 + n_cycle_2 + n_cycle_3 + n_cycle_4)
    #     pairs = combinations(tri1, 2)

    #     for pair in pairs:
    #         flag = False
    #         if all(el in boundary_range for el in pair):
    #             boundary_pair_indices = pair
    #             remaining_third_vertex = [el for el in tri1 if el not in boundary_range][0]
    #             new_triangle_bnd_pair_1 = [boundary_pair_indices[0], boundary_pair_indices[1], extra_index_counter]
    #             new_triangle_bnd_pair_2 = [boundary_pair_indices[0], remaining_third_vertex,  extra_index_counter]
    #             new_triangle_bnd_pair_3 = [boundary_pair_indices[1], remaining_third_vertex,  extra_index_counter]
    #             extra_index_counter += 1
    #             tri_list.append(new_triangle_bnd_pair_1)
    #             tri_list.append(new_triangle_bnd_pair_2)
    #             tri_list.append(new_triangle_bnd_pair_3)
    #             tri_list = [row for row in tri_list if not np.array_equal(row, tri1)]
    #             flag = True
    #             break
    #         if flag:
    #             break
    list_of_index_replacements = []
    for triangle in tri_square_2.simplices:
        for el in triangle:
            if el in range(
                4 + n_cycle_1 + n_cycle_2 + n_cycle_3 + n_cycle_4,
                8
                + n_cycle_1
                + n_cycle_2
                + n_cycle_3
                + n_cycle_4
                + n_interior_square_2
                + n_diagonal_1_square_2
                + n_diagonal_2_square_2,
            ):
                if len(list_of_index_replacements) == 0:
                    triangle[triangle == el] = extra_index_counter + el - 1
                    list_of_index_replacements.append(
                        (el, extra_index_counter + el - 1)
                    )
                else:
                    flag = False
                    for row in list_of_index_replacements:
                        if el == row[0]:
                            triangle[triangle == el] = row[1]
                            flag = True
                            break
                    if not flag:
                        max_value = max(list_of_index_replacements, key=lambda x: x[1])[
                            1
                        ]
                        triangle[triangle == el] = max_value + 1
                        list_of_index_replacements.append((el, max_value + 1))

        # if not any(np.array_equal(arr, triangle) for arr in tri_list):
        tri_list.append(triangle)

    tri_list = list(set(tuple(triangle) for triangle in tri_list))
    two_simplex = np.array(tri_list)

    sc = tnx.SimplicialComplex(two_simplex)

    tri_square_1 = Delaunay(points_square_1)
    tri_square_2 = Delaunay(points_square_2)

    return sc, tri_square_1, tri_square_2, points_square_1, points_square_2


def create_link_graph(incident_edges, E, link_faces):
    # Create the adjacency matrix for the link graph
    link_graph = np.zeros((len(link_faces), len(link_faces)))
    link_faces = list(link_faces)

    for edge in incident_edges:
        # Find the faces sharing this edge
        edge_faces = np.where(E[edge].toarray().flatten() != 0)[0]
        for i in range(len(edge_faces)):
            for j in range(i + 1, len(edge_faces)):
                if edge_faces[i] in link_faces and edge_faces[j] in link_faces:
                    idx1 = link_faces.index(edge_faces[i])
                    idx2 = link_faces.index(edge_faces[j])
                    link_graph[idx1, idx2] = 1
                    link_graph[idx2, idx1] = 1
    return link_graph


def create_1skel_adjacency(incidence_matrix):
    # Number of vertices (rows) and edges (columns)
    num_vertices, num_edges = incidence_matrix.shape

    # Create an adjacency matrix for the graph (initially all zeros)
    adj_matrix = np.zeros((num_vertices, num_vertices), dtype=int)

    # For each edge, find the two vertices that it connects
    for edge in range(num_edges):
        # Get the vertices that are incident to this edge (where incidence_matrix[:, edge] != 0)
        vertices = np.where(incidence_matrix[:, edge] != 0)[0]

        # If two or more vertices are part of the edge, connect them
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                # Connect the vertices with the appropriate sign
                adj_matrix[vertices[i], vertices[j]] = incidence_matrix[
                    vertices[i], edge
                ]
                adj_matrix[vertices[j], vertices[i]] = incidence_matrix[
                    vertices[j], edge
                ]

    # Convert the adjacency matrix to a sparse matrix format (CSR format)
    sparse_adj_matrix = csr_matrix(adj_matrix)
    return sparse_adj_matrix


def check_surface_homeomorphic(vertex_edge_matrix, edge_face_matrix):
    """
    Check if a simplicial complex represents a surface homeomorphic to a topological manifold.

    Args:
        vertex_edge_matrix (ndarray): Incidence matrix between vertices and edges.
        edge_face_matrix (ndarray): Incidence matrix between edges and faces.

    Returns:
        bool: True if the complex represents a surface homeomorphic to a topological manifold, False otherwise.
    """
    # Convert incidence matrices to sparse format for efficiency
    V = csr_matrix(vertex_edge_matrix)  # Vertex-edge incidence matrix
    E = csr_matrix(edge_face_matrix)  # Edge-face incidence matrix

    # Step 1: Check if links of all vertices are topological circles
    for vertex in range(V.shape[0]):
        # Find edges incident to the vertex
        incident_edges = V[vertex].nonzero()[1]

        # Find faces incident to these edges
        link_faces = set()
        for edge in incident_edges:
            link_faces.update(E[edge].nonzero()[1])

        # Create the adjacency matrix for the link graph
        link_graph = create_link_graph(incident_edges, E, link_faces)

        # print(f"Link graph for vertex {vertex}:\n{link_graph}")

        # Check if the link graph is a single connected cycle
        n_components, _ = connected_components(
            csr_matrix(link_graph), connection="strong"
        )
        degree_counts = np.sum(link_graph, axis=1)

        if n_components != 1 or np.any(degree_counts != 2):
            print(f"Link check failed for vertex {vertex}")
            return False

    # Step 2: Check if the entire complex (1-skeleton) is connected

    # Compute the adjacency matrix for the 1-skeleton (vertex-vertex adjacency)
    adjacency_matrix = create_1skel_adjacency(vertex_edge_matrix)

    # Use connected_components to check if the 1-skeleton is connected
    n_components, _ = connected_components(adjacency_matrix, directed=True)

    # The complex is connected if there's only one connected component
    return n_components == 1


def generate_genus_1_datapoints(
    n_lower: int = 5, n_upper: int = 25, no_of_points: int = 25
) -> np.ndarray:

    genus_1_datapoints = np.ndarray((no_of_points, 2), dtype=object)
    cnt = 0
    i = 0
    while cnt < no_of_points:
        n_cycle_1 = np.random.randint(n_lower, n_upper)
        n_cycle_2 = np.random.randint(n_lower, n_upper)
        n_interior = np.random.randint(n_lower, n_upper)

        points, tri, adj_matrix, quotiented_adj_matrix = (
            generate_genus_1_triangulations(
                n_cycle_1=n_cycle_1, n_cycle_2=n_cycle_2, n_interior=n_interior
            )
        )

        # Construct simplicial complex
        sc = construct_simplicial_complex(
            quotiented_adj_matrix.copy(),
            tri.simplices.copy(),
            n_cycle_1,
            n_cycle_2,
            n_interior,
        )

        D1 = sc.incidence_matrix(1).todense()
        D2 = sc.incidence_matrix(2).todense()
        ker_D1 = np.shape(D1)[1] - matrix_rank(D1)
        H1 = ker_D1 - matrix_rank(D2)

        # print(B1)
        # print(B2)

        # print("The 1st homology is", H1)
        is_surface = check_surface_homeomorphic(D1, D2)
        # print("The complex represents a surface:", is_surface)

        if is_surface and H1 == 2:
            if not any(np.array_equal(arr, [D1, D2]) for arr in genus_1_datapoints):
                genus_1_datapoints[i][0] = D1
                genus_1_datapoints[i][1] = D2
                cnt += 1

        if cnt == no_of_points:
            break
        elif i == no_of_points - 1:
            i = 0
            continue

        i += 1

    # # Filter out None values
    # filtered_rows = [row for row in genus_1_datapoints if row is not None and row[0] is not None and row[1] is \
    # not None]
    # genus_1_datapoints = np.array(filtered_rows, dtype=object)

    genus_1_datapoints = np.array(genus_1_datapoints, dtype=object)

    return genus_1_datapoints


def generate_genus_0_datapoints(
    n_cycle_1: int = 2,
    n_cycle_2: int = 3,
    n_cycle_3: int = 2,
    n_cycle_4: int = 4,
    n_interior: int = 3,
    n_interior_square_2: int = 4,
    n_diagonal_1_square_1: int = 2,
    n_diagonal_2_square_1: int = 3,
    n_diagonal_1_square_2: int = 2,
    n_diagonal_2_square_2: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:

    [sc_sphere, tri_square_1, tri_square_2, points_square_1, points_square_2] = (
        construct_simplicial_complex_genus_0(
            n_cycle_1=n_cycle_1,
            n_cycle_2=n_cycle_2,
            n_cycle_3=n_cycle_3,
            n_cycle_4=n_cycle_4,
            n_interior=n_interior,
            n_interior_square_2=n_interior_square_2,
            n_diagonal_1_square_1=n_diagonal_1_square_1,
            n_diagonal_2_square_1=n_diagonal_2_square_1,
            n_diagonal_1_square_2=n_diagonal_1_square_2,
            n_diagonal_2_square_2=n_diagonal_2_square_2,
        )
    )

    D1 = sc_sphere.incidence_matrix(1).todense()
    D2 = sc_sphere.incidence_matrix(2).todense()

    ker_D1 = np.shape(D1)[1] - matrix_rank(D1)
    H1 = ker_D1 - matrix_rank(D2)

    print(D1.shape)
    print(D2.shape)
    print("The 1st homology is", H1)
    is_surface = check_surface_homeomorphic(D1, D2)
    print("The complex represents a surface:", is_surface)

    return [tri_square_1, tri_square_2, points_square_1, points_square_2]


def save_datapoints(datapoints, folder_path, file_name):
    file_path = os.path.join(folder_path, file_name)
    np.save(file_path, datapoints)


def main() -> None:

    # genus_1_datapoints = generate_genus_1_datapoints(
    #     n_lower=5, n_upper=100, no_of_points=200
    # )
    # print(genus_1_datapoints.shape)
    # print(genus_1_datapoints[0])

    # save_datapoints(
    #     genus_1_datapoints,
    #     "data_gen/incidence_matrix_data",
    #     "tori_incidence_matrices.npy",
    # )

    [tri1, tri2, points_square1, points_square2] = generate_genus_0_datapoints(
        n_cycle_1=5,
        n_cycle_2=12,
        n_interior=8,
        n_cycle_3=7,
        n_cycle_4=9,
        n_interior_square_2=8,
        n_diagonal_1_square_1=5,
        n_diagonal_2_square_1=7,
        n_diagonal_1_square_2=4,
        n_diagonal_2_square_2=3,
    )
    # points = sample_random_vertices()

    # points_square1 = points[0]
    # points_square2 = points[1]

    # tri1 = Delaunay(points_square1)
    # tri2 = Delaunay(points_square2)
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

    plt.show()


if __name__ == "__main__":
    main()
