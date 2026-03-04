import hashlib
import os
import random
import traceback
from typing import Tuple

import numpy as np
import toponetx as tnx
from numpy.linalg import matrix_rank
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import Delaunay

rng = np.random.default_rng(seed=42)


def _array_key(a: np.ndarray) -> tuple:
    """
    Stable, hashable key for a numpy array based on dtype/shape/bytes.
    """
    a = np.asarray(a)
    a = np.ascontiguousarray(a)  # ensure deterministic tobytes()
    h = hashlib.blake2b(a.tobytes(), digest_size=16).hexdigest()
    return (str(a.dtype), a.shape, h)


def _pair_key(D1: np.ndarray, D2: np.ndarray) -> tuple:
    # ordered pair key; for unordered use tuple(sorted([_array_key(D1), _array_key(D2)]))
    return (_array_key(D1), _array_key(D2))


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
        left_right_vertices = rng.random(n_cycle_1)

        cycle_1_left = np.zeros((n_cycle_1, 2))
        cycle_1_left[:, 1] = left_right_vertices
        cycle_1_left = np.sort(cycle_1_left, axis=0)
        cycle_1_right = np.zeros((n_cycle_1, 2))
        cycle_1_right[:, 0] = 1
        cycle_1_right[:, 1] = left_right_vertices
        cycle_1_right = np.sort(cycle_1_right, axis=0)

        top_bottom_vertices = rng.random(n_cycle_2)

        cycle_2_bot = np.zeros((n_cycle_2, 2))
        cycle_2_top = np.zeros((n_cycle_2, 2))
        cycle_2_bot[:, 0] = top_bottom_vertices
        cycle_2_top[:, 0] = top_bottom_vertices
        cycle_2_top[:, 1] = 1
        cycle_2_top = np.sort(cycle_2_top, axis=0)
        cycle_2_bot = np.sort(cycle_2_bot, axis=0)

        interior = rng.random((n_interior, 2))

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
        n_interior = rng.random((n_interior, 2))
        n_second_square_interior = rng.random((n_second_square_interior, 2))

        # Generate vertices
        left_edge_square_1_left_edge_square_2 = rng.random(n_cycle_1)
        right_edge_square_1_right_edge_square_2 = rng.random(n_cycle_2)
        top_edge_square_1_top_edge_square_2 = rng.random(n_cycle_3)
        bottom_edge_square_1_bottom_edge_square_2 = rng.random(n_cycle_4)

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
        corner_diag_point1 = rng.uniform(0, 0.05)
        corner_diag_point2 = rng.uniform(0.95, 1)
        corner_diagonal_point1 = np.array([corner_diag_point1, corner_diag_point1])
        corner_diagonal_point1 = corner_diagonal_point1.reshape(1, -1)
        corner_diagonal_point2 = np.array([corner_diag_point2, corner_diag_point2])
        corner_diagonal_point2 = corner_diagonal_point2.reshape(1, -1)
        for i in range(n_diagonal_1_square_1):
            diag_point = rng.uniform(0, 1)
            diagonal_1_square_1[i] = np.array([diag_point, diag_point])
        diagonal_1_square_1 = np.concatenate(
            [corner_diagonal_point1, diagonal_1_square_1]
        )
        diagonal_1_square_1 = np.concatenate(
            [diagonal_1_square_1, corner_diagonal_point2]
        )

        # Create points along the diagonal_2 of square 1
        diagonal_2_square_1 = np.zeros((n_diagonal_2_square_1, 2))
        corner_diag_point1 = rng.uniform(0, 0.05)
        corner_diag_point2 = rng.uniform(0.95, 1)
        corner_diagonal_point1 = np.array([corner_diag_point1, 1 - corner_diag_point1])
        corner_diagonal_point1 = corner_diagonal_point1.reshape(1, -1)
        corner_diagonal_point2 = np.array([corner_diag_point2, 1 - corner_diag_point2])
        corner_diagonal_point2 = corner_diagonal_point2.reshape(1, -1)
        for i in range(n_diagonal_2_square_1):
            diag_point = rng.uniform(0, 1)
            diagonal_2_square_1[i] = np.array([diag_point, 1 - diag_point])
        diagonal_2_square_1 = np.concatenate(
            [corner_diagonal_point1, diagonal_2_square_1]
        )
        diagonal_2_square_1 = np.concatenate(
            [diagonal_2_square_1, corner_diagonal_point2]
        )

        # Create points along the diagonal_1 of square 2
        diagonal_1_square_2 = np.zeros((n_diagonal_1_square_2, 2))
        corner_diag_point1 = rng.uniform(0, 0.05)
        corner_diag_point2 = rng.uniform(0.95, 1)
        corner_diagonal_point1 = np.array([corner_diag_point1, corner_diag_point1])
        corner_diagonal_point1 = corner_diagonal_point1.reshape(1, -1)
        corner_diagonal_point2 = np.array([corner_diag_point2, corner_diag_point2])
        corner_diagonal_point2 = corner_diagonal_point2.reshape(1, -1)
        for i in range(n_diagonal_1_square_2):
            diag_point = rng.uniform(0, 1)
            diagonal_1_square_2[i] = np.array([diag_point, diag_point])
        diagonal_1_square_2 = np.concatenate(
            [corner_diagonal_point1, diagonal_1_square_2]
        )
        diagonal_1_square_2 = np.concatenate(
            [diagonal_1_square_2, corner_diagonal_point2]
        )

        # Create points along the diagonal_2 of square 2
        diagonal_2_square_2 = np.zeros((n_diagonal_2_square_2, 2))
        corner_diag_point1 = rng.uniform(0, 0.05)
        corner_diag_point2 = rng.uniform(0.95, 1)
        corner_diagonal_point1 = np.array([corner_diag_point1, 1 - corner_diag_point1])
        corner_diagonal_point1 = corner_diagonal_point1.reshape(1, -1)
        corner_diagonal_point2 = np.array([corner_diag_point2, 1 - corner_diag_point2])
        corner_diagonal_point2 = corner_diagonal_point2.reshape(1, -1)
        for i in range(n_diagonal_2_square_2):
            diag_point = rng.uniform(0, 1)
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


def construct_simplicial_complex(
    quotiented_adj_matrix: np.ndarray | None,
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

    return sc


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

        # if np.any(degree_counts != 2):
        #     print(f"Vertex {vertex} degree counts:", degree_counts)
        #     print("Incident edges:", incident_edges)

        if n_components != 1:
            # print(f"More than one component in the link graph for vertex {vertex}")
            return n_components, False
        elif np.any(degree_counts != 2):
            # print(f"Link check failed for vertex {vertex}")
            return n_components, False

    # Step 2: Check if the entire complex (1-skeleton) is connected

    # Compute the adjacency matrix for the 1-skeleton (vertex-vertex adjacency)
    adjacency_matrix = create_1skel_adjacency(vertex_edge_matrix)

    # Use connected_components to check if the 1-skeleton is connected
    n_components, _ = connected_components(adjacency_matrix, directed=True)

    # The complex is connected if there's only one connected component
    return n_components, n_components == 1


def generate_genus_1_datapoints(
    n_lower: int = 5, n_upper: int = 25, no_of_points: int = 25
) -> np.ndarray:

    genus_1_datapoints = np.ndarray((no_of_points, 2), dtype=object)
    cnt = 0
    i = 0
    while cnt < no_of_points:
        n_cycle_1 = rng.integers(n_lower, n_upper)
        n_cycle_2 = rng.integers(n_lower, n_upper)
        n_interior = rng.integers(n_lower, n_upper)

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
        n_components, is_surface = check_surface_homeomorphic(D1, D2)
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

    sc_sphere = construct_simplicial_complex_genus_0(
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

    D1 = sc_sphere.incidence_matrix(1).todense()
    D2 = sc_sphere.incidence_matrix(2).todense()
    ker_D1 = np.shape(D1)[1] - matrix_rank(D1)
    ker_D2 = np.shape(D2)[1] - matrix_rank(D2)

    H1 = ker_D1 - matrix_rank(D2)
    H2 = ker_D2
    H0, is_surface = check_surface_homeomorphic(D1, D2)

    if is_surface and H0 == 1 and H1 == 0 and H2 == 1:
        print("Success!")
        return D1, D2
    else:
        raise ValueError(
            f"The generated complex is not a sphere - is_surface: {is_surface}, b0: {H0}, b1: {H1}, b2: {H2}"
        )
        return None, None


def save_datapoints(datapoints, folder_path, file_name):
    file_path = os.path.join(folder_path, file_name)
    np.save(file_path, datapoints)


def generate_genus_0_dataset(
    n_lower: int = 5, n_upper: int = 25, no_of_points: int = 25
) -> None:

    genus_0_datapoints = np.full((no_of_points, 2), None, dtype=object)
    cnt = 0
    seen = set()

    while cnt < no_of_points:

        try:
            D1, D2 = generate_genus_0_datapoints(
                n_cycle_1=rng.integers(n_lower, n_upper),
                n_cycle_2=rng.integers(n_lower, n_upper),
                n_cycle_3=rng.integers(n_lower, n_upper),
                n_cycle_4=rng.integers(n_lower, n_upper),
                n_interior=rng.integers(n_lower, n_upper),
                n_interior_square_2=rng.integers(n_lower, n_upper),
                n_diagonal_1_square_1=rng.integers(n_lower, n_upper),
                n_diagonal_2_square_1=rng.integers(n_lower, n_upper),
                n_diagonal_1_square_2=rng.integers(n_lower, n_upper),
                n_diagonal_2_square_2=rng.integers(n_lower, n_upper),
            )

            key = _pair_key(D1, D2)
            if key in seen:
                continue

            seen.add(key)
            genus_0_datapoints[cnt, 0] = D1
            genus_0_datapoints[cnt, 1] = D2
            cnt += 1

        except Exception as e:
            print(f"Error in generate_genus_0_datapoint: {e}")
            continue

    return genus_0_datapoints


def generate_genus_1_datapoint(
    n_lower: int = 5, n_upper: int = 25
) -> Tuple[np.ndarray, np.ndarray]:
    sc_torus = generate_genus_n_simplicial_complex(
        genus=1, n_lower=n_lower, n_upper=n_upper
    )
    D1 = sc_torus.incidence_matrix(1).todense()
    D2 = sc_torus.incidence_matrix(2).todense()

    ker_D1 = np.shape(D1)[1] - matrix_rank(D1)
    ker_D2 = np.shape(D2)[1] - matrix_rank(D2)
    H1 = ker_D1 - matrix_rank(D2)
    H2 = ker_D2
    H0, is_surface = check_surface_homeomorphic(D1, D2)

    if is_surface and H0 == 1 and H1 == 2 and H2 == 1:
        print("Success!")
        return D1, D2
    else:
        raise ValueError(
            f"The generated complex is not a torus - is_surface: {is_surface}, b0: {H0}, b1: {H1}, b2: {H2}"
        )
        return None, None


def generate_genus_1_dataset(
    n_lower: int = 5, n_upper: int = 25, no_of_points: int = 25
) -> None:

    genus_1_datapoints = np.full((no_of_points, 2), None, dtype=object)
    cnt = 0
    seen = set()

    while cnt < no_of_points:
        try:
            D1, D2 = generate_genus_1_datapoint(n_lower=n_lower, n_upper=n_upper)
            key = _pair_key(D1, D2)
            if key in seen:
                continue

            seen.add(key)
            genus_1_datapoints[cnt, 0] = D1
            genus_1_datapoints[cnt, 1] = D2
            cnt += 1

        except Exception as e:
            print(f"Error in generate_genus_1_datapoint: {e}")
            continue

    return genus_1_datapoints


def generate_genus_2_datapoint(
    n_lower: int = 5, n_upper: int = 25, no_of_points: int = 25
) -> None:

    tri_1_n_cycle_1 = rng.integers(n_lower, n_upper)
    tri_1_n_cycle_2 = rng.integers(n_lower, n_upper)
    tri_1_n_interior = rng.integers(n_lower, n_upper)

    tri_2_n_cycle_1 = rng.integers(n_lower, n_upper)
    tri_2_n_cycle_2 = rng.integers(n_lower, n_upper)
    tri_2_n_interior = rng.integers(n_lower, n_upper)

    points_1, tri_1, adj_matrix_1, quotiented_adj_matrix_1 = (
        generate_genus_1_triangulations(
            n_cycle_1=tri_1_n_cycle_1,
            n_cycle_2=tri_1_n_cycle_2,
            n_interior=tri_1_n_interior,
        )
    )

    points_2, tri_2, adj_matrix_2, quotiented_adj_matrix_2 = (
        generate_genus_1_triangulations(
            n_cycle_1=tri_2_n_cycle_1,
            n_cycle_2=tri_2_n_cycle_2,
            n_interior=tri_2_n_interior,
        )
    )

    sc1 = construct_simplicial_complex(
        quotiented_adj_matrix_1.copy(),
        tri_1.simplices.copy(),
        tri_1_n_cycle_1,
        tri_1_n_cycle_2,
        tri_1_n_interior,
    )
    sc2 = construct_simplicial_complex(
        quotiented_adj_matrix_2.copy(),
        tri_2.simplices.copy(),
        tri_2_n_cycle_1,
        tri_2_n_cycle_2,
        tri_2_n_interior,
    )

    tri_1.simplices = sc1.get_all_maximal_simplices()
    tri_2.simplices = sc2.get_all_maximal_simplices()

    vertices_to_add = [np.max(triangle) for triangle in tri_1.simplices]
    vertices_to_add = max(vertices_to_add)

    two_simplex = []

    tri_1.simplices = np.array([triangle for triangle in tri_1.simplices])
    tri_2.simplices = np.array([triangle for triangle in tri_2.simplices])

    random_quotient_triangle_1 = random.choice(tri_1.simplices)

    random_quotient_triangle_2 = random.choice(tri_2.simplices)

    # print("Random quotient triangle 1:", random_quotient_triangle_1)
    # print("Random quotient triangle 2:", random_quotient_triangle_2)

    new_tri_2_simplices = []
    for triangle in tri_2.simplices:
        if not np.array_equal(triangle, random_quotient_triangle_2):
            new_tri_2_simplices.append(triangle)
    tri_2.simplices = np.array(new_tri_2_simplices)

    new_tri_2_list = []
    for triangle in tri_2.simplices:
        for i in enumerate(triangle):
            if i[1] == random_quotient_triangle_1[0]:
                triangle[i[0]] += vertices_to_add * 4
            elif i[1] == random_quotient_triangle_1[1]:
                triangle[i[0]] += vertices_to_add * 4
            elif i[1] == random_quotient_triangle_1[2]:
                triangle[i[0]] += vertices_to_add * 4
        new_tri_2_list.append(triangle)
    tri_2.simplices = np.array(new_tri_2_list).tolist()

    # print("Second triangulation simplices after shift of three vertices:", tri_2.simplices)

    new_tri_2_list = []
    for triangle in tri_2.simplices:
        for i in enumerate(triangle):
            if i[1] == random_quotient_triangle_2[0]:
                triangle[i[0]] = random_quotient_triangle_1[0]
            elif i[1] == random_quotient_triangle_2[1]:
                triangle[i[0]] = random_quotient_triangle_1[1]
            elif i[1] == random_quotient_triangle_2[2]:
                triangle[i[0]] = random_quotient_triangle_1[2]
        new_tri_2_list.append(triangle)
    tri_2.simplices = np.array(new_tri_2_list).tolist()
    # print("Second triangulation simplices after quotienting:", tri_2.simplices)

    new_tri_1_simplices = []
    for triangle in tri_1.simplices:
        if not np.array_equal(triangle, random_quotient_triangle_1):
            new_tri_1_simplices.append(triangle)
    tri_1.simplices = np.array(new_tri_1_simplices)

    print("Vertices to add:", vertices_to_add)
    for triangle in tri_2.simplices:
        for i in enumerate(triangle):
            if (
                i[1] != random_quotient_triangle_1[0]
                and i[1] != random_quotient_triangle_1[1]
                and i[1] != random_quotient_triangle_1[2]
            ):
                triangle[i[0]] += vertices_to_add * 2
            else:
                continue

    tri_2.simplices = np.array(tri_2.simplices)

    # print("First triangulation simplices after everything:", tri_1.simplices)

    final_tri_list = list(tri_1.simplices) + list(tri_2.simplices)
    final_tri_list = list(set(tuple(triangle) for triangle in final_tri_list))

    two_simplex = np.array(final_tri_list).tolist()
    # with np.printoptions(threshold=np.inf):
    #   print("Final triangulation simplices:", two_simplex)

    sc = tnx.SimplicialComplex(two_simplex)

    D1 = sc.incidence_matrix(1).todense()
    D2 = sc.incidence_matrix(2).todense()
    ker_D1 = np.shape(D1)[1] - matrix_rank(D1)
    H1 = ker_D1 - matrix_rank(D2)

    print("The 1st homology is", H1)
    n_components, is_surface = check_surface_homeomorphic(D1, D2)
    print("The complex represents a surface:", is_surface)


def generate_disconnected_datapoint(
    first_genus: int, second_genus: int, n_lower: int, n_upper: int
) -> Tuple[np.ndarray, np.ndarray]:

    if first_genus == 0:
        cycle_1 = rng.integers(n_lower, n_upper)
        cycle_2 = rng.integers(n_lower, n_upper)
        cycle_3 = rng.integers(n_lower, n_upper)
        cycle_4 = rng.integers(n_lower, n_upper)
        n_interior = rng.integers(n_lower, n_upper)
        n_interior_square_2 = rng.integers(n_lower, n_upper)
        n_diagonal_1_square_1 = rng.integers(n_lower, n_upper)
        n_diagonal_2_square_1 = rng.integers(n_lower, n_upper)
        n_diagonal_1_square_2 = rng.integers(n_lower, n_upper)
        n_diagonal_2_square_2 = rng.integers(n_lower, n_upper)

        sc1 = construct_simplicial_complex_genus_0(
            cycle_1,
            cycle_2,
            cycle_3,
            cycle_4,
            n_interior,
            n_interior_square_2,
            n_diagonal_1_square_1,
            n_diagonal_2_square_1,
            n_diagonal_1_square_2,
            n_diagonal_2_square_2,
        )

    elif first_genus >= 1:
        sc1 = generate_genus_n_simplicial_complex(first_genus, n_lower, n_upper)

    if second_genus == 0:
        cycle_1_2 = rng.integers(n_lower, n_upper)
        cycle_2_2 = rng.integers(n_lower, n_upper)
        cycle_3_2 = rng.integers(n_lower, n_upper)
        cycle_4_2 = rng.integers(n_lower, n_upper)
        n_interior_2 = rng.integers(n_lower, n_upper)
        n_interior_square_2_2 = rng.integers(n_lower, n_upper)
        n_diagonal_1_square_1_2 = rng.integers(n_lower, n_upper)
        n_diagonal_2_square_1_2 = rng.integers(n_lower, n_upper)
        n_diagonal_1_square_2_2 = rng.integers(n_lower, n_upper)
        n_diagonal_2_square_2_2 = rng.integers(n_lower, n_upper)

        sc2 = construct_simplicial_complex_genus_0(
            cycle_1_2,
            cycle_2_2,
            cycle_3_2,
            cycle_4_2,
            n_interior_2,
            n_interior_square_2_2,
            n_diagonal_1_square_1_2,
            n_diagonal_2_square_1_2,
            n_diagonal_1_square_2_2,
            n_diagonal_2_square_2_2,
        )

    elif second_genus >= 1:
        sc2 = generate_genus_n_simplicial_complex(second_genus, n_lower, n_upper)

    simplices_1 = sc1.get_all_maximal_simplices()
    simplices_2 = sc2.get_all_maximal_simplices()

    simplices_1 = np.array([triangle for triangle in simplices_1])
    simplices_2 = np.array([triangle for triangle in simplices_2])

    final_tri_list = list(simplices_1) + list(simplices_2)

    vertices_to_add = [np.max(triangle) for triangle in final_tri_list]
    vertices_to_add = max(vertices_to_add)

    new_tri_2_list = []
    for triangle in simplices_2:
        for i in enumerate(triangle):
            triangle[i[0]] += vertices_to_add * 2
        new_tri_2_list.append(triangle)
    simplices_2 = np.array(new_tri_2_list).tolist()

    final_tri_list = list(simplices_1) + list(simplices_2)
    final_tri_list = list(set(tuple(triangle) for triangle in final_tri_list))

    two_simplex = np.array(final_tri_list).tolist()
    # with np.printoptions(threshold=np.inf):
    # print("Final triangulation simplices:", two_simplex)

    sc = tnx.SimplicialComplex(two_simplex)

    D1 = sc.incidence_matrix(1).todense()
    D2 = sc.incidence_matrix(2).todense()
    ker_D1 = np.shape(D1)[1] - matrix_rank(D1)
    ker_D2 = np.shape(D2)[1] - matrix_rank(D2)
    H1 = ker_D1 - matrix_rank(D2)
    H2 = ker_D2
    H0, is_surface = check_surface_homeomorphic(D1, D2)
    # print("The complex represents a surface:", is_surface)

    if (
        H0 == 2
        and H1 == 2 * (first_genus + second_genus)
        and H2 == 2
        and not is_surface
    ):
        print("Successfully created disconnected datapoint with b1=", H1)
        return D1, D2
    else:
        print(f"Failed: H0 is {H0}, H1 is {H1}, H2 is {H2}")
        return None, None


def generate_disconnected_datapoint_with_klein_bottle(
    second_genus: int, n_lower: int, n_upper: int, first_genus=1
) -> Tuple[np.ndarray, np.ndarray]:

    # Make klein bottle for first simplicial complex
    n_klein_cycle_1 = rng.integers(n_lower, n_upper)
    n_klein_cycle_2 = rng.integers(n_lower, n_upper)
    n_klein_interior = rng.integers(n_lower, n_upper)
    sc1, _, _ = make_klein_bottles(n_klein_cycle_1, n_klein_cycle_2, n_klein_interior)

    if second_genus == 0:
        cycle_1_2 = rng.integers(n_lower, n_upper)
        cycle_2_2 = rng.integers(n_lower, n_upper)
        cycle_3_2 = rng.integers(n_lower, n_upper)
        cycle_4_2 = rng.integers(n_lower, n_upper)
        n_interior_2 = rng.integers(n_lower, n_upper)
        n_interior_square_2_2 = rng.integers(n_lower, n_upper)
        n_diagonal_1_square_1_2 = rng.integers(n_lower, n_upper)
        n_diagonal_2_square_1_2 = rng.integers(n_lower, n_upper)
        n_diagonal_1_square_2_2 = rng.integers(n_lower, n_upper)
        n_diagonal_2_square_2_2 = rng.integers(n_lower, n_upper)

        sc2 = construct_simplicial_complex_genus_0(
            cycle_1_2,
            cycle_2_2,
            cycle_3_2,
            cycle_4_2,
            n_interior_2,
            n_interior_square_2_2,
            n_diagonal_1_square_1_2,
            n_diagonal_2_square_1_2,
            n_diagonal_1_square_2_2,
            n_diagonal_2_square_2_2,
        )

    elif second_genus >= 1:
        sc2 = generate_genus_n_simplicial_complex(second_genus, n_lower, n_upper)

    simplices_1 = sc1.get_all_maximal_simplices()
    simplices_2 = sc2.get_all_maximal_simplices()

    simplices_1 = np.array([triangle for triangle in simplices_1])
    simplices_2 = np.array([triangle for triangle in simplices_2])

    final_tri_list = list(simplices_1) + list(simplices_2)

    vertices_to_add = [np.max(triangle) for triangle in final_tri_list]
    vertices_to_add = max(vertices_to_add)

    new_tri_2_list = []
    for triangle in simplices_2:
        for i in enumerate(triangle):
            triangle[i[0]] += vertices_to_add * 2
        new_tri_2_list.append(triangle)
    simplices_2 = np.array(new_tri_2_list).tolist()

    final_tri_list = list(simplices_1) + list(simplices_2)
    final_tri_list = list(set(tuple(triangle) for triangle in final_tri_list))

    two_simplex = np.array(final_tri_list).tolist()
    # with np.printoptions(threshold=np.inf):
    # print("Final triangulation simplices:", two_simplex)

    sc = tnx.SimplicialComplex(two_simplex)

    D1 = sc.incidence_matrix(1).todense()
    D2 = sc.incidence_matrix(2).todense()
    ker_D1 = np.shape(D1)[1] - matrix_rank(D1)
    ker_D2 = np.shape(D2)[1] - matrix_rank(D2)
    H0 = matrix_rank(D1) - np.shape(D1)[1]
    H1 = ker_D1 - matrix_rank(D2)
    H2 = ker_D2

    # print("The 0th homology is", H0)
    # print("The 1st homology is", H1)
    # print("The 2nd homology is", H2)
    H0, is_surface = check_surface_homeomorphic(D1, D2)
    # print("The complex represents a surface:", is_surface)

    if H0 == 2 and H1 == 1 + 2 * second_genus and H2 == 1 and not is_surface:
        print("Successfully created disconnected datapoint with klein bottle")
        return D1, D2
    else:
        print(f"Failed: H0 is {H0}, H1 is {H1}, H2 is {H2}")
        return None, None


def generate_disconnected_klein_plus_klein(
    n_lower: int, n_upper: int, first_genus=1
) -> Tuple[np.ndarray, np.ndarray]:

    # Make klein bottle for first simplicial complex
    n_klein_1_cycle_1 = rng.integers(n_lower, n_upper)
    n_klein_1_cycle_2 = rng.integers(n_lower, n_upper)
    n_klein_1_interior = rng.integers(n_lower, n_upper)
    sc1, _, _ = make_klein_bottles(
        n_klein_1_cycle_1, n_klein_1_cycle_2, n_klein_1_interior
    )

    # Make klein bottle for second simplicial complex
    n_klein_2_cycle_1 = rng.integers(n_lower, n_upper)
    n_klein_2_cycle_2 = rng.integers(n_lower, n_upper)
    n_klein_2_interior = rng.integers(n_lower, n_upper)
    sc2, _, _ = make_klein_bottles(
        n_klein_2_cycle_1, n_klein_2_cycle_2, n_klein_2_interior
    )

    simplices_1 = sc1.get_all_maximal_simplices()
    simplices_2 = sc2.get_all_maximal_simplices()

    simplices_1 = np.array([triangle for triangle in simplices_1])
    simplices_2 = np.array([triangle for triangle in simplices_2])

    final_tri_list = list(simplices_1) + list(simplices_2)

    vertices_to_add = [np.max(triangle) for triangle in final_tri_list]
    vertices_to_add = max(vertices_to_add)

    new_tri_2_list = []
    for triangle in simplices_2:
        for i in enumerate(triangle):
            triangle[i[0]] += vertices_to_add * 2
        new_tri_2_list.append(triangle)
    simplices_2 = np.array(new_tri_2_list).tolist()

    final_tri_list = list(simplices_1) + list(simplices_2)
    final_tri_list = list(set(tuple(triangle) for triangle in final_tri_list))

    two_simplex = np.array(final_tri_list).tolist()
    # with np.printoptions(threshold=np.inf):
    # print("Final triangulation simplices:", two_simplex)

    sc = tnx.SimplicialComplex(two_simplex)

    D1 = sc.incidence_matrix(1).todense()
    D2 = sc.incidence_matrix(2).todense()
    ker_D1 = np.shape(D1)[1] - matrix_rank(D1)
    ker_D2 = np.shape(D2)[1] - matrix_rank(D2)
    H0 = matrix_rank(D1) - np.shape(D1)[1]
    H1 = ker_D1 - matrix_rank(D2)
    H2 = ker_D2

    # print("The 0th homology is", H0)
    # print("The 1st homology is", H1)
    # print("The 2nd homology is", H2)
    H0, is_surface = check_surface_homeomorphic(D1, D2)
    # print("The complex represents a surface:", is_surface)

    if H0 == 2 and H1 == 2 and H2 == 0 and not is_surface:
        print("Successfully created klein + klein")
        return D1, D2
    else:
        print(f"Failed: H0 is {H0}, H1 is {H1}, H2 is {H2}")
        return None, None


def generate_genus_n_simplicial_complex(genus: int, n_lower: int, n_upper: int):

    tri_point_list = []
    for i in range(genus):
        tri_point_list.append(
            [
                rng.integers(n_lower, n_upper),
                rng.integers(n_lower, n_upper),
                rng.integers(n_lower, n_upper),
            ]
        )

    triangle_list = []
    for i in range(genus):
        triangle_list.append(
            generate_genus_1_triangulations(
                n_cycle_1=tri_point_list[i][0],
                n_cycle_2=tri_point_list[i][1],
                n_interior=tri_point_list[i][2],
            )
        )

    sc_list = []
    for i in range(genus):
        sc_list.append(
            construct_simplicial_complex(
                triangle_list[i][3].copy(),
                triangle_list[i][1].simplices.copy(),
                tri_point_list[i][0],
                tri_point_list[i][1],
                tri_point_list[i][2],
            )
        )

    maximal_simplices_list = []
    for i in range(genus):
        maximal_simplices_list.append(sc_list[i].get_all_maximal_simplices())

    array_simplex_list = []
    for i in range(genus):
        array_simplex_list.append(
            np.array([triangle for triangle in maximal_simplices_list[i]])
        )

    random_quotient_triangle_list = []
    for i in range(genus):
        random_quotient_triangle_list.append(random.choice(array_simplex_list[i]))

    running_quotient = np.array(array_simplex_list[0])
    sc = tnx.SimplicialComplex(running_quotient.tolist())

    for i in range(1, genus):

        vertices_to_add = [np.max(triangle) for triangle in running_quotient]
        vertices_to_add = max(vertices_to_add)

        new_tri_simplices = []
        for triangle in array_simplex_list[i]:
            if not np.array_equal(triangle, random_quotient_triangle_list[i]):
                new_tri_simplices.append(triangle)

        new_tri_simplices = np.array(new_tri_simplices)

        running_random = random.choice(running_quotient)

        new_tri_i_list = []
        for triangle in new_tri_simplices:
            for j in enumerate(triangle):
                if j[1] == running_random[0]:
                    triangle[j[0]] += vertices_to_add * 10
                elif j[1] == running_random[1]:
                    triangle[j[0]] += vertices_to_add * 10
                elif j[1] == running_random[2]:
                    triangle[j[0]] += vertices_to_add * 10
            new_tri_i_list.append(triangle)
        tri_i_simplices = np.array(new_tri_i_list).tolist()

        new_tri_i_list = []
        for triangle in tri_i_simplices:
            for k in enumerate(triangle):
                if k[1] == random_quotient_triangle_list[i][0]:
                    triangle[k[0]] = running_random[0]
                elif k[1] == random_quotient_triangle_list[i][1]:
                    triangle[k[0]] = running_random[1]
                elif k[1] == random_quotient_triangle_list[i][2]:
                    triangle[k[0]] = running_random[2]
            new_tri_i_list.append(triangle)
        tri_i_simplices = np.array(new_tri_i_list).tolist()

        new_running_quotient = []
        for triangle in running_quotient:
            if not np.array_equal(triangle, running_random):
                new_running_quotient.append(triangle)
        running_quotient = np.array(new_running_quotient)

        for triangle in tri_i_simplices:
            for l in enumerate(triangle):
                if (
                    l[1] != running_random[0]
                    and l[1] != running_random[1]
                    and l[1] != running_random[2]
                ):
                    triangle[l[0]] += vertices_to_add * 2
                else:
                    continue

        tri_i_simplices = np.array(tri_i_simplices)
        running_quotient_1 = list(tri_i_simplices) + list(running_quotient)
        running_quotient_2 = list(
            set(tuple(triangle) for triangle in running_quotient_1)
        )
        sc = tnx.SimplicialComplex(running_quotient_2)
        running_quotient_3 = sc.get_all_maximal_simplices()
        running_quotient = np.array([triangle for triangle in running_quotient_3])

    sc = tnx.SimplicialComplex(running_quotient.tolist())

    return sc
    # D1 = sc.incidence_matrix(1).todense()
    # D2 = sc.incidence_matrix(2).todense()
    # ker_D1 = np.shape(D1)[1] - matrix_rank(D1)
    # H1 = ker_D1 - matrix_rank(D2)
    # print("The 1st homology is", H1)
    # is_surface = check_surface_homeomorphic(D1, D2)
    # print("The complex represents a surface:", is_surface)


def generate_disconnected_dataset(
    no_of_points: int, first_genus: int, second_genus: int, n_lower: int, n_upper: int
) -> None:
    disconnected_datapoints = np.full((no_of_points, 2), None, dtype=object)
    cnt = 0
    seen = set()

    while cnt < no_of_points:

        try:
            D1, D2 = generate_disconnected_datapoint(
                first_genus, second_genus, n_lower, n_upper
            )

            key = _pair_key(D1, D2)
            if key in seen:
                continue

            seen.add(key)
            disconnected_datapoints[cnt, 0] = D1
            disconnected_datapoints[cnt, 1] = D2
            cnt += 1

        except Exception as e:
            print(f"Error in generate_disconnected_datapoint: {e}")
            traceback.print_exc()
            continue

    return disconnected_datapoints


def generate_disconnected_dataset_with_klein_bottle(
    no_of_points: int, second_genus: int, n_lower: int, n_upper: int
) -> None:
    disconnected_datapoints = np.full((no_of_points, 2), None, dtype=object)
    cnt = 0
    seen = set()

    while cnt < no_of_points:

        try:
            D1, D2 = generate_disconnected_datapoint_with_klein_bottle(
                second_genus, n_lower, n_upper
            )

            key = _pair_key(D1, D2)
            if key in seen:
                continue

            seen.add(key)
            disconnected_datapoints[cnt, 0] = D1
            disconnected_datapoints[cnt, 1] = D2
            cnt += 1

        except Exception as e:
            print(f"Error in generate_disconnected_datapoint_with_klein_bottle: {e}")
            traceback.print_exc()
            continue

    return disconnected_datapoints


def generate_disconnected_dataset_klein_plus_klein(
    no_of_points: int, n_lower: int, n_upper: int
) -> None:
    disconnected_datapoints = np.full((no_of_points, 2), None, dtype=object)
    cnt = 0
    seen = set()

    while cnt < no_of_points:

        try:
            D1, D2 = generate_disconnected_klein_plus_klein(n_lower, n_upper)

            key = _pair_key(D1, D2)
            if key in seen:
                continue

            seen.add(key)
            disconnected_datapoints[cnt, 0] = D1
            disconnected_datapoints[cnt, 1] = D2
            cnt += 1

        except Exception as e:
            print(f"Error in generate_disconnected_klein_plus_klein: {e}")
            traceback.print_exc()
            continue

    return disconnected_datapoints


def apply_vertex_map(triangles, vmap):
    new_tris = []
    for tri in triangles:
        t = tuple(vmap.get(v, v) for v in tri)
        new_tris.append(tuple(sorted(t)))
    return list(set(new_tris))


def make_klein_bottles(n_cycle_1: int = 5, n_cycle_2: int = 5, n_interior: int = 10):
    # To be implemented: Function to generate datapoints for Klein bottles
    # Generate triangulations similar to the torus but with different identifications
    pts, triangulations, _, _ = generate_genus_1_triangulations(
        n_cycle_1=n_cycle_1, n_cycle_2=n_cycle_2, n_interior=n_interior
    )
    # print(f"Original triangulations simplices:\n{triangulations.simplices}")

    # -----------------------------
    # Stage 1: left–right (twisted)
    # -----------------------------
    vmap_lr = {}

    L = list(range(4, 4 + n_cycle_1))
    R = list(range(4 + n_cycle_1, 4 + 2 * n_cycle_1))

    for l, r in zip(L, reversed(R)):
        vmap_lr[r] = l

    # twisted corner identification (from your figure)
    vmap_lr[2] = 0
    vmap_lr[1] = 3

    tri_stage1 = apply_vertex_map(triangulations.simplices, vmap_lr)

    # -----------------------------
    # Stage 2: bottom–top (untwisted)
    # -----------------------------
    vmap_tb = {}

    B = list(range(4 + 2 * n_cycle_1, 4 + 2 * n_cycle_1 + n_cycle_2))
    T = list(range(4 + 2 * n_cycle_1 + n_cycle_2, 4 + 2 * n_cycle_1 + 2 * n_cycle_2))

    for b, t in zip(B, T):
        vmap_tb[t] = b

    # remaining corners
    vmap_tb[3] = 0

    tri_final = apply_vertex_map(tri_stage1, vmap_tb)

    sc = tnx.SimplicialComplex(np.array(tri_final))

    D1 = sc.incidence_matrix(1).todense()
    D2 = sc.incidence_matrix(2).todense()

    ker_D1 = np.shape(D1)[1] - matrix_rank(D1)
    H0 = np.shape(D1)[0] - matrix_rank(D1)
    H1 = ker_D1 - matrix_rank(D2)
    H2 = np.shape(D2)[1] - matrix_rank(D2)
    H0, is_surface = check_surface_homeomorphic(D1, D2)

    if is_surface and H0 == 1 and H1 == 1 and H2 == 0:
        # print("Successfully created Klein bottle datapoint")
        return sc, D1, D2
    else:
        raise ValueError("Generated complex is not a Klein bottle...continuing")
        return None, None, None


def generate_klein_bottle_datapoints(
    n_lower: int = 5, n_upper: int = 25, no_of_points: int = 25
) -> np.ndarray:

    klein_bottle_datapoints = np.ndarray((no_of_points, 2), dtype=object)
    cnt = 0
    i = 0

    while cnt < no_of_points:
        n_cycle_1 = rng.integers(n_lower, n_upper)
        n_cycle_2 = rng.integers(n_lower, n_upper)
        n_interior = rng.integers(n_lower, n_upper)
        try:
            sc, D1, D2 = make_klein_bottles(n_cycle_1, n_cycle_2, n_interior)
            if not any(
                np.array_equal(arr, [D1, D2]) for arr in klein_bottle_datapoints
            ):
                klein_bottle_datapoints[i][0] = D1
                klein_bottle_datapoints[i][1] = D2
            cnt += 1
        except Exception:
            continue
        if cnt == no_of_points:
            break
        elif i == no_of_points - 1:
            i = 0
            continue

        i += 1

    klein_bottle_datapoints = np.array(klein_bottle_datapoints, dtype=object)

    return klein_bottle_datapoints


def generate_arbitrary_disjoint_union_datapoint(
    n_lower: int, n_upper: int, n_klein: int = 1, n_sphere: int = 1, n_torus: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a single datapoint for an arbitrary disjoint union of spheres, tori and klein bottles"""

    # Only implemented for one each of klein, sphere and torus for now, but can be easily extended to more
    sc_klein, _, _ = make_klein_bottles(
        rng.integers(n_lower, n_upper),
        rng.integers(n_lower, n_upper),
        rng.integers(n_lower, n_upper),
    )
    sc_torus = generate_genus_n_simplicial_complex(1, n_lower, n_upper)
    sc_sphere = construct_simplicial_complex_genus_0(
        n_cycle_1=rng.integers(n_lower, n_upper),
        n_cycle_2=rng.integers(n_lower, n_upper),
        n_cycle_3=rng.integers(n_lower, n_upper),
        n_cycle_4=rng.integers(n_lower, n_upper),
        n_interior=rng.integers(n_lower, n_upper),
        n_interior_square_2=rng.integers(n_lower, n_upper),
        n_diagonal_1_square_1=rng.integers(n_lower, n_upper),
        n_diagonal_2_square_1=rng.integers(n_lower, n_upper),
        n_diagonal_1_square_2=rng.integers(n_lower, n_upper),
        n_diagonal_2_square_2=rng.integers(n_lower, n_upper),
    )

    # Disjoint union of klein and torus first. Then add sphere to the disjoint union of those two.
    simplices_klein = sc_klein.get_all_maximal_simplices()
    simplices_torus = sc_torus.get_all_maximal_simplices()
    simplices_sphere = sc_sphere.get_all_maximal_simplices()

    simplices_klein = np.array([triangle for triangle in simplices_klein])
    simplices_torus = np.array([triangle for triangle in simplices_torus])
    simplices_sphere = np.array([triangle for triangle in simplices_sphere])

    intermediate_tri_list = list(simplices_klein) + list(simplices_torus)
    vertices_to_add = [np.max(triangle) for triangle in intermediate_tri_list]
    vertices_to_add = max(vertices_to_add)

    new_tri_torus_list = []
    for triangle in simplices_torus:
        for i in enumerate(triangle):
            triangle[i[0]] += vertices_to_add * 2
        new_tri_torus_list.append(triangle)
    simplices_torus = np.array(new_tri_torus_list).tolist()

    intermediate_tri_list = list(simplices_klein) + list(simplices_torus)
    simplices_sphere = np.array([triangle for triangle in simplices_sphere])
    vertices_to_add = [np.max(triangle) for triangle in intermediate_tri_list]
    vertices_to_add = max(vertices_to_add)

    new_tri_sphere_list = []
    for triangle in simplices_sphere:
        for i in enumerate(triangle):
            triangle[i[0]] += vertices_to_add * 3
        new_tri_sphere_list.append(triangle)
    simplices_sphere = np.array(new_tri_sphere_list).tolist()

    final_tri_list = intermediate_tri_list + list(simplices_sphere)
    final_tri_list = list(set(tuple(triangle) for triangle in final_tri_list))

    two_simplex = np.array(final_tri_list).tolist()
    sc = tnx.SimplicialComplex(two_simplex)

    D1 = sc.incidence_matrix(1).todense()
    D2 = sc.incidence_matrix(2).todense()
    ker_D1 = np.shape(D1)[1] - matrix_rank(D1)
    ker_D2 = np.shape(D2)[1] - matrix_rank(D2)
    H1 = ker_D1 - matrix_rank(D2)
    H2 = ker_D2
    H0, is_surface = check_surface_homeomorphic(D1, D2)

    if H0 == 3 and H1 == 3 and H2 == 2 and not is_surface:
        print("Successfully created klein + torus + sphere")
        return D1, D2
    else:
        raise ValueError(
            f"Failed: is_surface is {is_surface}, H0 is {H0}, H1 is {H1}, H2 is {H2}"
        )
        return None, None


def generate_arbitrary_disjoint_union_dataset(
    no_of_points: int,
    n_lower: int,
    n_upper: int,
    n_klein: int = 1,
    n_sphere: int = 1,
    n_torus: int = 1,
) -> None:
    """
    Generate datapoints corresponding to disjoint union of an arbitrary number of spheres, tori and klein bottles

    Parameters
    ----------
    no_of_points : int
        Total number of datapoints to generate
    n_klein : int
        Number of klein bottles in the disjoint union
    n_sphere : int
        Number of spheres in the disjoint union
    n_torus : int
        Number of tori in the disjoint union
    n_lower : int
        Lower bound for the random parameters used in triangulation generation
    n_upper : int
        Upper bound for the random parameters used in triangulation generation
    """

    # Only implemented for one each of klein, sphere and torus for now, but can be easily extended to more

    datapoints = np.full((no_of_points, 2), None, dtype=object)
    cnt = 0
    seen = set()

    while cnt < no_of_points:

        try:
            D1, D2 = generate_arbitrary_disjoint_union_datapoint(
                n_lower, n_upper, n_klein, n_sphere, n_torus
            )
            key = _pair_key(D1, D2)
            if key in seen:
                continue
            seen.add(key)
            datapoints[cnt, 0] = D1
            datapoints[cnt, 1] = D2
            cnt += 1
        except Exception as e:
            print(f"Error in generate_arbitrary_disjoint_union_datapoint: {e}")
            continue

    return datapoints


def main() -> None:
    klein_plus_torus_sphere_data = generate_arbitrary_disjoint_union_dataset(
        no_of_points=510, n_lower=5, n_upper=125, n_klein=1, n_sphere=1, n_torus=1
    )
    save_datapoints(
        klein_plus_torus_sphere_data,
        folder_path="surface_triangulations/data_gen/incidence_matrix_data",
        file_name="klein_plus_torus_sphere_data.npy",
    )

    # klein_data = generate_klein_bottle_datapoints(no_of_points=500, n_lower=10, n_upper=100)
    # save_datapoints(klein_data, folder_path="surface_triangulations/data_gen/incidence_matrix_data", file_name="klein_bottle_data.npy")


if __name__ == "__main__":
    main()
