import os
from typing import Tuple

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

        # Check if the link graph is a single connected cycle
        n_components, _ = connected_components(
            csr_matrix(link_graph), connection="strong"
        )
        degree_counts = np.sum(link_graph, axis=1)

        if n_components != 1 or np.any(degree_counts != 2):
            # print(f"Link check failed for vertex {vertex}")
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
    # filtered_rows = [row for row in genus_1_datapoints if row is not None and row[0] is not None and row[1] is not None]
    # genus_1_datapoints = np.array(filtered_rows, dtype=object)

    genus_1_datapoints = np.array(genus_1_datapoints, dtype=object)

    return genus_1_datapoints


def save_datapoints(datapoints, folder_path, file_name):
    file_path = os.path.join(folder_path, file_name)
    np.save(file_path, datapoints)


def main() -> None:

    genus_1_datapoints = generate_genus_1_datapoints(
        n_lower=5, n_upper=100, no_of_points=200
    )
    print(genus_1_datapoints.shape)
    print(genus_1_datapoints[0])

    save_datapoints(
        genus_1_datapoints,
        "data_gen/incidence_matrix_data",
        "tori_incidence_matrices.npy",
    )


if __name__ == "__main__":
    main()
