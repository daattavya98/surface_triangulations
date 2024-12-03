import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


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
                adj_matrix[vertices[i], vertices[j]] = incidence_matrix[vertices[i], edge]
                adj_matrix[vertices[j], vertices[i]] = incidence_matrix[vertices[j], edge]

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
    E = csr_matrix(edge_face_matrix)    # Edge-face incidence matrix

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
        n_components, _ = connected_components(csr_matrix(link_graph), connection='strong')
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


# # Example usage: Tetrahedron incidence matrices
# vertex_edge_matrix = np.array([
#     [1, 0, 0, 0, 1, 1],
#     [0, 1, 0, 1, 0, 1],
#     [0, 0, 1, 1, 1, 0],
#     [1, 1, 1, 0, 0, 0],
# ])

# edge_face_matrix = np.array([
#     [0, 1, 1, 0],
#     [1, 0, 1, 0],
#     [1, 1, 0, 0],
#     [1, 0, 0, 1],
#     [0, 1, 0, 1],
#     [0, 0, 1, 1],
# ])


# # Cube
vertex_edge_matrix = np.array([
    [1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # v0 connects to e0, e3, e4
    [1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # v1 connects to e0, e1, e6
    [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],  # v2 connects to e5, e6, e7
    [0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0],  # v3 connects to e4, e5, e9
    [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # v4 connects to e1, e2, e10
    [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1],  # v5 connects to e2, e3, e11
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0],  # v6 connects to e7, e8, e10
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1],  # v7 connects to e8, e9, e11
])

edge_face_matrix = np.array([
    [1, 0, 0, 0, 0, 1],  # e0 connects to f0, f5
    [0, 1, 0, 0, 0, 1],  # e1 connects to f1, f5
    [0, 0, 1, 0, 0, 1],  # e2 connects to f2, f5
    [0, 0, 0, 1, 0, 1],  # e3 connects to f3, f5
    [1, 0, 0, 1, 0, 0],  # e4 connects to f0, f3
    [1, 0, 0, 0, 1, 0],  # e5 connects to f0, f4
    [1, 1, 0, 0, 0, 0],  # e6 connects to f0, f1
    [0, 1, 0, 0, 1, 0],  # e7 connects to f1, f4
    [0, 0, 1, 0, 1, 0],  # e8 connects to f2, f4
    [0, 0, 0, 1, 1, 0],  # e9 connects to f3, f4
    [0, 1, 1, 0, 0, 0],  # e10 connects to f1, f2
    [0, 0, 1, 1, 0, 0],  # e11 connects to f2, f3
])

# A random, disconnected graph
# vertex_edge_matrix = np.array([
#     [1, 1, 0, 0, 0],
#     [1, 0, 1, 0, 0],
#     [0, 1, 1, 0, 0],
#     [0, 0, 0, 1, 0],
#     [0, 0, 0, 0, 1],
# ])

# edge_face_matrix = np.array([
#     [1],
#     [1],
#     [1],
#     [0],
#     [0],
# ])

# # Möbius strip
# vertex_edge_matrix = np.array([
#     [1, 0, 1, 0, 0, 0, 0, 1, 1, 0],
#     [1, 1, 0, 0, 1, 0, 0, 0, 0, 1],
#     [0, 1, 1, 0, 0, 0, 1, 0, 0, 0],
#     [0, 0, 0, 1, 1, 1, 0, 0, 1, 0],
#     [0, 0, 0, 1, 0, 1, 1, 1, 0, 1]
# ])

# edge_face_matrix = np.array([
#     [1, 0, 0, 0, 1],
#     [1, 1, 0, 0, 0],
#     [1, 0, 0, 0, 0],
#     [0, 1, 1, 0, 0],
#     [0, 1, 0, 0, 0],
#     [0, 0, 1, 1, 0],
#     [0, 0, 1, 0, 0],
#     [0, 0, 0, 1, 1],
#     [0, 0, 0, 1, 0],
#     [0, 0, 0, 0, 1],
# ])

# # Torus
# vertex_edge_matrix = np.array([
#     [1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#      0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
#     [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
#     [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
#     [1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0,
#      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
#      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
#      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,
#      1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0,
#      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1,
#      1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
#      0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
#      0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
#      0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#      0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#      0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#      0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#      0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0,
#      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#      0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
#      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
#      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0,
#      0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1,
#      1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
#      1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
#      0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
#      0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#      0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#      0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#      0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#      0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1]
# ]
# )

# edge_face_matrix = np.array([
#     [1, 0, 0, 0, 1],
#     [1, 1, 0, 0, 0],
#     [0, 1, 0, 0, 0],
#     [0, 0, 1, 1, 0],
#     [0, 0, 1, 0, 1],
#     [0, 0, 0, 1, 1],
# ])


# Check if the simplicial complex represents a surface
is_surface = check_surface_homeomorphic(vertex_edge_matrix, edge_face_matrix)
print("The complex represents a surface:", is_surface)
