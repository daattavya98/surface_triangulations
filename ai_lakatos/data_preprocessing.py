import logging
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from beartype import beartype

# Configure logging
logging.basicConfig(
    filename=os.path.join(os.getcwd(), "ai_lakatos", "logs", "data_preprocessing.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@beartype
def validate_incidence_matrices_dataset(dataset: np.ndarray) -> Tuple[bool, List]:
    """
    Validate the vertexd-edge and edge-face incidence matrices by checking that the number of edges
    are consistent and ensuring that the number of non-zero elements are correct for each column representing
    an edge and face respectively.

    Args:
        dataset (np.ndarray): The dataset to validate.

    Returns:
        Tuple[bool, List]: bool: Whether the dataset is valid or not. List: A list of indices of None values.
    """

    none_indices = []
    for idx, data_point in enumerate(dataset):
        vertex_edge_incidence_matrix = data_point[0]
        edge_face_incidence_matrix = data_point[1]

        if vertex_edge_incidence_matrix is None or edge_face_incidence_matrix is None:
            logger.error(
                f"Data point {idx}: Vertex-edge or edge-face incidence matrix is None"
            )
            none_indices.append(idx)
            continue

        num_edges = vertex_edge_incidence_matrix.shape[1]
        num_faces = edge_face_incidence_matrix.shape[1]

        if num_edges != edge_face_incidence_matrix.shape[0]:
            logger.error(
                f"""Data point {idx}: Number of edges is not consistent between vertex-edge
                          and edge-face matrices
                         """
            )
            return False, none_indices

        for edge_idx in range(num_edges):
            nonzero_count = np.count_nonzero(vertex_edge_incidence_matrix[:, edge_idx])
            if nonzero_count != 2:
                logger.error(
                    f"Data point {idx}, Edge {edge_idx}: Expected 2 non-zero elements, found {nonzero_count}"
                )
                return False, none_indices

        for face_idx in range(num_faces):
            nonzero_count = np.count_nonzero(edge_face_incidence_matrix[:, face_idx])
            if nonzero_count != 3:
                logger.error(
                    f"Data point {idx}, Face {face_idx}: Expected 3 non-zero elements, found {nonzero_count}"
                )
                return False, none_indices

    if len(none_indices) == 0:
        logger.info("All dataset entries passed validation.")
    else:
        logger.error(
            f"Data points {none_indices} contain None values but rest are validated."
        )

    return True, none_indices


@beartype
def load_datafile(filename: str) -> np.ndarray:
    """
    Load the dataset from the specified file.

    Args:
        filepath (str): The path to the file containing the dataset.

    Returns:
        np.ndarray: The dataset.
    """
    filepath = os.path.join(os.getcwd(), "data_gen", "incidence_matrix_data", filename)
    dataset = np.load(filepath, allow_pickle=True)
    return dataset


@beartype
def height_of_matrix(matrix: np.ndarray) -> int:
    """
    Get the height of the matrix.

    Args:
        matrix (np.ndarray): The matrix to get the height of.

    Returns:
        int: The height of the matrix.
    """
    return int(matrix.shape[0])


@beartype
def width_of_matrix(matrix: np.ndarray) -> int:
    """
    Get the width of the matrix.

    Args:
        matrix (np.ndarray): The matrix to get the width of.

    Returns:
        int: The width of the matrix.
    """
    return int(matrix.shape[1])


@beartype
def rank_of_matrix(matrix: np.ndarray) -> int:
    """
    Get the rank of the matrix.

    Args:
        matrix (np.ndarray): The matrix to get the rank of.

    Returns:
        int: The rank of the matrix.
    """
    return int(np.linalg.matrix_rank(matrix))


@beartype
def nullity_of_matrix(matrix: np.ndarray) -> int:
    """
    Get the nullity of the matrix.

    Args:
        matrix (np.ndarray): The matrix to get the nullity of.

    Returns:
        int: The nullity of the matrix.
    """
    return int(width_of_matrix(matrix=matrix) - rank_of_matrix(matrix=matrix))


@beartype
def first_homology_of_surface(genus: int = 0) -> int:
    if genus == 0:
        return 0
    elif genus == 1:
        return 2


@beartype
def compute_features(row: np.ndarray) -> List:
    """
    Compute all the required features for a given pair of incidence matrices.

    Parameters
    ----------
    row : np.ndarray
        Pair of incidence matrices

    Returns
    -------
    List
        All the required features in order
    """

    vertex_edge_incidence_matrix = row[0]
    edge_face_incidence_matrix = row[1]

    width_vertex_edge = width_of_matrix(matrix=vertex_edge_incidence_matrix)
    width_edge_face = width_of_matrix(matrix=edge_face_incidence_matrix)
    height_vertex_edge = height_of_matrix(matrix=vertex_edge_incidence_matrix)
    height_edge_face = height_of_matrix(matrix=edge_face_incidence_matrix)
    rank_vertex_edge = rank_of_matrix(matrix=vertex_edge_incidence_matrix)
    rank_edge_face = rank_of_matrix(matrix=edge_face_incidence_matrix)
    nullity_vertex_edge = nullity_of_matrix(matrix=vertex_edge_incidence_matrix)
    nullity_edge_face = nullity_of_matrix(matrix=edge_face_incidence_matrix)
    first_homology = first_homology_of_surface(genus=1)

    return [
        width_vertex_edge,
        width_edge_face,
        height_vertex_edge,
        height_edge_face,
        rank_vertex_edge,
        rank_edge_face,
        nullity_vertex_edge,
        nullity_edge_face,
        first_homology,
    ]


@beartype
def dataset_to_dataframe(dataset: np.ndarray) -> pd.DataFrame:
    """
    Convert the incidence matrix dataset to a dataframe with the following features:
    - Width of the vertex-edge incidence matrix
    - Width of the edge-face incidence matrix
    - Height of the vertex-edge incidence matrix
    - Height of the edge-face incidence matrix
    - Rank of the vertex-edge incidence matrix
    - Rank of the edge-face incidence matrix
    - Nullity of the vertex-edge incidence matrix
    - Nullity of the edge-face incidence matrix
    - First Homology of surface

    Parameters
    ----------
    dataset : np.ndarray
        The cleaned dataset of points to convert to a dataframe.

    Returns
    -------
    pd.DataFrame
        The dataframe containing the features of the dataset.
    """
    computed_features = np.apply_along_axis(compute_features, axis=1, arr=dataset)
    df = pd.DataFrame(
        computed_features,
        columns=[
            "width_δ1",
            "width_δ2",
            "height_δ1",
            "height_δ2",
            "rank_δ1",
            "rank_δ2",
            "nullity_δ1",
            "nullity_δ2",
            "H1",
        ],
    )
    return df


@beartype
def save_dataframe_to_csv(df: pd.DataFrame, filename: str) -> None:
    """
    Save the dataframe to a CSV file.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to save.
    filename : str
        The name of the file to save the dataframe to.
    """
    df.to_csv(
        os.path.join(os.getcwd(), "data_gen", "incidence_matrix_dataframes", filename)
    )


""" # Validate sphere dataset

    # sphere_im_dataset = load_datafile("additional_sphere_incidence_matrices.npy")
    # [chk, none_indices] = validate_incidence_matrices_dataset(sphere_im_dataset)
    # if chk:
    #     if len(none_indices) == 0:
    #         logger.info("Sphere dataset is valid.")
    #     else:
    #         logger.error(f"Sphere dataset is valid but contains None values at indices {none_indices}")
    #         # Remove data points with None values
    #         sphere_im_dataset = np.delete(sphere_im_dataset, none_indices, axis=0)
    #         # Update the cleaned dataset
    #         cleaned_data = np.load(os.path.join(os.getcwd(), "data_gen", "incidence_matrix_data",
    #                                             "cleaned_sphere_incidence_matrices.npy"),
    #                                allow_pickle=True)
    #         cleaned_data = np.concatenate((cleaned_data, sphere_im_dataset), axis=0)
    #         np.save(os.path.join(os.getcwd(), "data_gen", "incidence_matrix_data",
    #                              "cleaned_sphere_incidence_matrices.npy"),
    #                 cleaned_data)
    # else:
    #     logger.error("Sphere dataset is invalid.") """

""" # Validate torus dataset
# torus_im_dataset = load_datafile("additional_torus_incidence_matrices.npy")
# [chk, none_indices] = validate_incidence_matrices_dataset(torus_im_dataset)
# if chk:
#     if len(none_indices) == 0:
#         logger.info("Torus dataset is valid.")
#     else:
#         logger.error(
#             f"Torus dataset is valid but contains None values at indices {none_indices}"
#         )
#         # Remove data points with None values
#         torus_im_dataset = np.delete(torus_im_dataset, none_indices, axis=0)
#         # Update the cleaned dataset
#         cleaned_data = np.load(
#             os.path.join(
#                 os.getcwd(),
#                 "data_gen",
#                 "incidence_matrix_data",
#                 "cleaned_torus_incidence_matrices.npy",
#             ),
#             allow_pickle=True,
#         )
#         cleaned_data = np.concatenate((cleaned_data, torus_im_dataset), axis=0)
#         np.save(
#             os.path.join(
#                 os.getcwd(),
#                 "data_gen",
#                 "incidence_matrix_data",
#                 "cleaned_torus_incidence_matrices.npy",
#             ),
#             cleaned_data,
#         )
# else:
#     logger.error("Torus dataset is invalid.") """


def breakup_sphere_dataset_for_git_lfs(data: np.ndarray) -> None:
    """
    Break up the sphere dataset into smaller chunks to be uploaded to Git LFS.

    Parameters
    ----------
    data : np.ndarray
        The dataset to break up.
    """
    num_chunks = 4
    chunk_size = int(data.shape[0] / num_chunks)

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size
        chunk = data[start_idx:end_idx]
        np.save(
            os.path.join(
                os.getcwd(),
                "data_gen",
                "incidence_matrix_data",
                f"sphere_incidence_matrices_{i}.npy",
            ),
            chunk,
        )


@beartype
def main() -> None:

    sphere_dataset = load_datafile("cleaned_sphere_incidence_matrices.npy")
    breakup_sphere_dataset_for_git_lfs(sphere_dataset)


if __name__ == "__main__":
    main()
