import logging
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from beartype import beartype

rng = np.random.default_rng(seed=42)

# Configure logging
logging.basicConfig(
    filename=os.path.join(
        os.getcwd(), "surface_triangulations", "logs", "data_preprocessing.log"
    ),
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
    filepath = os.path.join(
        os.getcwd(),
        "surface_triangulations",
        "data_gen",
        "incidence_matrix_data",
        filename,
    )
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
def first_homology_of_surface(genus_1: int = 0, genus_2: int = 0) -> int:
    return 2 * (genus_1 + genus_2)


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
    zero_betti = height_vertex_edge - rank_vertex_edge
    one_betti = nullity_vertex_edge - rank_edge_face
    two_betti = width_edge_face - rank_edge_face

    return [
        width_vertex_edge,
        width_edge_face,
        height_vertex_edge,
        height_edge_face,
        rank_vertex_edge,
        rank_edge_face,
        nullity_vertex_edge,
        nullity_edge_face,
        zero_betti,
        one_betti,
        two_betti,
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
            "b0",
            "b1",
            "b2",
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
        os.path.join(
            os.getcwd(),
            "surface_triangulations",
            "data_gen",
            "incidence_matrix_dataframes",
            filename,
        )
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
def validate_homology(dataset: np.ndarray, b0: int, b1: int, b2: int) -> List:
    """
    Check if each datapoint has valid b0, b1 and b2. Returns a list of indices for datapoints whose
    homoloy is incorrect

    Parameters
    ----------
    dataset : np.ndarray
        datafile to check
    b0 : int
        correct b0 value for the dataset
    b1 : int
        correct b1 value for the dataset
    b2 : int
        correct b2 value for the dataset

    Returns
    -------
    List
        indices of datapoints with incorrect homology
    """

    incorrect_homology_indices = []
    for idx, datapoint in enumerate(dataset):
        features = compute_features(datapoint)
        if features[8] != b0 or features[9] != b1 or features[10] != b2:
            incorrect_homology_indices.append(idx)
            logger.error(
                f"Data point {idx}: Incorrect homology. Expected (b0, b1, b2) = ({b0}, {b1}, {b2}), found (b0, b1, b2) = ({features[8]}, {features[9]}, {features[10]})"
            )
    if len(incorrect_homology_indices) == 0:
        logger.info("All dataset entries have correct homology.")
        return []
    else:
        logger.error(
            f"Data points {incorrect_homology_indices} have incorrect homology."
        )
        return incorrect_homology_indices


@beartype
def cleanup_dataset(datafile_path: str) -> np.ndarray:
    """
    Load the dataset, validate it, and remove any data points that are invalid.

    Parameters
    ----------
    datafile_path : str
        The path to the dataset file.

    Returns
    -------
    np.ndarray
        The cleaned dataset with invalid data points removed.
    """
    dataset = load_datafile(datafile_path)
    [chk, none_indices] = validate_incidence_matrices_dataset(dataset)
    if chk:
        if len(none_indices) == 0:
            logger.info(f"{datafile_path}: Dataset is valid.")
            return dataset
        else:
            logger.error(
                f"{datafile_path}: Dataset is valid but contains None values at indices {none_indices}"
            )
            # Remove datapoints with None values
            cleaned_dataset = np.delete(dataset, none_indices, axis=0)
            # Remove datapoints with incorrect homology
            incorrect_homology_indices = validate_homology(
                cleaned_dataset, b0=2, b1=0, b2=2
            )
            if len(incorrect_homology_indices) > 0:
                cleaned_dataset = np.delete(
                    cleaned_dataset, incorrect_homology_indices, axis=0
                )
            return cleaned_dataset
    else:
        logger.error(f"{datafile_path}: Dataset is invalid.")
        return np.array([])


def break_chain_complices(csv_file: str, no_of_violations: int):
    """
    Violate the chain complex condition by breaking height_D2 = width_D1

    Parameters
    ----------
    csv_file : str
        Dataset to break the chain complex condition for

    Returns
    -------
    str
        The path to the modified CSV file.
    """

    df = pd.read_csv(csv_file)
    indices_to_modify = rng.choice(df.index, size=no_of_violations, replace=False)
    for idx in indices_to_modify:
        modification = rng.integers(low=1, high=int(0.1 * df.at[idx, "width_δ1"]))
        df.at[idx, "width_δ1"] += modification
        df.at[idx, "rank_δ1"] += modification
        df.at[idx, "nullity_δ1"] -= modification
    modified_csv_file = csv_file.replace(".csv", "_chain_complex_violated.csv")
    modified_df = df.loc[indices_to_modify].copy()
    save_dataframe_to_csv(modified_df, filename=modified_csv_file)
    return modified_csv_file


@beartype
def main() -> None:

    file_name = os.path.join(
        os.getcwd(),
        "surface_triangulations",
        "data_gen",
        "incidence_matrix_dataframes",
        "torus_data.csv",
    )
    break_chain_complices(file_name, no_of_violations=5)

    # klein_plus_torus_plus_sphere_data = load_datafile("klein_plus_torus_plus_sphere_data.npy")
    # [chk, none_indices] = validate_incidence_matrices_dataset(klein_plus_torus_plus_sphere_data)
    # if chk:
    #     if len(none_indices) == 0:
    #         logger.info("klein plus torus plus sphere dataset is valid.")
    #     else:
    #         logger.error(f"klein plus torus plus sphere dataset is valid but contains None values at indices {none_indices}")
    #         # Remove data points with None values
    #         klein_plus_torus_plus_sphere_data = np.delete(klein_plus_torus_plus_sphere_data, none_indices, axis=0)
    # else:
    #     logger.error("klein plus torus plus sphere dataset is invalid.")
    #     return
    # # compute features and save to dataframe
    # df = dataset_to_dataframe(klein_plus_torus_plus_sphere_data)
    # save_dataframe_to_csv(df, filename="klein_plus_torus_plus_sphere_data.csv")

    # df1 = pd.read_csv(
    #     os.path.join(
    #         os.getcwd(),
    #         "surface_triangulations",
    #         "data_gen",
    #         "incidence_matrix_dataframes",
    #         "extra_torus_data.csv",
    #     )
    # )
    # df2 = pd.read_csv(
    #     os.path.join(
    #         os.getcwd(),
    #         "surface_triangulations",
    #         "data_gen",
    #         "incidence_matrix_dataframes",
    #         "torus_dataset.csv",
    #     )
    # )
    # common_columns = df1.columns.intersection(df2.columns)
    # merged_df = (
    #     pd.concat([df1[common_columns], df2[common_columns]], ignore_index=True)
    #     .drop_duplicates()
    #     .reset_index(drop=True)
    # )
    # save_dataframe_to_csv(merged_df, filename="torus_data.csv")


if __name__ == "__main__":
    main()
