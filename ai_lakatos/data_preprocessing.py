import logging
import os
from typing import List, Tuple

import numpy as np
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
def main() -> None:

    # Validate sphere dataset

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
    #     logger.error("Sphere dataset is invalid.")

    # Validate torus dataset
    torus_im_dataset = load_datafile("additional_torus_incidence_matrices.npy")
    [chk, none_indices] = validate_incidence_matrices_dataset(torus_im_dataset)
    if chk:
        if len(none_indices) == 0:
            logger.info("Torus dataset is valid.")
        else:
            logger.error(
                f"Torus dataset is valid but contains None values at indices {none_indices}"
            )
            # Remove data points with None values
            torus_im_dataset = np.delete(torus_im_dataset, none_indices, axis=0)
            # Update the cleaned dataset
            cleaned_data = np.load(
                os.path.join(
                    os.getcwd(),
                    "data_gen",
                    "incidence_matrix_data",
                    "cleaned_torus_incidence_matrices.npy",
                ),
                allow_pickle=True,
            )
            cleaned_data = np.concatenate((cleaned_data, torus_im_dataset), axis=0)
            np.save(
                os.path.join(
                    os.getcwd(),
                    "data_gen",
                    "incidence_matrix_data",
                    "cleaned_torus_incidence_matrices.npy",
                ),
                cleaned_data,
            )
    else:
        logger.error("Torus dataset is invalid.")


if __name__ == "__main__":
    main()
