import os

import numpy as np
import pandas as pd
from pysr import PySRRegressor, jl


def load_datafile(filename: str) -> np.ndarray:
    """
    Load the dataset from the specified file.

    Args:
        filepath (str): The path to the file containing the dataset.

    Returns:
        np.ndarray: The dataset.
    """
    filepath = os.path.join(
        os.getcwd(), "data_gen", "incidence_matrix_dataframes", filename
    )
    dataset = pd.read_csv(filepath, header=None).to_numpy()
    return dataset


def create_regressor_model(loss_func) -> PySRRegressor:

    model = PySRRegressor(
        populations=32,
        niterations=50,
        ncycles_per_iteration=1000,
        binary_operators=["+", "*", "logical_or", "logical_and"],
        unary_operators=[
            "neg",
        ],
        # elementwise_loss="loss(prediction, target) = (prediction - target)^2",
        # ^ Custom loss function (julia syntax)
        loss_function=loss_func,
    )
    return model


# Load the signum_loss object from the julia_snippets.signum_loss module
jl.include("ai_lakatos/julia_snippets/signum_loss.jl")
# signum_loss = a.signum_loss

loss = jl.seval("""SignumLoss([(1, 1), (2, 1), (2, 2)]; complexityWeight=0.1)""")


X_sphere = load_datafile("sphere_dataset.csv")[1:, 1:]
X_torus = load_datafile("torus_dataset.csv")[1:, 1:]
# X = pd.concat([X_sphere, X_torus], axis=0, ignore_index=True)
X = np.concatenate((X_sphere, X_torus), axis=0)

variable_names = [
    "width_D1",
    "width_D2",
    "height_D1",
    "height_D2",
    "rank_D1",
    "rank_D2",
    "nullity_D1",
    "nullity_D2",
    "H1",
]
y = np.zeros(len(X))

model = create_regressor_model(loss_func=loss)
model.fit(X, y, variable_names=variable_names)
print(model.get_best().equation)
# print(model.get_best().to_latex())
