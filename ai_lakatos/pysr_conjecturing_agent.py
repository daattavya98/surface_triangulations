import os
from typing import Any

import numpy as np
import pandas as pd
import sympy
from pysr import PySRRegressor, jl


class sympy_p(sympy.Function):
    pass


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


def create_regressor_model(
    loss_func: Any, maxsize: int = 15, warmup_maxsize_by: float = 1.5
) -> PySRRegressor:

    model = PySRRegressor(
        populations=32,
        niterations=50,
        ncycles_per_iteration=1000,
        binary_operators=["+", "*", "equals", "implies", "-"],
        unary_operators=[
            "logical_neg",
        ],
        # elementwise_loss="loss(prediction, target) = (prediction - target)^2",
        # ^ Custom loss function (julia syntax)
        loss_function=loss_func,
        extra_sympy_mappings={
            "equals": sympy_p,
            "implies": sympy_p,
            "logical_neg": sympy_p,
        },
        maxsize=maxsize,
        warmup_maxsize_by=warmup_maxsize_by,
    )
    return model


# Define equality operator
jl.seval(
    """
         function equals(x::T, y::T) where T
            if x == y
                return T(1.0)
            else
                return T(0.0)
            end
         end
         """
)

# Define implication operator
jl.seval(
    """
         function implies(x::T, y::T) where T
            if (x != T(0.0) && x != T(1.0)) || (y != T(0.0) && y != T(1.0))
                return T(NaN)
            elseif (x == T(1.0) && y == T(0.0))
                return T(0.0)
            else
                return T(1.0)
            end
         end
         """
)

# Define logical negation operator
jl.seval(
    """
        function logical_neg(x::T) where T
            if x == T(0.0)
                return T(1.0)
            elseif x == T(1.0)
                return T(0.0)
            else
                return T(NaN)
            end
        end
         """
)

# Load the signum_loss object from the julia_snippets.signum_loss module
jl.include("ai_lakatos/julia_snippets/signum_loss.jl")
# signum_loss = a.signum_loss


loss = jl.seval(
    """SignumLoss([(1, 1), (2, 1), (2, 3), (2, 4), (2, 5)]; complexityWeight=1.0, unusedFunctionPenalty=1e4,
                punishConstant=0.0, logicalErrorPenalty=1e3)"""
)


X_sphere = load_datafile("sphere_dataset.csv")[1:, 1:-1]
X_torus = load_datafile("torus_dataset.csv")[1:, 1:-1]
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
]
y = np.array([2.0 for _ in range(len(X))])

model = create_regressor_model(loss_func=loss, maxsize=30)
model.fit(X, y, variable_names=variable_names)
print(model.get_best().equation)
# print(model.get_best().to_latex())
