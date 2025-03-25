import os
from typing import Any

import numpy as np
import pandas as pd
import sympy
from equation_to_lean_statement import translate_to_lean
from pysr import PySRRegressor, TensorBoardLoggerSpec, jl


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
    log_spec: TensorBoardLoggerSpec,
    loss_func: Any,
    maxsize: int = 15,
    warmup_maxsize_by: float = 1.5,
) -> PySRRegressor:

    model = PySRRegressor(
        populations=15,
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
        # nested_constraints={"logical_neg": {"logical_neg": 2}},
        maxsize=maxsize,
        warmup_maxsize_by=warmup_maxsize_by,
        complexity_of_constants=100,
        should_optimize_constants=False,
        weight_mutate_constant=0.0,
        logger_spec=log_spec,
    )
    return model


# Create logger
logger_spec = TensorBoardLoggerSpec(
    log_dir=os.path.join(os.getcwd(), "logs/runs"),
    log_interval=10,
)

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
            if (x == T(1.0) && y == T(0.0))
                return T(0.0)
            elseif (x == T(0.0) && y == T(1.0)) || (x == T(1.0) && y == T(1.0)) || (x == T(0.0) && y == T(0.0))
                return T(1.0)
            else
                return T(NaN)
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
    """SignumLoss(
        unusedFunctionPenalty=1e2,
        punishConstant=0.0,
        logicalErrorPenalty=1e2,
        complexityWeight=1.0,
        logicalNegUsedBoost=1.0,
        equalsUsedBoost=2.0,
        impliesUsedBoost=2.0,
        minusUsedBoost=0.5,
        plusUsedBoost=1.5,
        timesUsedBoost=0.5,
        outerNodePenalty=100.0,
        equalsLogicalCompErrorPenalty=40.0,
        impliesLogicalCompErrorPenalty=16.0,
        negLogicalCompErrorPenalty=12.0,
        plusLogicalCompErrorPenalty=2.0,
        timesLogicalCompErrorPenalty=2.0,
        minusLogicalCompErrorPenalty=2.0,
        equalsSemanticErrorPenalty=50.0,
        impliesSemanticErrorPenalty=50.0,
        dataErrorPenalty=18.0,
    )
    """
)


X_sphere = load_datafile("sphere_dataset.csv")[1:, 1:-1]
X_torus = load_datafile("torus_dataset.csv")[1:, 1:-1]
# X = pd.concat([X_sphere, X_torus], axis=0, ignore_index=True)
X = np.concatenate((X_sphere, X_torus), axis=0)
# X = np.insert(X, 8, -2, axis=1)
# X = np.insert(X, 8, -1, axis=1)
X = np.insert(X, 8, 0, axis=1)
X = np.insert(X, 9, 1, axis=1)
# X = np.insert(X, 12, 2, axis=1)

variable_names = [
    "width_D1",
    "width_D2",
    "height_D1",
    "height_D2",
    "rank_D1",
    "rank_D2",
    "nullity_D1",
    "nullity_D2",
    "_0",
    "_1",
]
# y = np.concatenate((np.array([2.0 for _ in range(len(X_sphere))]),
# np.array([0.0 for _ in range(len(X_torus))])), axis=0)
y = np.array([1.0 for _ in range(len(X))])

model = create_regressor_model(loss_func=loss, maxsize=35, log_spec=logger_spec)
model.fit(X, y, variable_names=variable_names)
a = model.get_best().equation
print(a)
lean_translation = translate_to_lean(a)
print(lean_translation)
