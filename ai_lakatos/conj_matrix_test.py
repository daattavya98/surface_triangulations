import numpy as np
from pysr import PySRRegressor, jl

# Julia imports
jl.seval(
    """
    import Pkg
    Pkg.add("LinearAlgebra")
    using LinearAlgebra
"""
)

# Load the signum_loss object from the julia_snippets.signum_loss module
jl.include("ai_lakatos/julia_snippets/signum_loss.jl")

loss = jl.seval("""SignumLoss([(2, 2), (2, 3)])""")

# Prepare x and y data

# X = np.load("data_gen/random_matrices.npy")

# Prepare X data
X = np.random.rand(100, 2)

y = np.zeros(X.shape[0])

model = PySRRegressor(
    populations=32,
    niterations=50,
    ncycles_per_iteration=1000,
    binary_operators=["+", "*", "logical_or", "logical_and"],
    unary_operators=[
        "LinearAlgebra.rank",
        "exp",
        "neg",
        # ^ Custom operator (julia syntax)
    ],
    # ^ Define operator for SymPy as well
    # elementwise_loss="loss(prediction, target) = (prediction - target)^2",
    # ^ Custom loss function (julia syntax)
    loss_function=loss,
    constraints={"logical_or": (3, 3), "logical_and": (3, 3), "neg": 3},
)

model.fit(X, y)
print(model.get_best().equation)
