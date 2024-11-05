import numpy as np
from pysr import PySRRegressor, jl
from julia_snippets.signum_loss import signum_loss
import sympy as sp
# from pathlib import Path

# Trying to get certain julia functions to work
# jl.seval("""
# import Combinatorics
# """)
# jl.seval("""
# function primorial(n::T) where T
#     if (0 < n && isinteger(n))
#          return T(Combinatorics.primorial(Int(n)))
#     else
#          return T(NaN)
#     end
# end
# """)

jl.seval("import Primes")
jl.seval("""
function p(i::T) where T
    if (0.5 < i < 1000)
        return T(Primes.prime(round(Int, i)))
    else
        return T(NaN)
    end
end
""")


class sympy_p(sp.Function):
    pass


loss = signum_loss
primes = {i: jl.p(i*1.0) for i in range(1, 999)}

X = np.random.randint(0, 100, 100)[:, None]
y = [primes[3*X[i, 0] + 1] - 5 for i in range(100)]


model = PySRRegressor(
    populations=32,
    niterations=50,
    ncycles_per_iteration=1000,
    binary_operators=["+", "*", "logical_or", "logical_and"],
    unary_operators=[
        "exp",
        "neg",
        "p"
        # ^ Custom operator (julia syntax)
    ],
    extra_sympy_mappings={"p": sympy_p},
    # ^ Define operator for SymPy as well
    # elementwise_loss="loss(prediction, target) = (prediction - target)^2",
    # ^ Custom loss function (julia syntax)
    loss_function=loss,
)

model.fit(X, y)
print(model.get_best().equation)
