signum_loss = """
struct SignumLoss <: Function
  requiredOps::AbstractVector{Tuple{Integer,Integer}}
  complexityWeight::Real
  unusedFunctionPenalty::Real
  # tightnessStrength::Real
  # tightnessLengthScale::Real
end

function SignumLoss(requiredOps=[(1, 2), (2, 3), (2, 4)]; complexityWeight=1.0, unusedFunctionPenalty=1e5)
  return SignumLoss(requiredOps, complexityWeight, unusedFunctionPenalty)
end

function (loss::SignumLoss)(tree, dataset::Dataset{T,L}, options, idx) where {T,L}
  X = copy(dataset.X)
  y = copy(dataset.y)
  if idx !== nothing  # Batching support
    X = X[:, idx]
    y = y[idx]
  end

  complexity = 0  # Start at 4 for the 4 basic operators

  # Punish not using certain operators
  required_f = loss.requiredOps
  required_f = Dict{Tuple{Integer,Integer},Integer}([f => false for f in required_f])
  foreach(tree) do node
    complexity += 1
    if node.degree > 0 && node.feature == 0 && !(node.constant)
      required_f[(node.degree, node.op)] = true
    end
  end

  unusedFunctionCount = sum(.!values(required_f))
  if unusedFunctionCount > 0
    # Short circuit return if not all required functions are used
    return unusedFunctionCount * loss.unusedFunctionPenalty
  end

  # Evaluate expression tree
  y_pred, ok = eval_tree_array(tree, X, options)
  if !ok
    return L(Inf)
  end

  # Signum loss
  signumLoss = mean(y_pred .> y) + 1 / complexity * loss.complexityWeight
  return L(signumLoss)
end
"""
