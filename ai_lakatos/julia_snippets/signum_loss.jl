using DynamicExpressions
using LossFunctions
using Statistics
using SymbolicRegression: Dataset

struct SignumLoss <: Function
  requiredOps::AbstractVector{Tuple{Integer,Integer}}
  complexityWeight::Real
  unusedFunctionPenalty::Real
  punishConstant::Real
  logicalErrorPenalty::Real
  # tightnessStrength::Real
  # tightnessLengthScale::Real
end

function SignumLoss(requiredOps=[(1, 2), (2, 3), (2, 4)]; complexityWeight=0.5, unusedFunctionPenalty=1e1, punishConstant=1e3, logicalErrorPenalty=1e6)
  return SignumLoss(requiredOps, complexityWeight, unusedFunctionPenalty, punishConstant, logicalErrorPenalty)
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

  # Punish using constants
  constant_count = 0
  foreach(tree) do node
    if node.constant
      constant_count += 1
    end
  end

  unusedFunctionCount = sum(.!values(required_f))
  if unusedFunctionCount > 0
    # Short circuit return if not all required functions are used
    return unusedFunctionCount * loss.unusedFunctionPenalty
  end

  # Require that `implies` always acts on an atomic formula. Both left and right children must either have `logical_neg` or `equals` as an operator.
  equals_index = 3
  implies_index = 4
  logical_neg_index = 6

  num_logical_errors = 0
  foreach(tree) do node
    is_implies_node = node.degree == 2 && node.op == implies_index
    if is_implies_node
      left_child = node.l
      right_child = node.r
      if (left_child.degree == 2 && left_child.op != equals_index) || (left_child.degree == 1 && left_child.op != logical_neg_index) || (left_child.degree == 0) || (right_child.degree == 2 && right_child.op != equals_index) || (right_child.degree == 1 && right_child.op != logical_neg_index) || (right_child.degree == 0)
        num_logical_errors += 1
      end
    end
  end

  # if num_logical_errors > 0
  #   # Short circuit return if logical errors are present (that is, `implies` is not used correctly)
  #   return L(loss.logicalErrorPenalty)
  # end

  # Evaluate expression tree
  y_pred, ok = eval_tree_array(tree, X, options)
  if !ok
    return L(Inf)
  end

  # Signum loss
  signumLoss = mean(y_pred .= y) + 1 / complexity * loss.complexityWeight + constant_count * loss.punishConstant + num_logical_errors * loss.logicalErrorPenalty
  return L(signumLoss)
end

# # Logical error penalty
# logical_error_implies = any(tree) do node

#   is_implies_node = node.degree == 2 && node.op == implies_index

#   if is_implies_node
#     any(node) do node2
#       (node2.degree == 2 && node2.op != equals_index) || (node2.degree == 1 && node2.op != logical_neg_index) || (node2.degree == 0)
#     end
#   else
#     false
#   end
# end

# if logical_error_implies
#   # Short circuit return if logical errors are present (that is, `implies` is not used correctly)
#   return L(loss.logicalErrorPenalty)
# end
