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
  logicalNegPenalty::Real
  equalsPenalty::Real
  impliesPenalty::Real
  outerNodePenalty::Real
  # tightnessStrength::Real
  # tightnessLengthScale::Real
end

function SignumLoss(requiredOps=[(1, 2), (2, 3), (2, 4)]; complexityWeight=0.5, unusedFunctionPenalty=1e1, punishConstant=1e3, logicalErrorPenalty=1e6, logicalNegPenalty=1e2, equalsPenalty=1e2, impliesPenalty=1e2, outerNodePenalty=1e2)
  return SignumLoss(requiredOps, complexityWeight, unusedFunctionPenalty, punishConstant, logicalErrorPenalty, logicalNegPenalty, equalsPenalty, impliesPenalty, outerNodePenalty)
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

  # # Punish using real constants
  # constant_count = 0
  # foreach(tree) do node
  #   if node.constant
  #     constant_count += 1
  #   end
  # end

  # Require that `implies` always acts on an atomic formula. Both left and right children must either have `logical_neg` or `equals` as an operator.
  # Require that `logical_neg` always acts on an atomic formula.
  # Require that the arithmetic operators act correctly.

  plus_index = 1
  times_index = 2
  minus_index = 3
  equals_index = 3
  implies_index = 4
  logical_neg_index = 6

  logical_neg_count = 0
  equals_count = 0
  implies_count = 0

  num_logical_errors = 0
  for node in tree
    is_equals_node = node.degree == 2 && node.op == equals_index
    is_implies_node = node.degree == 2 && node.op == implies_index
    is_neg_node = node.degree == 1 && node.op == logical_neg_index
    is_plus_node = node.degree == 2 && node.op == plus_index
    is_times_node = node.degree == 2 && node.op == times_index
    is_minus_node = node.degree == 2 && node.op == minus_index

    if is_equals_node
      equals_count += 1
      left_child = node.l
      right_child = node.r
      if (left_child.degree == 2 && left_child.op == equals_index) || (left_child.degree == 1 && left_child.op == logical_neg_index) || (left_child.degree == 2 && left_child.op == implies_index) || (right_child.degree == 2 && right_child.op == equals_index) || (right_child.degree == 1 && right_child.op == logical_neg_index) || (right_child.degree == 2 && right_child.op == implies_index)
        num_logical_errors += 1
      end
    end
    if is_implies_node
      implies_count += 1
      left_child = node.l
      right_child = node.r
      if (left_child.degree == 2 && left_child.op != equals_index) || (left_child.degree == 1 && left_child.op != logical_neg_index) || (left_child.degree == 0) || (right_child.degree == 2 && right_child.op != equals_index) || (right_child.degree == 1 && right_child.op != logical_neg_index) || (right_child.degree == 0)
        num_logical_errors += 1
      end
    end
    if is_neg_node
      logical_neg_count += 1
      child = node.l
      if (child.degree == 2 && (child.op != equals_index)) || (child.degree == 2 && (child.op != implies_index)) || (child.degree == 1 && child.op != logical_neg_index) || (child.degree == 0 && child.constant) || (child.degree == 0 && !child.constant)
        num_logical_errors += 1
      end
    end
    if is_plus_node
      left_child = node.l
      right_child = node.r
      if (left_child.degree == 2 && left_child.op == implies_index) || (left_child.degree == 2 && left_child.op == equals_index) || (left_child.degree == 1 && left_child.op == logical_neg_index) || (right_child.degree == 2 && right_child.op == implies_index) || (right_child.degree == 2 && right_child.op == equals_index) || (right_child.degree == 1 && right_child.op == logical_neg_index)
        num_logical_errors += 1
      end
    end
    if is_times_node
      left_child = node.l
      right_child = node.r
      if (left_child.degree == 2 && left_child.op == implies_index) || (left_child.degree == 2 && left_child.op == equals_index) || (left_child.degree == 1 && left_child.op == logical_neg_index) || (right_child.degree == 2 && right_child.op == implies_index) || (right_child.degree == 2 && right_child.op == equals_index) || (right_child.degree == 1 && right_child.op == logical_neg_index)
        num_logical_errors += 1
      end
    end
    if is_minus_node
      left_child = node.l
      right_child = node.r
      if (left_child.degree == 2 && left_child.op == implies_index) || (left_child.degree == 2 && left_child.op == equals_index) || (left_child.degree == 1 && left_child.op == logical_neg_index) || (right_child.degree == 2 && right_child.op == implies_index) || (right_child.degree == 2 && right_child.op == equals_index) || (right_child.degree == 1 && right_child.op == logical_neg_index)
        num_logical_errors += 1
      end
    end
  end

  # if num_logical_errors > 0
  #   # Short circuit return if logical errors are present (that is, `implies` is not used correctly)
  #   # println("Logical errors detected: ", num_logical_errors)
  #   return L(loss.logicalErrorPenalty)
  # end

  unusedFunctionCount = sum(.!values(required_f))
  # if unusedFunctionCount > 0
  #   # Short circuit return if not all required functions are used
  #   return unusedFunctionCount * loss.unusedFunctionPenalty
  # end

  outerNodeLogical = false
  if (tree.degree == 1 && tree.op == logical_neg_index) || (tree.degree == 2 && tree.op == implies_index)
    outerNodeLogical = true
  end

  if outerNodeLogical
    outer_penalty = 0
  else
    outer_penalty = loss.outerNodePenalty
  end

  # Evaluate expression tree
  y_pred, ok = eval_tree_array(tree, X, options)
  if !ok
    return L(Inf)
  end


  # Signum loss
  signumLoss = exp(mean(y_pred .= y) - 1 / complexity * loss.complexityWeight + num_logical_errors * loss.logicalErrorPenalty - logical_neg_count * loss.logicalNegPenalty - equals_count * loss.equalsPenalty - implies_count * loss.impliesPenalty + outer_penalty)
  return L(signumLoss)
end

# unusedFunctionCount * loss.unusedFunctionPenalty
