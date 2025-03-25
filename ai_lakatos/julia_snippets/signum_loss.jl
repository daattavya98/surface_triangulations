using DynamicExpressions
using LossFunctions
using Statistics
using SymbolicRegression: Dataset

struct SignumLoss <: Function
  complexityWeight::Real
  unusedFunctionPenalty::Real
  punishConstant::Real
  logicalErrorPenalty::Real
  logicalNegUsedBoost::Real
  equalsUsedBoost::Real
  impliesUsedBoost::Real
  outerNodePenalty::Real
  equalsLogicalCompErrorPenalty::Real
  impliesLogicalCompErrorPenalty::Real
  negLogicalCompErrorPenalty::Real
  plusLogicalCompErrorPenalty::Real
  timesLogicalCompErrorPenalty::Real
  minusLogicalCompErrorPenalty::Real
  # tightnessStrength::Real
  # tightnessLengthScale::Real
end

function SignumLoss(;
  complexityWeight=0.5,
  unusedFunctionPenalty=1e1,
  punishConstant=1e3,
  logicalErrorPenalty=1e6,
  logicalNegUsedBoost=1e2,
  equalsUsedBoost=1e2,
  impliesUsedBoost=1e2,
  outerNodePenalty=1e2,
  equalsLogicalCompErrorPenalty=1e2,
  impliesLogicalCompErrorPenalty=1e2,
  negLogicalCompErrorPenalty=1e2,
  plusLogicalCompErrorPenalty=1e2,
  timesLogicalCompErrorPenalty=1e2,
  minusLogicalCompErrorPenalty=1e2
  )
  return SignumLoss(
    complexityWeight,
    unusedFunctionPenalty,
    punishConstant,
    logicalErrorPenalty,
    logicalNegUsedBoost,
    equalsUsedBoost,
    impliesUsedBoost,
    outerNodePenalty,
    equalsLogicalCompErrorPenalty,
    impliesLogicalCompErrorPenalty,
    negLogicalCompErrorPenalty,
    plusLogicalCompErrorPenalty,
    timesLogicalCompErrorPenalty,
    minusLogicalCompErrorPenalty
)
end

function (loss::SignumLoss)(tree, dataset::Dataset{T,L}, options, idx) where {T,L}
  X = copy(dataset.X)
  y = copy(dataset.y)
  if idx !== nothing  # Batching support
    X = X[:, idx]
    y = y[idx]
  end

  complexity = sum(1 for _ in tree)

  # Punish not using certain operators
  # required_f = loss.requiredOps
  # required_f = Dict{Tuple{Integer,Integer},Integer}([f => false for f in required_f])
  # foreach(tree) do node
  #   complexity += 1
  #   if node.degree > 0 && node.feature == 0 && !(node.constant)
  #     required_f[(node.degree, node.op)] = true
  #   end
  # end

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
  minus_index = 5
  equals_index = 3
  implies_index = 4
  logical_neg_index = 6

  logical_neg_used_count = 0
  equals_used_count = 0
  implies_used_count = 0

  logical_neg_logical_error_count = 0
  equals_logical_error_count = 0
  implies_logical_error_count = 0
  plus_logical_error_count = 0
  times_logical_error_count = 0
  minus_logical_error_count = 0

  for node in tree
    is_equals_node = node.degree == 2 && node.op == equals_index
    is_implies_node = node.degree == 2 && node.op == implies_index
    is_neg_node = node.degree == 1 && node.op == logical_neg_index
    is_plus_node = node.degree == 2 && node.op == plus_index
    is_times_node = node.degree == 2 && node.op == times_index
    is_minus_node = node.degree == 2 && node.op == minus_index

    if is_equals_node
      equals_used_count += 1
      left_child = node.l
      right_child = node.r
      if (left_child.degree == 2 && left_child.op == equals_index) || (left_child.degree == 1 && left_child.op == logical_neg_index) || (left_child.degree == 2 && left_child.op == implies_index) || (right_child.degree == 2 && right_child.op == equals_index) || (right_child.degree == 1 && right_child.op == logical_neg_index) || (right_child.degree == 2 && right_child.op == implies_index)
        equals_logical_error_count += 1
      end
    end
    if is_implies_node
      implies_used_count += 1
      left_child = node.l
      right_child = node.r
      if (left_child.degree == 2 && left_child.op != equals_index) || (left_child.degree == 1 && left_child.op != logical_neg_index) || (left_child.degree == 0) || (right_child.degree == 2 && right_child.op != equals_index) || (right_child.degree == 1 && right_child.op != logical_neg_index) || (right_child.degree == 0)
        implies_logical_error_count += 1
      end
    end
    if is_neg_node
      logical_neg_used_count += 1
      child = node.l
      if (child.degree == 2 && (child.op != equals_index)) || (child.degree == 2 && (child.op != implies_index)) || (child.degree == 1 && child.op != logical_neg_index) || (child.degree == 0 && child.constant) || (child.degree == 0 && !child.constant)
        logical_neg_logical_error_count += 1
      end
    end
    if is_plus_node
      left_child = node.l
      right_child = node.r
      if (left_child.degree == 2 && left_child.op == implies_index) || (left_child.degree == 2 && left_child.op == equals_index) || (left_child.degree == 1 && left_child.op == logical_neg_index) || (right_child.degree == 2 && right_child.op == implies_index) || (right_child.degree == 2 && right_child.op == equals_index) || (right_child.degree == 1 && right_child.op == logical_neg_index)
        plus_logical_error_count += 1
      end
    end
    if is_times_node
      left_child = node.l
      right_child = node.r
      if (left_child.degree == 2 && left_child.op == implies_index) || (left_child.degree == 2 && left_child.op == equals_index) || (left_child.degree == 1 && left_child.op == logical_neg_index) || (right_child.degree == 2 && right_child.op == implies_index) || (right_child.degree == 2 && right_child.op == equals_index) || (right_child.degree == 1 && right_child.op == logical_neg_index)
        times_logical_error_count += 1
      end
    end
    if is_minus_node
      left_child = node.l
      right_child = node.r
      if (left_child.degree == 2 && left_child.op == implies_index) || (left_child.degree == 2 && left_child.op == equals_index) || (left_child.degree == 1 && left_child.op == logical_neg_index) || (right_child.degree == 2 && right_child.op == implies_index) || (right_child.degree == 2 && right_child.op == equals_index) || (right_child.degree == 1 && right_child.op == logical_neg_index)
        minus_logical_error_count += 1
      end
    end
  end

  # if num_logical_errors > 0
  #   # Short circuit return if logical errors are present (that is, `implies` is not used correctly)
  #   # println("Logical errors detected: ", num_logical_errors)
  #   return L(loss.logicalErrorPenalty)
  # end

  # unusedFunctionCount = sum(.!values(required_f))
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
  signumLoss = exp(mean(y_pred .= y) - 1 / complexity * loss.complexityWeight + logical_neg_logical_error_count * loss.negLogicalCompErrorPenalty + equals_logical_error_count * loss.equalsLogicalCompErrorPenalty + implies_logical_error_count * loss.impliesLogicalCompErrorPenalty + plus_logical_error_count * loss.plusLogicalCompErrorPenalty + times_logical_error_count * loss.timesLogicalCompErrorPenalty + minus_logical_error_count * loss.minusLogicalCompErrorPenalty - logical_neg_used_count * loss.logicalNegUsedBoost - equals_used_count * loss.equalsUsedBoost - implies_used_count * loss.impliesUsedBoost + outer_penalty)
  return L(signumLoss)
end

# unusedFunctionCount * loss.unusedFunctionPenalty
