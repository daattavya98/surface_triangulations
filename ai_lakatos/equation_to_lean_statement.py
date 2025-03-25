import re

# Mapping of logical symbols to lean equivalents (we no longer simply replace implies/equals)
logic_symbols = {
    "logical_neg": "¬",
}

# Mapping of arithmetic symbols to lean equivalents
arithmetic_symbols = {
    "_0": "0",
    "_1": "1",
    "+": "+",
    "-": "-",
    "*": "•",
}

# Mapping of linear algebra symbols to lean equivalents,
# with a new entry for height_D1.
linear_algebra_symbols = {
    "width_D1": "Module.finrank ℚ V1",
    "height_D1": "Module.finrank ℚ V0",  # Corrected replacement
    "height_D2": "Module.finrank ℚ V1",
    "width_D2": "Module.finrank ℚ V2",
    "rank_D1": "Module.finrank ℚ (linear_map.range D1)",
    "rank_D2": "Module.finrank ℚ (linear_map.range D2)",
    "nullity_D1": "Module.finrank ℚ (linear_map.ker D1)",
    "nullity_D2": "Module.finrank ℚ (linear_map.ker D2)",
}


def detect_maps(formula: str):
    """
    Detects if D1 and/or D2 appear in the formula.
    """
    use_D1 = "D1" in formula
    use_D2 = "D2" in formula
    return use_D1, use_D2


def split_args(s: str) -> list:
    """
    Splits the string s into two arguments at the comma that is not nested inside parentheses.
    """
    parts = []
    balance = 0
    current = []
    for char in s:
        if char == "(":
            balance += 1
        elif char == ")":
            balance -= 1
        if char == "," and balance == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    parts.append("".join(current).strip())
    return parts


def convert_ops(expr: str) -> str:
    """
    Recursively converts function-style operator calls for 'implies' and 'equals'
    into infix (postfix) notation.
    """
    expr = expr.strip()
    # Handle implies(...)
    if expr.startswith("implies(") and expr.endswith(")"):
        inner = expr[len("implies(") : -1]
        args = split_args(inner)
        # Recursively process both arguments
        left = convert_ops(args[0])
        right = convert_ops(args[1])
        return f"({left} → {right})"
    # Handle equals(...)
    if expr.startswith("equals(") and expr.endswith(")"):
        inner = expr[len("equals(") : -1]
        args = split_args(inner)
        left = convert_ops(args[0])
        right = convert_ops(args[1])
        return f"({left} = {right})"
    # Otherwise, return the expression (it might contain arithmetic or linear algebra symbols)
    return expr


def replace_symbols(formula: str) -> str:
    """
    Replaces arithmetic and linear algebra symbols using regex.
    """
    # Replace arithmetic symbols
    for symbol, lean_symbol in arithmetic_symbols.items():
        # Use word boundaries for symbols like _0 and _1.
        formula = re.sub(rf"\b{re.escape(symbol)}\b", lean_symbol, formula)
    # Replace linear algebra symbols
    for symbol, lean_symbol in linear_algebra_symbols.items():
        formula = re.sub(rf"\b{re.escape(symbol)}\b", lean_symbol, formula)
    return formula


def translate_to_lean(formula: str) -> str:
    """
    Translates a first-order logical formula into Lean syntax, considering only D1 and D2.
    """
    use_D1, use_D2 = detect_maps(formula)

    # First, replace non-operator symbols (arithmetic and linear algebra)
    formula = replace_symbols(formula)

    # Now convert implies(...) and equals(...) from function style to infix notation.
    formula = convert_ops(formula)

    # Define Lean lemma header dynamically
    vector_spaces = []
    linear_maps = []
    vector_spaces.append(
        "(V0 : Type*) [AddCommGroup V0] [module ℚ V0] [FiniteDimensional ℚ V0]"
    )
    vector_spaces.append(
        "(V1 : Type*) [AddCommGroup V1] [module ℚ V1] [FiniteDimensional ℚ V1]"
    )
    vector_spaces.append(
        "(V2 : Type*) [AddCommGroup V2] [module ℚ V2] [FiniteDimensional ℚ V2]"
    )

    if use_D1:
        linear_maps.append("(D1 : V1 →ₗ[ℚ] V0)")

    if use_D2:
        linear_maps.append("(D2 : V2 →ₗ[ℚ] V1)")

    lean_code = (
        f"lemma linear_map_D1_D2_ℚ \n"
        f"{' '.join(vector_spaces)} \n"
        f"{' '.join(linear_maps)} : \n"
        f"  {formula} := \n"
        "  sorry"  # Placeholder for proof
    )

    return lean_code


# Test with the provided formula
input_formula = (
    "implies(equals(_1, _0 + nullity_D2), equals((width_D1 + width_D1) + "
    "((_1 + nullity_D2) + (height_D1 + (height_D1 + height_D2))), nullity_D2 + nullity_D2))"
)
lean_output = translate_to_lean(input_formula)
print(lean_output)
