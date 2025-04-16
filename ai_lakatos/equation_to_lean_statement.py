import re

# --- TOKENIZATION ---
def tokenize(s):
    """
    Splits the input string into tokens.
    Recognizes:
      - Parentheses: ( and )
      - Brackets: [ and ]
      - Commas: ,
      - Arithmetic operators: +, -, *
      - Other sequences of non-whitespace characters.
    """
    tokens = []
    token = ""
    i = 0
    while i < len(s):
        c = s[i]
        if c.isspace():
            if token:
                tokens.append(token)
                token = ""
            i += 1
            continue
        elif c in "(),+-*[]":
            if token:
                tokens.append(token)
                token = ""
            tokens.append(c)
            i += 1
        else:
            token += c
            i += 1
    if token:
        tokens.append(token)
    return tokens

# --- PARSER NODE ---
class Node:
    def __init__(self, func, args=None):
        self.func = func
        self.args = args if args is not None else []
    def __repr__(self):
        if self.args:
            return f"{self.func}({', '.join(map(str, self.args))})"
        else:
            return self.func

# --- PARSING FUNCTIONS ---
# We implement a recursive descent parser with the grammar:
#
#   expr   -> sum
#   sum    -> term ( ("+" | "-") term )*
#   term   -> factor ( "*" factor )*
#   factor -> identifier [ "(" expr_list ")" ]
#            | "(" expr ")"
#
#   expr_list -> expr ( "," expr )*
#
# This supports function calls, arithmetic operations, and parenthesized expressions.
def parse_expression(s):
    tokens = tokenize(s)
    node = parse_sum(tokens)
    if tokens:
        raise ValueError("Extra tokens remaining after parsing")
    return node

def parse_sum(tokens):
    node = parse_term(tokens)
    # Handle addition and subtraction (left-associative)
    while tokens and tokens[0] in {"+", "-"}:
        op = tokens.pop(0)
        right = parse_term(tokens)
        if op == "+":
            node = Node("plus", [node, right])
        else:  # op == "-"
            node = Node("minus", [node, right])
    return node

def parse_term(tokens):
    node = parse_factor(tokens)
    # Handle multiplication (left-associative)
    while tokens and tokens[0] == "*":
        tokens.pop(0)  # consume '*'
        right = parse_factor(tokens)
        node = Node("times", [node, right])
    return node

def parse_factor(tokens):
    if not tokens:
        raise ValueError("Unexpected end of tokens in factor.")
    if tokens[0] == "(":
        tokens.pop(0)  # consume '('
        node = parse_sum(tokens)
        if not tokens or tokens[0] != ")":
            raise ValueError("Expected ')' in factor.")
        tokens.pop(0)  # consume ')'
        return node
    else:
        token = tokens.pop(0)
        # Check for a function call: an identifier immediately followed by "("
        if tokens and tokens[0] == "(":
            tokens.pop(0)  # consume '('
            args = []
            if tokens and tokens[0] != ")":
                while True:
                    arg = parse_sum(tokens)
                    args.append(arg)
                    if tokens and tokens[0] == ",":
                        tokens.pop(0)  # consume comma
                    else:
                        break
            if not tokens or tokens[0] != ")":
                raise ValueError("Expected ')' after function call arguments.")
            tokens.pop(0)  # consume ')'
            return Node(token, args)
        else:
            return Node(token)

# --- TRANSLATION FUNCTIONS ---
def translate_node(node):
    """
    Recursively translates an expression tree into a Lean string.
    Handles:
      - Logical operators: logical_neg, logical_and, equals, implies.
      - Arithmetic operators: plus, minus, times.
      - Symbol replacements for both arithmetic constants and linear map symbols.
    """
    symbol_mapping = {
        # Arithmetic constants.
        "_0": "0",
        "_1": "1",
        # Linear map–related symbols.
        "width_D1": "Module.finrank R V1",
        "height_D1": "Module.finrank R V0",
        "height_D2": "Module.finrank R V1",
        "width_D2": "Module.finrank R V2",
        "rank_D1": "Module.finrank R (LinearMap.range D1)",
        "rank_D2": "Module.finrank R (LinearMap.range D2)",
        "nullity_D1": "Module.finrank R (LinearMap.ker D1)",
        "nullity_D2": "Module.finrank R (LinearMap.ker D2)"
    }
    
    if not node.args:
        return symbol_mapping.get(node.func, node.func)
    
    if node.func == "logical_neg":
        if len(node.args) != 1:
            raise ValueError("logical_neg expects one argument")
        return "¬(" + translate_node(node.args[0]) + ")"
    elif node.func == "logical_and":
        if len(node.args) != 2:
            raise ValueError("logical_and expects two arguments")
        return "(" + translate_node(node.args[0]) + " ∧ " + translate_node(node.args[1]) + ")"
    elif node.func == "equals":
        if len(node.args) != 2:
            raise ValueError("equals expects two arguments")
        return "(" + translate_node(node.args[0]) + " = " + translate_node(node.args[1]) + ")"
    elif node.func == "implies":
        if len(node.args) != 2:
            raise ValueError("implies expects two arguments")
        return "(" + translate_node(node.args[0]) + " → " + translate_node(node.args[1]) + ")"
    
    elif node.func == "plus":
        if len(node.args) != 2:
            raise ValueError("plus expects two arguments")
        return "(" + translate_node(node.args[0]) + " + " + translate_node(node.args[1]) + ")"
    elif node.func == "minus":
        if len(node.args) != 2:
            raise ValueError("minus expects two arguments")
        return "(" + translate_node(node.args[0]) + " - " + translate_node(node.args[1]) + ")"
    elif node.func == "times":
        if len(node.args) != 2:
            raise ValueError("times expects two arguments")
        return "(" + translate_node(node.args[0]) + " • " + translate_node(node.args[1]) + ")"
    else:
        translated_args = ", ".join(translate_node(arg) for arg in node.args)
        return node.func + "(" + translated_args + ")"

def gather_premises(node):
    """
    Given the left-hand side of an implication, collects premises into a flat list.
    If the node is a conjunction (logical_and), splits it into its individual components.
    """
    premises = []
    if node.func == "logical_and" and len(node.args) == 2:
        premises.extend(gather_premises(node.args[0]))
        premises.extend(gather_premises(node.args[1]))
    else:
        premises.append(translate_node(node))
    return premises

def reindex_fixed_premises(fixed_premises):
    """
    Processes the fixed premises block line by line and re-indexes each parameter.
    Each nonempty line (expected to be a complete parameter enclosed in parentheses or brackets)
    is replaced with one of the form:
      (h0 : <contents>) or [h0 : <contents>]
    """
    lines = fixed_premises.strip().splitlines()
    reindexed = []
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        if (line.startswith("(") and line.endswith(")")) or (line.startswith("[") and line.endswith("]")):
            # Remove the outer parentheses/brackets.
            inner = line[1:-1].strip()
            # Try to split on the first colon.
            if ":" in inner:
                # Split into name and type (only on the first colon).
                parts = inner.split(":", 1)
                type_part = parts[1].strip()
                new_param = line[0] + f"h{i} : " + type_part + line[-1]
            else:
                new_param = line[0] + f"h{i} : " + inner + line[-1]
            reindexed.append(new_param)
        else:
            # If the line is not enclosed, leave it as is.
            reindexed.append(line)
    return reindexed

def translate_fol_to_lean(fol_str, definitions, fixed_premises, theorem_name="conjecture"):
    """
    Main translation function.
    
    Parameters:
      - fol_str: A string representing the FOL logical formula.
      - definitions: A string with fixed definitions (e.g. V0, V1, V2, D1, D2).  
                     These are inserted verbatim into the theorem signature.
      - fixed_premises: A string with fixed premises to be re‑indexed.
                        Each line is treated as one parameter and re‑indexed as h0, h1, ….
      - theorem_name: The name of the theorem (default is "conjecture").
    
    For an implication, extracts the premises from the LHS (even nested via logical_and)
    and constructs a Lean theorem whose signature consists of:
      1. The theorem name.
      2. The definitions (verbatim).
      3. The fixed premises (re‑indexed).
      4. The automatically extracted premises (indexed continuing after the fixed premises).
    The conclusion appears after a colon, and the proof is left as a placeholder.
    """
    expr = parse_expression(fol_str)
    output_lines = []
    
    # Use definitions verbatim.
    definitions_part = definitions.strip()
    # Process fixed premises and reindex them.
    fixed_params = reindex_fixed_premises(fixed_premises)
    n_fixed = len(fixed_params)
    
    if expr.func == "implies" and len(expr.args) == 2:
        A = expr.args[0]
        B = expr.args[1]
        auto_premises = gather_premises(A)
        conclusion = translate_node(B)
        
        # Build the theorem declaration.
        decl = "theorem " + theorem_name + " " + definitions_part
        if fixed_params:
            decl += " " + " ".join(fixed_params)
        for j, premise in enumerate(auto_premises, start=n_fixed):
            decl += " (h" + str(j) + " : " + premise + ")"
        decl += " : " + conclusion + " := by"
        output_lines.append(decl)
        output_lines.append("  sorry")
    else:
        # If the formula is not an implication, just output its translation.
        output_lines.append(translate_node(expr))
    
    return "\n".join(output_lines)

# --- EXAMPLE USAGE ---
if __name__ == "__main__":
    # Definitions: these will be included verbatim.
    definitions = """
(V0 : Type u) [AddCommGroup V0] [Module R V0] [FiniteDimensional R V0]
(V1 : Type u) [AddCommGroup V1] [Module R V1] [FiniteDimensional R V1]
(V2 : Type u) [AddCommGroup V2] [Module R V2] [FiniteDimensional R V2]
(D1 : V1 →ₗ[R] V0) (D2 : V2 →ₗ[R] V1)
    """
    
    # Fixed premises: each line will be re-indexed as h0, h1, etc.
    fixed_premises = """
(h_fixed1 : (Module.finrank R (LinearMap.range D1) : ℤ) + (Module.finrank R (LinearMap.ker D1) : ℤ) = (Module.finrank R V1 : ℤ))
(h_fixed2 : (Module.finrank R (LinearMap.range D2) : ℤ) + (Module.finrank R (LinearMap.ker D2) : ℤ) = (Module.finrank R V2 : ℤ))
    """

    try:
        lean_code = translate_fol_to_lean(fol_statement, definitions, fixed_premises, theorem_name="conjecture")
        print("Translated Lean Code:")
        print("---------------------")
        print(lean_code)
    except ValueError as e:
        print("Error during parsing:", e)
