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
def parse_expression(s):
    tokens = tokenize(s)
    node = parse_sum(tokens)
    if tokens:
        raise ValueError("Extra tokens remaining after parsing")
    return node

def parse_sum(tokens):
    node = parse_term(tokens)
    while tokens and tokens[0] in {"+", "-"}:
        op = tokens.pop(0)
        right = parse_term(tokens)
        node = Node("plus" if op == "+" else "minus", [node, right])
    return node

def parse_term(tokens):
    node = parse_factor(tokens)
    while tokens and tokens[0] == "*":
        tokens.pop(0)
        right = parse_factor(tokens)
        node = Node("times", [node, right])
    return node

def parse_factor(tokens):
    if not tokens:
        raise ValueError("Unexpected end of tokens in factor.")
    if tokens[0] == "(":
        tokens.pop(0)
        node = parse_sum(tokens)
        if not tokens or tokens[0] != ")":
            raise ValueError("Expected ')' in factor.")
        tokens.pop(0)
        return node
    else:
        token = tokens.pop(0)
        if tokens and tokens[0] == "(":
            tokens.pop(0)
            args = []
            if tokens and tokens[0] != ")":
                while True:
                    args.append(parse_sum(tokens))
                    if tokens and tokens[0] == ",":
                        tokens.pop(0)
                    else:
                        break
            if not tokens or tokens[0] != ")":
                raise ValueError("Expected ')' after function call arguments.")
            tokens.pop(0)
            return Node(token, args)
        else:
            return Node(token)

# --- TRANSLATION FUNCTIONS ---
def translate_node(node):

    symbol_mapping = {
        # Arithmetic constants.
        "_0": "(0 : ℤ)",
        "_1": "(1 : ℤ)",
        # Linear map–related symbols.
        "width_D1": "(Module.finrank R V1 : ℤ)",
        "height_D1": "(Module.finrank R V0 : ℤ)",
        "height_D2": "(Module.finrank R V1 : ℤ)",
        "width_D2": "(Module.finrank R V2 : ℤ)",
        "rank_D1": "(Module.finrank R (LinearMap.range D1) : ℤ)",
        "rank_D2": "(Module.finrank R (LinearMap.range D2) : ℤ)",
        "nullity_D1": "(Module.finrank R (LinearMap.ker D1) : ℤ)",
        "nullity_D2": "(Module.finrank R (LinearMap.ker D2) : ℤ)"
    } 
    
    if not node.args:
        return symbol_mapping.get(node.func, node.func)
    if node.func == "logical_neg":
        return "¬(" + translate_node(node.args[0]) + ")"
    if node.func == "logical_and":
        return "(" + translate_node(node.args[0]) + " ∧ " + translate_node(node.args[1]) + ")"
    if node.func == "equals":
        return "(" + translate_node(node.args[0]) + " = " + translate_node(node.args[1]) + ")"
    if node.func == "implies":
        return "(" + translate_node(node.args[0]) + " → " + translate_node(node.args[1]) + ")"
    if node.func == "plus":
        return "(" + translate_node(node.args[0]) + " + " + translate_node(node.args[1]) + ")"
    if node.func == "minus":
        return "(" + translate_node(node.args[0]) + " - " + translate_node(node.args[1]) + ")"
    if node.func == "times":
        return "(" + translate_node(node.args[0]) + " • " + translate_node(node.args[1]) + ")"
    # fallback for other function calls
    args_str = ", ".join(translate_node(arg) for arg in node.args)
    return f"{node.func}({args_str})"

def gather_premises(node):
    premises = []
    if node.func == "logical_and" and len(node.args) == 2:
        premises.extend(gather_premises(node.args[0]))
        premises.extend(gather_premises(node.args[1]))
    else:
        premises.append(translate_node(node))
    return premises

def reindex_fixed_premises(fixed_premises):
    lines = fixed_premises.strip().splitlines()
    reindexed = []
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        if (line.startswith("(") and line.endswith(")")) or (line.startswith("[") and line.endswith("]")):
            inner = line[1:-1].strip()
            if ":" in inner:
                _, rest = inner.split(":", 1)
                new = f"{line[0]}h{i} : {rest.strip()}{line[-1]}"
            else:
                new = f"{line[0]}h{i} : {inner}{line[-1]}"
            reindexed.append(new)
        else:
            reindexed.append(line)
    return reindexed

def translate_fol_to_lean(fol_str, definitions, fixed_premises, theorem_name="conjecture"):
    expr = parse_expression(fol_str)
    definitions_part = definitions.strip()
    fixed_params = reindex_fixed_premises(fixed_premises)
    n_fixed = len(fixed_params)

    # Decide if it's an implication
    is_imp = (expr.func == "implies" and len(expr.args) == 2)
    if is_imp:
        # gather auto premises and conclusion
        auto_premises = gather_premises(expr.args[0])
        conclusion = translate_node(expr.args[1])
    else:
        auto_premises = []
        conclusion = translate_node(expr)

    # Build the theorem signature
    decl = f"theorem {theorem_name} {definitions_part}"
    if fixed_params:
        decl += " " + " ".join(fixed_params)
    for j, prem in enumerate(auto_premises, start=n_fixed):
        decl += f" (h{j} : {prem})"
    decl += f" : {conclusion} := by"

    return "\n".join([decl, "  sorry"])

# --- EXAMPLE USAGE ---
if __name__ == "__main__":
    definitions = """
(V0 : Type u) [AddCommGroup V0] [Module R V0] [FiniteDimensional R V0]
(V1 : Type u) [AddCommGroup V1] [Module R V1] [FiniteDimensional R V1]
(V2 : Type u) [AddCommGroup V2] [Module R V2] [FiniteDimensional R V2]
(D1 : V1 →ₗ[R] V0) (D2 : V2 →ₗ[R] V1)
    """

    fixed_premises = """
(h_fixed1 : (Module.finrank R (LinearMap.range D1) : ℤ) + (Module.finrank R (LinearMap.ker D1) : ℤ) = (Module.finrank R V1 : ℤ))
(h_fixed2 : (Module.finrank R (LinearMap.range D2) : ℤ) + (Module.finrank R (LinearMap.ker D2) : ℤ) = (Module.finrank R V2 : ℤ))
    """

