from equation_to_lean_statement import translate_to_lean
from lean_dojo import LeanGitRepo

# Test with the provided formula
input_formula = (
    "implies(equals(_1, _0 + nullity_D2), equals((width_D1 + width_D1) + "
    "((_1 + nullity_D2) + (height_D1 + (height_D1 + height_D2))), nullity_D2 + nullity_D2))"
)

theorem_name = "theorem linear_map_D1_D2_ℚ"

lean_theorem_to_prove = translate_to_lean(input_formula)
theorem_file_name = "lean_files/conjecture_translation_6f36e6cf.lean"
path_to_theorems = (
    "/Users/daattavya/Desktop/research_projects_coding/ai-lakatos/lean_files"
)

# Create a LeanGitRepo object

repository = LeanGitRepo(path_to_theorems, "equations-lean-translation")

# lean_theorem = Theorem(repo=repository, file_path=theorem_file_name, full_name=theorem_name)
# print(lean_theorem)

# with Dojo(lean_theorem) as (dojo, init_state):
#     print(init_state)
