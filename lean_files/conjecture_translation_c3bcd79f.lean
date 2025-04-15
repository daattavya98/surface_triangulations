theorem linear_map_D1_D2_ℚ
(V0 : Type*) [AddCommGroup V0] [module ℚ V0] [FiniteDimensional ℚ V0] (V1 : Type*) [AddCommGroup V1] [module ℚ V1] [FiniteDimensional ℚ V1] (V2 : Type*) [AddCommGroup V2] [module ℚ V2] [FiniteDimensional ℚ V2]
(D1 : V1 →ₗ[ℚ] V0) (D2 : V2 →ₗ[ℚ] V1) :
  ((1 = 0 + Module.finrank ℚ (linear_map.ker D2)) → ((Module.finrank ℚ V1 + Module.finrank ℚ V1) + ((1 + Module.finrank ℚ (linear_map.ker D2)) + (Module.finrank ℚ V0 + (Module.finrank ℚ V0 + Module.finrank ℚ V1))) = Module.finrank ℚ (linear_map.ker D2) + Module.finrank ℚ (linear_map.ker D2))) := by
  sorry
