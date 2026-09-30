# Continuous route guidance: full development suite

378 trials: three worlds (meander, bend, hairpin), three frontends (ApiaViz, Sobel + colour, Ardin-style), three wiring seeds (19, 31, 43), seven release/disturbance conditions, and both initial search/passing directions.

The seven conditions are aligned; 50 cm left/right releases; ±40° initial yaw; and 50 cm left/right imposed displacement before movement step 36. Both the navigator and the avoidance reflex receive the same scheduled phase.

The tested continuous controller, RGB-only reflex, 199 × 51 input, 10 cm centreline teaching, encoders and mushroom-body memories are unchanged. Every camera view and turn is charged against the original 360 s / 2600 observation budgets; at most 200 macro movements. Original 20 cm arrival radius and collision checks are retained.

Primary outcome: arrival among all scheduled trials. Secondary outcomes: sustained route return, renewed loss, deviation, observation/time costs and failure reasons. Invalid imposed displacements remain in the primary denominator and are identified separately. Paired method comparisons average phases, then seeds/scenarios within each world. World-bootstrap intervals are descriptive; three development worlds do not support a strong significance or held-out generalization claim.

Original results remain separate. Sources, protocol, scenes and checkpoints are hashed before execution. Each world writes resumable trials; final audits, summary and figures are generated automatically. Rendered panoramas remain cached.
