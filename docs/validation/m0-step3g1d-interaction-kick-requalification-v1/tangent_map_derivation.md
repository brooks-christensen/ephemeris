# Canonical Interaction-Kick Tangent Derivation

For the projected internal map, F_projected(q)=P F(P q) and

J_projected(q) delta_q = P J_F(P q) P delta_q.

The kick tangent is delta_q'=delta_q and
delta_momentum'=delta_momentum+h J_projected(q) delta_q. The provider
direction is delta_x=A^-1 P delta_q; its acceleration JVP is mass weighted,
transformed by A^-T, checked against the same derived closure bound, and
projected by P. Thus a COM-only direction has no internal-force response and
both force and JVP output COM rows are exact zero.

Finite-difference applicability is selected from the analytic fixture before
any ladder value is observed. The dense quadratic force, complete kick, and
fixed linear projection are AFFINE_EXACT: exact arithmetic has no Taylor
truncation term. Their gate requires independent analytic-oracle acceptance,
the unchanged 2e-7 cap at the largest epsilon and at the ladder minimum, finite
values, and consistency with
gamma_128*max(1,evaluation_scale_ratio)/epsilon, where
gamma_k=k*u/(1-k*u) and u=2^-53. No improvement count or U-shaped curve is
required. The nonlinear radial quartic is NONLINEAR_SMOOTH and retains the
unchanged Manifest 27 ladder, three-early-improvement, cap, and roundoff-turn
requirements.

The full phase Jacobian is M=[[I,0],[h J_projected,I]], so
M^T Omega M=Omega when J_projected is symmetric. The nonsymmetric control
must violate both symmetry and symplecticity gates.

Nonzero physical calls perform force once. Nonzero tangent calls perform force
then JVP once each on the same immutable state and context. Zero duration
performs neither call.
