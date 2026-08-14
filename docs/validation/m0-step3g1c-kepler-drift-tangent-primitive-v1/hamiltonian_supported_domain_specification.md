# Hamiltonian and Supported Domain

The isolated canonical pair map advances only relative variables `(q,P)` with
`H_pair = P^2/(2*mu_r) - mu_r*mu_g/r`, hence `dq/dt = P/mu_r` and
`dP/dt = -mu_r*mu_g*q/r^3`. Velocity is internal `P/mu_r`; public state uses
canonical momentum. The immutable plan binds masses, parameters, layout, units,
domain limits, solver budgets, and fingerprints.

Qualification is only noncollision bound elliptic motion with `e <= 0.92`,
positive minimum periapsis, nondegenerate angular momentum, and
`|n*dt| <= 0.999*2*pi`. Invalid conics, radial/collision states, multiple
revolutions, and incompatible states are rejected.

No center-of-mass drift, multi-pair composition, interaction kick, force
callback, or N-body dynamics belongs to this primitive.
