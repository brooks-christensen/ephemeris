# Canonical Interaction Kick and Provider Specification

## Qualified Projected Map

The retained-COM canonical Jacobi map is q'=q and
momentum'=momentum+h F_projected(q), with
F_projected(q)=P F(P q). The projector P replaces canonical row zero with
exact binary64 zeros and leaves internal rows unchanged.

The v2 force boundary returns inertial Cartesian acceleration. The explicit
fixed-mass adapter applies x=A^-1 q, f_x=diag(m) a(x), and
F_q=A^-T f_x; it never relabels acceleration as generalized force.

Before projection, each COM residual is checked componentwise against
B_axis=gamma_(2n-1)*kappa_inf(A^-T)*sum_i abs(m_i*a_i_axis), where binary64
unit roundoff is u=2^-53 and gamma_k=k*u/(1-k*u). The raw residual,
component bounds, norms, conditioning, term count, and projection flag are
immutable result metadata. Above-bound closure fails before a result exists.

Only the synthetic dense quadratic and nonlinear radial quartic providers were
qualified. No physical force provider or integration was evaluated.
