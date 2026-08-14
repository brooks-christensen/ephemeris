# Fixed-Mass Canonical Jacobi Convention

## Boundary

This primitive maps an inertial canonical state `(x,p)` to canonical Jacobi `(q,P)`. It does not accept velocity as momentum and does not perform a velocity-to-momentum conversion. The future Step 3g0 `(x,v)` adapter remains unimplemented.

## Identity And Order

Bodies retain the exact `CompiledLayout.body_ids` order and the declared central body must be first. No body identity is inferred from an index and no mismatch is reordered. The center-of-mass pair `(q_0,P_0)` is retained. Barycenter-at-origin and zero-total-momentum constraints are not imposed.

## Forward Map

For fixed positive masses, `eta_i = sum_(j=0)^i m_j` is evaluated left-to-right in binary64. Then:

```text
q_0 = sum_j(m_j*x_j)/eta_(N-1)
P_0 = sum_j p_j
q_i = x_i - sum_(j<i)(m_j*x_j)/eta_(i-1)
P_i = (eta_(i-1)/eta_i)*p_i - (m_i/eta_i)*sum_(j<i)p_j
```

Thus `q=A*x` and `P=A^(-T)*p`. Relative coordinates are body minus the inner center of mass.

## Inverse And Tangent

The inverse uses the preregistered O(N) center-of-mass and backward-prefix recurrences. It is an algebraic inverse; general binary64 round trips are bounded rather than described as exact. The tangent map applies the same constant operators to `(delta_x,delta_p)` and is independent of base-state values.

## Units And Flattening

Rows are body-major `(x,y,z)`. Positions are metres and momenta are kg*m/s under `si_v1`. Full phase vectors flatten all body-major position rows first and all body-major momentum rows second. Symplecticity uses the resulting complete `6N` canonical matrix.
