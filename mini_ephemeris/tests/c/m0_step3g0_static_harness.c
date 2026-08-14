#include <math.h>
#include <stddef.h>
#include <stdio.h>

int me_gr_tangent_pointwise(
    size_t n_real,
    const double* positions,
    const double* masses,
    const double* delta_positions,
    double gravitational_constant,
    double coefficient_scale,
    double c_m_per_s,
    int include_central_response,
    double* accelerations,
    double* tangent_accelerations
);

int main(void){
    const double positions[9] = {
        1.0e8, -2.0e8, 3.0e8,
        5.8e10, 1.0e10, -2.0e9,
        -9.0e10, 7.0e10, 4.0e9
    };
    const double masses[3] = {1.9884098713264225e30, 3.3009873694619664e23, 4.867305814842006e24};
    const double delta[9] = {3.0, -2.0, 1.0, 20.0, -40.0, 10.0, -7.0, 11.0, 2.0};
    double acceleration[9] = {0.0};
    double tangent[9] = {0.0};
    int result = me_gr_tangent_pointwise(
        3, positions, masses, delta, 6.67430e-11, 1.0, 299792458.0, 1, acceleration, tangent
    );
    if (result != 0){
        fprintf(stderr, "pointwise error %d\n", result);
        return 1;
    }
    for (size_t index = 0; index < 9; index++){
        if (!isfinite(acceleration[index]) || !isfinite(tangent[index])){
            fprintf(stderr, "nonfinite output at %zu\n", index);
            return 2;
        }
    }
    for (size_t axis = 0; axis < 3; axis++){
        double force = 0.0;
        double tangent_force = 0.0;
        for (size_t body = 0; body < 3; body++){
            force += masses[body] * acceleration[3 * body + axis];
            tangent_force += masses[body] * tangent[3 * body + axis];
        }
        if (fabs(force) > 1.0e8 || fabs(tangent_force) > 1.0e4){
            fprintf(stderr, "closure failure on axis %zu: %.17g %.17g\n", axis, force, tangent_force);
            return 3;
        }
    }
    return 0;
}
