#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "rebound.h"

#define ME_GR_TANGENT_API_VERSION 1u
#define ME_GR_TANGENT_STATE_MAGIC UINT64_C(0x4d45475254414e47)

struct me_gr_tangent_state {
    uint64_t magic;
    double coefficient_scale;
    double c_m_per_s;
    int include_central_response;
    uint64_t callback_invocations;
    double real_gr_accel_norm_max;
    double real_gr_accel_norm_sum;
    uint64_t real_gr_accel_norm_count;
    double tangent_gr_accel_norm_max;
    double tangent_gr_accel_norm_sum;
    uint64_t tangent_gr_accel_norm_count;
    uint64_t nonfinite_result_count;
};

struct me_gr_tangent_stats {
    uint64_t callback_invocations;
    double real_gr_accel_norm_max;
    double real_gr_accel_norm_sum;
    uint64_t real_gr_accel_norm_count;
    double tangent_gr_accel_norm_max;
    double tangent_gr_accel_norm_sum;
    uint64_t tangent_gr_accel_norm_count;
    uint64_t nonfinite_result_count;
};

static void me_gr_tangent_cleanup(struct reb_simulation* r);
void me_gr_tangent_additional_forces(struct reb_simulation* const r);

static struct me_gr_tangent_state* me_state(struct reb_simulation* r) {
    struct me_gr_tangent_state* state;
    if (r == NULL || r->extras == NULL) {
        return NULL;
    }
    if (r->additional_forces != me_gr_tangent_additional_forces ||
        r->extras_cleanup != me_gr_tangent_cleanup) {
        return NULL;
    }
    state = (struct me_gr_tangent_state*)r->extras;
    return state->magic == ME_GR_TANGENT_STATE_MAGIC ? state : NULL;
}

static void me_record_norm(
    const double x,
    const double y,
    const double z,
    double* maximum,
    double* sum,
    uint64_t* count,
    uint64_t* nonfinite_count
) {
    const double norm = sqrt(x * x + y * y + z * z);
    if (!isfinite(norm)) {
        *nonfinite_count += 1u;
        return;
    }
    if (norm > *maximum) {
        *maximum = norm;
    }
    *sum += norm;
    *count += 1u;
}

static void me_pair_terms(
    const double dx,
    const double dy,
    const double dz,
    const double ddx,
    const double ddy,
    const double ddz,
    const double mu2_over_c2,
    double* ax,
    double* ay,
    double* az,
    double* tax,
    double* tay,
    double* taz
) {
    const double r2 = dx * dx + dy * dy + dz * dz;
    *ax = 0.0;
    *ay = 0.0;
    *az = 0.0;
    *tax = 0.0;
    *tay = 0.0;
    *taz = 0.0;
    if (r2 <= 0.0 || mu2_over_c2 == 0.0) {
        return;
    }

    {
        const double inv_r2 = 1.0 / r2;
        const double inv_r4 = inv_r2 * inv_r2;
        const double coefficient = -6.0 * mu2_over_c2 * inv_r4;
        const double dot = dx * ddx + dy * ddy + dz * ddz;
        const double prefactor = -6.0 * mu2_over_c2;
        const double common = 4.0 * dot * inv_r4 * inv_r2;
        *ax = coefficient * dx;
        *ay = coefficient * dy;
        *az = coefficient * dz;
        *tax = prefactor * (ddx * inv_r4 - dx * common);
        *tay = prefactor * (ddy * inv_r4 - dy * common);
        *taz = prefactor * (ddz * inv_r4 - dz * common);
    }
}

void me_gr_tangent_additional_forces(struct reb_simulation* const r) {
    struct me_gr_tangent_state* state = me_state(r);
    unsigned int n_real;
    struct reb_particle* particles;
    struct reb_particle* central;
    double central_mass;
    double mu2_over_c2;
    double central_ax = 0.0;
    double central_ay = 0.0;
    double central_az = 0.0;
    unsigned int i;
    unsigned int start;

    if (state == NULL) {
        return;
    }
    state->callback_invocations += 1u;
    if (r->N_var < 0 || (unsigned int)r->N_var > r->N) {
        state->nonfinite_result_count += 1u;
        return;
    }
    n_real = r->N - (unsigned int)r->N_var;
    if (n_real == 0u || r->particles == NULL) {
        return;
    }
    particles = r->particles;
    central = &particles[0];
    central_mass = central->m;
    if (central_mass == 0.0 || state->c_m_per_s == 0.0) {
        state->nonfinite_result_count += 1u;
        return;
    }
    mu2_over_c2 = state->coefficient_scale *
        (r->G * central_mass) * (r->G * central_mass) /
        (state->c_m_per_s * state->c_m_per_s);

    for (i = 1u; i < n_real; ++i) {
        struct reb_particle* particle = &particles[i];
        const double dx = particle->x - central->x;
        const double dy = particle->y - central->y;
        const double dz = particle->z - central->z;
        double ax;
        double ay;
        double az;
        double unused_tax;
        double unused_tay;
        double unused_taz;
        me_pair_terms(
            dx, dy, dz, 0.0, 0.0, 0.0, mu2_over_c2,
            &ax, &ay, &az, &unused_tax, &unused_tay, &unused_taz
        );
        particle->ax += ax;
        particle->ay += ay;
        particle->az += az;
        me_record_norm(
            ax, ay, az,
            &state->real_gr_accel_norm_max,
            &state->real_gr_accel_norm_sum,
            &state->real_gr_accel_norm_count,
            &state->nonfinite_result_count
        );
        if (state->include_central_response != 0) {
            const double mass_ratio = particle->m / central_mass;
            central_ax -= mass_ratio * ax;
            central_ay -= mass_ratio * ay;
            central_az -= mass_ratio * az;
        }
    }
    if (state->include_central_response != 0) {
        central->ax += central_ax;
        central->ay += central_ay;
        central->az += central_az;
    }

    /* The frozen Python oracle applies one full n_real block at a time. */
    for (start = n_real; start + n_real <= r->N; start += n_real) {
        struct reb_particle* central_var = &particles[start];
        double central_tax = 0.0;
        double central_tay = 0.0;
        double central_taz = 0.0;
        for (i = 1u; i < n_real; ++i) {
            struct reb_particle* particle = &particles[i];
            struct reb_particle* var_particle = &particles[start + i];
            const double dx = particle->x - central->x;
            const double dy = particle->y - central->y;
            const double dz = particle->z - central->z;
            const double ddx = var_particle->x - central_var->x;
            const double ddy = var_particle->y - central_var->y;
            const double ddz = var_particle->z - central_var->z;
            double unused_ax;
            double unused_ay;
            double unused_az;
            double tax;
            double tay;
            double taz;
            me_pair_terms(
                dx, dy, dz, ddx, ddy, ddz, mu2_over_c2,
                &unused_ax, &unused_ay, &unused_az, &tax, &tay, &taz
            );
            var_particle->ax += tax;
            var_particle->ay += tay;
            var_particle->az += taz;
            me_record_norm(
                tax, tay, taz,
                &state->tangent_gr_accel_norm_max,
                &state->tangent_gr_accel_norm_sum,
                &state->tangent_gr_accel_norm_count,
                &state->nonfinite_result_count
            );
            if (state->include_central_response != 0) {
                const double mass_ratio = particle->m / central_mass;
                central_tax -= mass_ratio * tax;
                central_tay -= mass_ratio * tay;
                central_taz -= mass_ratio * taz;
            }
        }
        if (state->include_central_response != 0) {
            central_var->ax += central_tax;
            central_var->ay += central_tay;
            central_var->az += central_taz;
        }
    }
}

static void me_gr_tangent_cleanup(struct reb_simulation* r) {
    struct me_gr_tangent_state* state = me_state(r);
    if (state != NULL) {
        state->magic = 0u;
        free(state);
    }
    if (r != NULL) {
        r->extras = NULL;
        if (r->additional_forces == me_gr_tangent_additional_forces) {
            r->additional_forces = NULL;
        }
        if (r->extras_cleanup == me_gr_tangent_cleanup) {
            r->extras_cleanup = NULL;
        }
    }
}

int me_gr_tangent_attach(
    struct reb_simulation* r,
    const double coefficient_scale,
    const double c_m_per_s,
    const int include_central_response
) {
    struct me_gr_tangent_state* state;
    if (r == NULL || !isfinite(coefficient_scale) ||
        !isfinite(c_m_per_s) || c_m_per_s <= 0.0) {
        return -1;
    }
    if (r->extras != NULL || r->additional_forces != NULL || r->extras_cleanup != NULL) {
        return -2;
    }
    state = (struct me_gr_tangent_state*)calloc(1u, sizeof(*state));
    if (state == NULL) {
        return -3;
    }
    state->magic = ME_GR_TANGENT_STATE_MAGIC;
    state->coefficient_scale = coefficient_scale;
    state->c_m_per_s = c_m_per_s;
    state->include_central_response = include_central_response != 0;
    r->extras = state;
    r->additional_forces = me_gr_tangent_additional_forces;
    r->extras_cleanup = me_gr_tangent_cleanup;
    r->force_is_velocity_dependent = 0u;
    return 0;
}

int me_gr_tangent_detach(struct reb_simulation* r) {
    if (me_state(r) == NULL) {
        return -1;
    }
    me_gr_tangent_cleanup(r);
    return 0;
}

int me_gr_tangent_is_attached(struct reb_simulation* r) {
    return me_state(r) != NULL;
}

int me_gr_tangent_get_stats(struct reb_simulation* r, struct me_gr_tangent_stats* output) {
    const struct me_gr_tangent_state* state = me_state(r);
    if (state == NULL || output == NULL) {
        return -1;
    }
    output->callback_invocations = state->callback_invocations;
    output->real_gr_accel_norm_max = state->real_gr_accel_norm_max;
    output->real_gr_accel_norm_sum = state->real_gr_accel_norm_sum;
    output->real_gr_accel_norm_count = state->real_gr_accel_norm_count;
    output->tangent_gr_accel_norm_max = state->tangent_gr_accel_norm_max;
    output->tangent_gr_accel_norm_sum = state->tangent_gr_accel_norm_sum;
    output->tangent_gr_accel_norm_count = state->tangent_gr_accel_norm_count;
    output->nonfinite_result_count = state->nonfinite_result_count;
    return 0;
}

int me_gr_tangent_pointwise(
    const size_t n,
    const double* positions,
    const double* masses,
    const double* delta_positions,
    const double gravitational_constant,
    const double coefficient_scale,
    const double c_m_per_s,
    const int include_central_response,
    double* accelerations,
    double* tangent
) {
    const double central_mass = n > 0u ? masses[0] : 0.0;
    double mu2_over_c2;
    size_t i;
    if (n == 0u || positions == NULL || masses == NULL || accelerations == NULL ||
        !isfinite(gravitational_constant) || !isfinite(coefficient_scale) ||
        !isfinite(c_m_per_s) || c_m_per_s <= 0.0 || central_mass == 0.0) {
        return -1;
    }
    for (i = 0u; i < 3u * n; ++i) {
        accelerations[i] = 0.0;
        if (tangent != NULL) {
            tangent[i] = 0.0;
        }
    }
    mu2_over_c2 = coefficient_scale *
        (gravitational_constant * central_mass) *
        (gravitational_constant * central_mass) /
        (c_m_per_s * c_m_per_s);
    for (i = 1u; i < n; ++i) {
        const size_t o = 3u * i;
        const double dx = positions[o] - positions[0];
        const double dy = positions[o + 1u] - positions[1];
        const double dz = positions[o + 2u] - positions[2];
        const double ddx = delta_positions != NULL ? delta_positions[o] - delta_positions[0] : 0.0;
        const double ddy = delta_positions != NULL ? delta_positions[o + 1u] - delta_positions[1] : 0.0;
        const double ddz = delta_positions != NULL ? delta_positions[o + 2u] - delta_positions[2] : 0.0;
        double ax;
        double ay;
        double az;
        double tax;
        double tay;
        double taz;
        me_pair_terms(
            dx, dy, dz, ddx, ddy, ddz, mu2_over_c2,
            &ax, &ay, &az, &tax, &tay, &taz
        );
        accelerations[o] += ax;
        accelerations[o + 1u] += ay;
        accelerations[o + 2u] += az;
        if (include_central_response != 0) {
            const double mass_ratio = masses[i] / central_mass;
            accelerations[0] -= mass_ratio * ax;
            accelerations[1] -= mass_ratio * ay;
            accelerations[2] -= mass_ratio * az;
        }
        if (tangent != NULL) {
            tangent[o] += tax;
            tangent[o + 1u] += tay;
            tangent[o + 2u] += taz;
            if (include_central_response != 0) {
                const double mass_ratio = masses[i] / central_mass;
                tangent[0] -= mass_ratio * tax;
                tangent[1] -= mass_ratio * tay;
                tangent[2] -= mass_ratio * taz;
            }
        }
    }
    return 0;
}

uint32_t me_gr_tangent_api_version(void) {
    return ME_GR_TANGENT_API_VERSION;
}

size_t me_gr_tangent_sizeof_simulation(void) {
    return sizeof(struct reb_simulation);
}

size_t me_gr_tangent_sizeof_particle(void) {
    return sizeof(struct reb_particle);
}

size_t me_gr_tangent_offsetof_additional_forces(void) {
    return offsetof(struct reb_simulation, additional_forces);
}

size_t me_gr_tangent_offsetof_extras(void) {
    return offsetof(struct reb_simulation, extras);
}

uintptr_t me_gr_tangent_callback_address(void) {
    return (uintptr_t)(void (*)(struct reb_simulation* const))me_gr_tangent_additional_forces;
}
