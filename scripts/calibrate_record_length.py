"""How long a record does the estimator actually need?

Every argument of the last two days -- 400 Myr or 800 Myr, windowed slope or
two-term fit, what tolerance -- is the same question asked without measuring
it. This measures it.

Method: a system in the same dynamical class as Pluto (a massless body in a
giant planet's resonance-overlap zone) but with a Lyapunov time of thousands of
years instead of millions. Integrate it for many hundreds of Lyapunov times so
lambda is known by convergence, then truncate the record to 10, 20, 40 ...
Lyapunov times and ask each estimator what it would have said.

The answer is a number: the record length, in Lyapunov times, at which each
estimator is right to within 10 percent. Pluto's 800 Myr is then just a point
on that curve.
"""
from __future__ import annotations
import json, math, sys, time
import numpy as np, rebound
sys.path.insert(0, "/home/claude/eph2/mini_ephemeris/src")

def build(dt=0.05, a_tp=2.95):
    sim = rebound.Simulation(); sim.units = ("yr","AU","Msun")
    sim.integrator = "whfast"; sim.dt = dt
    sim.add(m=1.0)
    sim.add(m=9.5458e-4, a=5.2027, e=0.0484, f=0.0)
    sim.add(m=0.0, a=a_tp, e=0.15, f=1.1, omega=0.4)
    sim.move_to_com(); sim.init_megno(seed=4242)
    return sim

def logn(sim):
    tot = 0.0
    for p in sim.particles[sim.N_real:]:
        tot += p.x*p.x+p.y*p.y+p.z*p.z+p.vx*p.vx+p.vy*p.vy+p.vz*p.vz
    return 0.5*math.log(tot)

years, n = 3.0e6, 6000
sim = build(); e0 = sim.energy(); l0 = logn(sim)
times = np.linspace(years/n, years, n)
S = np.empty(n); Y = np.empty(n); drift = 0.0
t0 = time.time()
for i, t in enumerate(times):
    sim.integrate(t, exact_finish_time=0)
    S[i] = logn(sim) - l0
    Y[i] = sim.megno()
    drift = max(drift, abs((sim.energy()-e0)/e0))
np.savetxt("fast_chaos_record.csv",
           np.column_stack([times, S, Y]), delimiter=",",
           header="time_years,cumulative_log_growth,mean_megno", comments="")
print(f"integrated {years:.1e} yr in {(time.time()-t0)/60:.1f} min, dE/E = {drift:.2e}")
print(f"S(T) = {S[-1]:.2f},  <Y>(T) = {Y[-1]:.2f}")
half = n//2
truth = float(np.polyfit(times[half:], S[half:], 1)[0])
print(f"ground-truth lambda (LS slope, second half) = {truth:.6e} /yr")
print(f"  -> Lyapunov time {1/truth:.1f} yr;  record = {years*truth:.0f} Lyapunov times")
