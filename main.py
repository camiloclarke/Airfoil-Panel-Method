"""Run the panel method, validate it, and save the plots.

1. Pressure distribution on a NACA 2412 at 5 degrees
2. Lift curves for NACA 0012 and 2412 vs thin airfoil theory
3. Panel-count convergence study (how Cl error falls as panels are added)
"""

import numpy as np
import matplotlib.pyplot as plt

from naca import naca4, zero_lift_angle
from panel import solve

plt.rcParams.update({"figure.dpi": 120, "font.size": 10})


def cp_plot(code="2412", alpha=5.0, n=160):
    x, y = naca4(code, n)
    r = solve(x, y, alpha)
    half = n // 2
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(r["xc"][half:], r["cp"][half:], label="Upper surface")
    ax1.plot(r["xc"][:half], r["cp"][:half], label="Lower surface")
    ax1.invert_yaxis()
    ax1.set_ylabel("Pressure coefficient $C_p$")
    ax1.set_title(f"NACA {code}, α = {alpha:g}°  —  $C_l$ = {r['cl']:.3f}")
    ax1.grid(alpha=0.3)
    ax1.legend()
    ax2.fill(x, y, color="0.75")
    ax2.set_aspect("equal")
    ax2.set_xlabel("x / c")
    ax2.set_yticks([])
    fig.tight_layout()
    fig.savefig("cp_distribution.png")
    plt.close(fig)


def lift_curve(codes=("0012", "2412"), n=160):
    alphas = np.arange(-4, 10.5, 1.0)
    fig, ax = plt.subplots(figsize=(7, 5))
    print("\nLift-curve slope (per radian), fitted over -4 to 10 degrees")
    print(f"{'Airfoil':<10}{'Panel method':>14}{'Thin theory':>13}{'Thickness-corrected':>22}")
    for code in codes:
        x, y = naca4(code, n)
        cl = np.array([solve(x, y, a)["cl"] for a in alphas])
        a0 = zero_lift_angle(code)
        t = int(code[2:]) / 100
        thin = 2 * np.pi * np.radians(alphas - a0)
        slope = np.polyfit(np.radians(alphas), cl, 1)[0]
        corrected = 2 * np.pi * (1 + 0.77 * t)
        print(f"NACA {code:<5}{slope:>14.3f}{2 * np.pi:>13.3f}{corrected:>22.3f}")
        line, = ax.plot(alphas, cl, "o-", ms=4, label=f"NACA {code} — panel method")
        ax.plot(alphas, thin, "--", color=line.get_color(),
                label=f"NACA {code} — thin airfoil theory")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.axvline(0, color="0.6", lw=0.8)
    ax.set_xlabel("Angle of attack α (degrees)")
    ax.set_ylabel("Lift coefficient $C_l$")
    ax.set_title("Lift curves: panel method vs thin airfoil theory")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("lift_curve.png")
    plt.close(fig)


def convergence(code="2412", alpha=5.0):
    ns = np.array([20, 40, 80, 160, 320, 640])
    ref = solve(*naca4(code, 2560), alpha)["cl"]
    cls = np.array([solve(*naca4(code, n), alpha)["cl"] for n in ns])
    err = np.abs(cls - ref)
    # Very coarse grids (N < 80) are pre-asymptotic: the error is not yet
    # monotonic, so the order is fitted over the regular part only.
    fit = ns >= 80
    order = np.polyfit(np.log(ns[fit]), np.log(err[fit]), 1)[0]
    print(f"\nConvergence (NACA {code}, α = {alpha:g}°, reference N = 2560, Cl = {ref:.5f})")
    for n, c, e in zip(ns, cls, err):
        print(f"  N = {n:4d}   Cl = {c:.5f}   error = {e:.2e}")
    print(f"  Observed order of convergence (N >= 80): {-order:.2f}")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.loglog(ns, err, "o-", label="Panel method")
    k = np.argmax(fit)
    ax.loglog(ns[fit], err[k] * (ns[k] / ns[fit]), "--", color="0.5",
              label="Slope −1 (first order)")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlabel("Number of panels N")
    ax.set_ylabel("|Cl − Cl(N = 2560)|")
    ax.set_title(f"Convergence with panel count (order ≈ {-order:.2f})")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig("convergence.png")
    plt.close(fig)


def consistency_check(code="2412", alpha=5.0, n=160):
    r = solve(*naca4(code, n), alpha)
    diff = abs(r["cl"] - r["cl_pressure"]) / abs(r["cl"]) * 100
    print(f"\nTwo ways to get lift (NACA {code}, α = {alpha:g}°, N = {n})")
    print(f"  From circulation (Kutta-Joukowski): {r['cl']:.4f}")
    print(f"  From integrating pressure:          {r['cl_pressure']:.4f}")
    print(f"  Difference: {diff:.2f}%")


if __name__ == "__main__":
    print(f"Thin airfoil theory zero-lift angle, NACA 2412: {zero_lift_angle('2412'):.2f}°")
    cp_plot()
    lift_curve()
    convergence()
    consistency_check()
    print("\nSaved: cp_distribution.png, lift_curve.png, convergence.png")
