"""Hess-Smith panel method for 2D inviscid, incompressible flow over an airfoil.

Each panel carries a constant-strength source (its own value), and every panel
carries the same constant-strength vortex. That gives N + 1 unknowns:
  - N equations: no flow through each panel (flow tangency at control points)
  - 1 equation:  the Kutta condition (flow leaves the trailing edge smoothly,
                 so tangential velocities on the two trailing-edge panels match)

Once solved, we get the surface velocity, the pressure coefficient Cp, and the
lift coefficient Cl (both from the circulation and by integrating Cp, as a check).

Reference: Katz & Plotkin, Low-Speed Aerodynamics, ch. 11; Hess & Smith (1967).
"""

import numpy as np


def solve(x, y, alpha_deg):
    """Solve the flow around the closed airfoil (x, y) at angle of attack alpha.

    Returns a dict with control points, Cp, Cl (circulation), Cl (pressure).
    """
    alpha = np.radians(alpha_deg)
    n = len(x) - 1

    # Panel geometry
    xc = 0.5 * (x[:-1] + x[1:])          # control points (panel midpoints)
    yc = 0.5 * (y[:-1] + y[1:])
    dx, dy = np.diff(x), np.diff(y)
    length = np.hypot(dx, dy)
    theta = np.arctan2(dy, dx)            # panel angle

    # Influence of panel j (unit source / unit vortex) on control point i,
    # written in panel j's local frame then rotated into panel i's normal/tangent.
    # r_ij  = distances from control point i to the start/end of panel j
    # beta  = angle subtended by panel j at control point i
    xi, yi = xc[:, None], yc[:, None]
    r1 = np.hypot(xi - x[None, :-1], yi - y[None, :-1])
    r2 = np.hypot(xi - x[None, 1:], yi - y[None, 1:])
    beta =(np.arctan2(yi - y[None, 1:], xi - x[None, 1:])
            - np.arctan2(yi - y[None, :-1], xi - x[None, :-1]))
    beta = (beta + np.pi) % (2 * np.pi) - np.pi   # wrap to (-pi, pi]
    np.fill_diagonal(beta, np.pi)                  # self-influence
    log_r = np.log(r2 / r1)
    np.fill_diagonal(log_r, 0.0)

    dth = theta[:, None] - theta[None, :]          # theta_i - theta_j
    s, c = np.sin(dth), np.cos(dth)

    # Normal and tangential velocity at i induced by unit sources on j
    An = (s * log_r + c * beta) / (2 * np.pi)
    At = (s * beta - c * log_r) / (2 * np.pi)
    # A unit vortex on j induces the source pattern rotated by 90 degrees
    Bn = -At
    Bt = An

    # Build the (N+1) x (N+1) linear system
    A = np.zeros((n + 1, n + 1))
    b = np.zeros(n + 1)
    A[:n, :n] = An
    A[:n, n] = Bn.sum(axis=1)
    b[:n] = -np.sin(alpha - theta)                 # freestream normal component

    # Kutta condition: Vt on first panel + Vt on last panel = 0
    A[n, :n] = At[0] + At[-1]
    A[n, n] = Bt[0].sum() + Bt[-1].sum()
    b[n] = -np.cos(alpha - theta[0]) - np.cos(alpha - theta[-1])

    sol = np.linalg.solve(A, b)
    sigma, gamma = sol[:n], sol[n]

    vt = np.cos(alpha - theta) + At @ sigma + gamma * Bt.sum(axis=1)
    cp = 1 - vt**2

    # Lift two ways: Kutta-Joukowski (circulation) and integrating pressure
    cl_gamma = 2 * gamma * length.sum()
    fx = -(cp * length * -np.sin(theta)).sum()     # force from -Cp * n * ds
    fy = -(cp * length * np.cos(theta)).sum()
    cl_pressure = fy * np.cos(alpha) - fx * np.sin(alpha)

    return {"xc": xc, "yc": yc, "cp": cp, "vt": vt,
            "cl": cl_gamma, "cl_pressure": cl_pressure,
            "residual_tangency": np.abs(A[:n] @ sol - b[:n]).max()}
