"""NACA 4-digit airfoil geometry.

Generates the surface coordinates of a NACA 4-digit section (e.g. "0012", "2412")
using cosine spacing, which clusters points near the leading and trailing edges
where the pressure changes fastest.
"""

import numpy as np


def naca4(code, n_panels=160):
    """Return (x, y) surface points for a NACA 4-digit airfoil with unit chord.

    Points run clockwise: trailing edge -> lower surface -> leading edge ->
    upper surface -> trailing edge. The first and last points coincide, so
    there are n_panels + 1 points and n_panels panels.
    """
    if n_panels % 2:
        raise ValueError("n_panels must be even")

    m = int(code[0]) / 100  # maximum camber
    p = int(code[1]) / 10   # position of maximum camber
    t = int(code[2:]) / 100  # maximum thickness

    # Cosine spacing from leading edge (0) to trailing edge (1)
    beta = np.linspace(0, np.pi, n_panels // 2 + 1)
    x = 0.5 * (1 - np.cos(beta))

    # Thickness distribution (closed trailing edge: last coefficient -0.1036)
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2
                  + 0.2843 * x**3 - 0.1036 * x**4)

    # Camber line and its slope
    yc = np.zeros_like(x)
    dyc = np.zeros_like(x)
    if m > 0:
        front = x < p
        back = ~front
        yc[front] = m / p**2 * (2 * p * x[front] - x[front]**2)
        dyc[front] = 2 * m / p**2 * (p - x[front])
        yc[back] = m / (1 - p)**2 * ((1 - 2 * p) + 2 * p * x[back] - x[back]**2)
        dyc[back] = 2 * m / (1 - p)**2 * (p - x[back])

    theta = np.arctan(dyc)
    xu, yu = x - yt * np.sin(theta), yc + yt * np.cos(theta)
    xl, yl = x + yt * np.sin(theta), yc - yt * np.cos(theta)

    # Lower surface TE->LE, then upper surface LE->TE (skip duplicate LE point)
    xs = np.concatenate([xl[::-1], xu[1:]])
    ys = np.concatenate([yl[::-1], yu[1:]])
    return xs, ys


def zero_lift_angle(code):
    """Thin airfoil theory zero-lift angle (degrees) for a NACA 4-digit camber line."""
    m = int(code[0]) / 100
    p = int(code[1]) / 10
    if m == 0:
        return 0.0
    # alpha_L0 = -(1/pi) * integral_0^pi dyc/dx * (cos(theta) - 1) dtheta
    th = np.linspace(0, np.pi, 20001)
    x = 0.5 * (1 - np.cos(th))
    dyc = np.where(x < p, 2 * m / p**2 * (p - x), 2 * m / (1 - p)**2 * (p - x))
    integrand = dyc * (np.cos(th) - 1)
    alpha_l0 = -np.trapezoid(integrand, th) / np.pi
    return np.degrees(alpha_l0)
