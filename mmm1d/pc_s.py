"""
Vectorized translation of pc_s.m: numerical inversion of the capillary
pressure -> saturation relationship
p_c(s) = -0.00011*exp(-44.02*(s-0.496)) + 278.3*exp(8.103*(s-0.496)) - 191.8

Given a capillary pressure p_c, returns the liquid water saturation s
consistent with this constitutive curve, by tabulating p_c(s) in steps
of 0.001 between s_im and 0.9 and locating where the curve crosses
p_c_in.
"""
import numpy as np


def pc_s(pc_in, s_im, strict=False):
    """Inverts the GDL capillary pressure curve to obtain the saturation.

    Parameters
    ----------
    pc_in : array_like
        Local capillary pressure, p_c = P_liq - P_gas [Pa].
    s_im : float
        Immobile liquid water saturation (floor of s).
    strict : bool, default False
        If False (default), this **exactly** reproduces the behavior
        of the original reference MMM1D.m -- including a quirk in the
        line ``if pc_in<min(pc)`` in MATLAB, which compares the
        **entire vector** ``pc_in`` (not the element ``pc_in(i)``)
        against ``min(pc)``. In MATLAB, an ``if`` over an array is
        only true when ALL elements satisfy the condition; since
        ``min(pc)`` is a fairly negative number, this makes the
        condition false for the whole vector almost all the time, so
        the function returns ``s_im`` for EVERY point, regardless of
        the actual value of ``pc_in(i)`` -- the "search" for the
        crossing point is rarely executed. This may be a bug in the
        original code (the expected behavior would be to compare
        ``pc_in[i] < min(pc)`` element-wise) or it may be intentional
        (keeping the liquid phase close to immobile over most of the
        domain). It is worth checking this against the reference paper
        before relying on the results.
        If True, use the element-wise comparison (``pc_in[i]``), which
        is probably the original intent of the code.

    Returns
    -------
    s_in : np.ndarray
        Liquid water saturation, same shape as ``pc_in``.
    """
    pc_in = np.atleast_1d(np.asarray(pc_in, dtype=float))
    s = np.arange(s_im, 0.9 + 1e-12, 0.001)
    pc = -0.00011 * np.exp(-44.02 * (s - 0.496)) + 278.3 * np.exp(8.103 * (s - 0.496)) - 191.8
    pc_min = pc.min()

    s_in = np.empty_like(pc_in)
    # "global" condition as in the original MATLAB: if pc_in < min(pc)
    # (true only if ALL elements of pc_in satisfy it)
    all_below = bool(np.all(pc_in < pc_min))

    for i in range(pc_in.size):
        cond = (pc_in[i] < pc_min) if strict else all_below
        if cond:
            # find(pc >= pc_in(i), 1) -> first (1-based) index with pc >= pc_in(i)
            idx = np.argmax(pc >= pc_in[i])
            if not (pc >= pc_in[i]).any():
                idx = pc.size  # find() would return empty; limiting behavior
            l = max(1, idx)  # MATLAB: max(2, idx_1based) - 1 in 0-based -> max(1, idx_0based)
            l = min(l, s.size - 1)
            s_in[i] = max(s_im, (s[l] + s[l - 1]) / 2.0)
        else:
            s_in[i] = s_im

    return s_in
