import numpy as np
import math
from typing import Tuple, Optional
from rfmodel.core.units import db_to_linear, linear_to_db, w_to_dbm, dbm_to_w

'''
System Level Equations
'''

def eirp(
    *,
    pt_dbm: float,
    gt_db: float
) -> Tuple[float, float]:
    """
    Effective Isotropic Radiated Power (EIRP)

        EIRP = Pt * Gt

    Parameters
    ----------
    pt_dbm : float
        Transmit power in dBm
    gt_db : float
        Transmit antenna gain in dB

    Returns
    -------
    (eirp_w, eirp_dbm)

    Usage
    -----
    eirp_w, eirp_dbm = eirp(pt_dbm=10.0, gt_db=12.0)
    """
    pt_w = dbm_to_w(pt_dbm)
    gt_lin = db_to_linear(gt_db)

    eirp_w = pt_w * gt_lin
    eirp_dbm = w_to_dbm(eirp_w)
    return eirp_w, eirp_dbm


def friis_received_power(
    pt_w: float,
    gt_lin: float,
    gr_lin: float,
    wavelength_m: float,
    range_m: float
) -> float:
    """
    Friis free-space received power (ideal, no losses)

        Pr = (Gt * Gr * λ² / (4πR)²) * Pt

    Returns
    -------
    Pr_w : float
        Received power in Watts

    Usage
    -----
    pr_w = friis_received_power(
        pt_w=0.01,
        gt_lin=10.0,
        gr_lin=6.3,
        wavelength_m=0.125,
        range_m=1000.0
    )
    """
    fs_factor = (wavelength_m / (4 * math.pi * range_m)) ** 2
    return pt_w * gt_lin * gr_lin * fs_factor


def free_space_path_loss_db(range_m: float, wavelength_m: float) -> float:
    """
    Free-space path loss (FSPL)

        L0 = 20 log10(4πR / λ)

    Returns
    -------
    L0_db : float
        Positive loss term in dB

    Usage
    -----
    L0_db = free_space_path_loss_db(range_m=1000.0, wavelength_m=0.125)
    """
    return 20 * math.log10(4 * math.pi * range_m / wavelength_m)


def link_budget_pr_dbm(
    pt_dbm: float,
    gt_db: float,
    gr_db: float,
    *,
    lt_db: float = 0.0,
    lr_db: float = 0.0,
    l0_db: float = 0.0,
    la_db: float = 0.0
) -> float:
    """
    Received power from link budget

        Pr = Pt - Lt + Gt - L0 - LA + Gr - Lr

    All loss terms are positive numbers.

    Returns
    -------
    Pr_dbm : float

    Usage
    -----
    pr_dbm = link_budget_pr_dbm(
        pt_dbm=10.0,
        gt_db=12.0,
        gr_db=6.0,
        lt_db=1.0,
        lr_db=1.0,
        l0_db=100.0,
        la_db=0.5
    )
    """
    return pt_dbm - lt_db + gt_db - l0_db - la_db + gr_db - lr_db


def impedance_mismatch_loss_db(gamma_mag: float) -> float:
    """
    Impedance mismatch loss

        Limp = -10 log10(1 - |Γ|²)

    Returns
    -------
    Limp_db : float

    Usage
    -----
    Limp_db = impedance_mismatch_loss_db(gamma_mag=0.2)
    """
    if gamma_mag < 0 or gamma_mag >= 1:
        raise ValueError("|Gamma| must be in [0, 1)")
    return -10 * math.log10(1 - gamma_mag ** 2)


def link_margin_db(pr_dbm: float, pr_min_dbm: float) -> float:
    """
    Link margin

        LM = Pr - Pr(min)

    Returns
    -------
    LM_db : float

    Usage
    -----
    lm_db = link_margin_db(pr_dbm=-85.0, pr_min_dbm=-95.0)
    """
    return pr_dbm - pr_min_dbm
