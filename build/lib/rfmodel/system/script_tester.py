import math
from rfmodel.core import db_to_linear, dbm_to_w, w_to_dbm
from rfmodel.system import (
    eirp,
    friis_received_power,
    free_space_path_loss_db,
    link_budget_pr_dbm,
    impedance_mismatch_loss_db,
    link_margin_db,
)

### --- Self testing the equations --- ### 

def test_eirp():
    print("Testing EIRP...")
    eirp_w, eirp_dbm = eirp(pt_dbm=10.0, gt_db=12.0)

    assert math.isclose(eirp_dbm, 22.0, abs_tol=1e-6)
    assert math.isclose(w_to_dbm(eirp_w), eirp_dbm, abs_tol=1e-6)
    print("  ✓ EIRP OK")


def test_friis_vs_link_budget():
    print("Testing Friis vs link budget consistency...")

    # Parameters
    pt_dbm = 10.0
    gt_db = 12.0
    gr_db = 6.0
    R = 1000.0          # meters
    f = 2.4e9           # Hz
    c = 3e8
    lam = c / f

    # Friis (linear)
    pt_w = dbm_to_w(pt_dbm)
    gt_lin = db_to_linear(gt_db)
    gr_lin = db_to_linear(gr_db)

    pr_friis_w = friis_received_power(
        pt_w=pt_w,
        gt_lin=gt_lin,
        gr_lin=gr_lin,
        wavelength_m=lam,
        range_m=R,
    )
    pr_friis_dbm = w_to_dbm(pr_friis_w)

    # Link budget
    L0_db = free_space_path_loss_db(R, lam)
    pr_lb_dbm = link_budget_pr_dbm(
        pt_dbm=pt_dbm,
        gt_db=gt_db,
        gr_db=gr_db,
        l0_db=L0_db,
    )

    assert math.isclose(pr_friis_dbm, pr_lb_dbm, abs_tol=1e-6)
    print("  ✓ Friis and link budget match")


def test_impedance_mismatch():
    print("Testing impedance mismatch loss...")
    limp = impedance_mismatch_loss_db(gamma_mag=0.2)

    assert limp > 0
    print(f"  ✓ Limp = {limp:.2f} dB")


def test_link_margin():
    print("Testing link margin...")
    lm = link_margin_db(pr_dbm=-85.0, pr_min_dbm=-95.0)

    assert lm == 10.0
    print("  ✓ Link margin OK")


if __name__ == "__main__":
    print("Running link budget tests...\n")

    test_eirp()
    test_friis_vs_link_budget()
    test_impedance_mismatch()
    test_link_margin()

    print("\nAll tests passed ✔")
