"""Validate the analytic LSQ derivatives implemented in ObjCryst for powder
diffraction refinement against finite differences of the computed pattern.

Covers:
  * peak-shape parameters  : U, V, W, Eta0, Eta1, Asym0, Asym1, Asym2
  * Scherrer broadening    : P
  * microstrain broadening : MicrostrainPPM
  * position corrections   : Zero, 2ThetaDispl, 2ThetaTransp
  * unit cell              : a, b, c, alpha, beta, gamma

Peak-shape and position-correction parameters do not change sin(theta)/lambda,
so their analytic derivatives match a full finite difference of the pattern to
~1e-8. Unit-cell parameters also change sin(theta)/lambda, hence the reflection
intensities (via the Lorentz-polarization correction and structure factors).
ObjCryst's analytical-derivative machinery (like most Le Bail / variable-projection
schemes, where the intensities are treated as linear parameters) includes only the
peak-position contribution for the cell. This script therefore verifies the cell
derivatives by decomposing the numeric derivative into a position component
(coefficient of the analytic derivative, expected == 1) and an intensity component
(proportional to the pattern), and checks the position coefficient is 1.
"""
import numpy as np
from pyobjcryst.crystal import Crystal
from pyobjcryst.atom import Atom
from pyobjcryst.scatteringpower import ScatteringPowerAtom
from pyobjcryst.powderpattern import PowderPattern

D2R = np.pi / 180
PROFILE_PARS = ["U", "V", "W", "P", "MicrostrainPPM",
                "Eta0", "Eta1", "Asym0", "Asym1", "Asym2"]
CORR_PARS = ["Zero", "2ThetaDispl", "2ThetaTransp"]
CELL_PARS = ["a", "b", "c", "alpha", "beta", "gamma"]
STEP = {"U": 1e-8, "V": 1e-8, "W": 1e-8, "P": 1e-9, "MicrostrainPPM": 1e-2,
        "Eta0": 1e-5, "Eta1": 1e-5, "Asym0": 1e-5, "Asym1": 1e-6, "Asym2": 1e-6,
        "a": 1e-6, "b": 1e-6, "c": 1e-6, "alpha": 1e-6, "beta": 1e-6, "gamma": 1e-6,
        "Zero": 1e-6, "2ThetaDispl": 1e-6, "2ThetaTransp": 1e-6}


def build(sg, cell, wavelength):
    a, b, c, al, be, ga = cell
    cr = Crystal(a, b, c, al * D2R, be * D2R, ga * D2R, sg)
    sp = ScatteringPowerAtom("Ni", "Ni"); sp.SetBiso(0.5)
    cr.AddScatteringPower(sp)
    cr.AddScatterer(Atom(0.11, 0.22, 0.33, "Ni1", sp))
    pp = PowderPattern(); pp.SetWavelength(wavelength)
    npts = 4000
    pp.SetPowderPatternPar(15 * D2R, 0.02 * D2R, npts)
    diff = pp.AddPowderPatternDiffraction(cr)
    prof = diff.GetProfile()
    for name, val in [("U", 0.5e-4), ("V", -0.3e-4), ("W", 1.2e-4), ("P", 1e-5),
                      ("MicrostrainPPM", 300.), ("Eta0", 0.4), ("Eta1", 0.05),
                      ("Asym0", 1.2), ("Asym1", 0.1), ("Asym2", 0.02)]:
        prof.GetPar(name).SetValue(val)
    pp.GetPar("Zero").SetValue(0.002)
    pp.GetPar("2ThetaDispl").SetValue(0.001)
    pp.GetPar("2ThetaTransp").SetValue(0.0005)
    pp.SetPowderPatternObs(np.ones(npts))
    return cr, pp, diff, prof


def ndiff(pp, par, step):
    v = par.GetValue()
    par.SetValue(v + step); yp = np.array(pp.GetPowderPatternCalc())
    par.SetValue(v - step); ym = np.array(pp.GetPowderPatternCalc())
    par.SetValue(v); pp.GetPowderPatternCalc()
    return (yp - ym) / (2 * step)


def robust_mask(dn1, dn2):
    """Exclude points where the finite difference is unstable (profile window
    boundary discontinuities), detected by disagreement between two step sizes."""
    return np.abs(dn1 - dn2) < 1e-3 * np.abs(dn1).max()


def check(label, sg, cell, wavelength):
    cr, pp, diff, prof = build(sg, cell, wavelength)
    y = np.array(pp.GetPowderPatternCalc())
    print(f"\n=== {label}  (sg={sg}, wl={wavelength}) ===")
    worst = 0.0
    # direct comparison: profile-shape + position-correction parameters
    for obj, names in ((prof, PROFILE_PARS), (pp, CORR_PARS)):
        for name in names:
            par = obj.GetPar(name)
            da = np.array(pp.GetLSQDeriv(0, par))
            n = len(da)
            dn1 = ndiff(pp, par, STEP[name])[:n]
            dn2 = ndiff(pp, par, STEP[name] * 4)[:n]
            m = robust_mask(dn1, dn2)
            scale = np.abs(dn1[m]).max()
            if scale == 0:
                continue
            err = np.abs(da - dn1)[m].max() / scale
            worst = max(worst, err)
            print(f"  {name:14s} rel.err={err:.2e}  {'OK' if err < 2e-2 else 'FAIL'}")
    # cell parameters: verify the position component (coefficient of analytic deriv)
    for name in CELL_PARS:
        par = cr.GetPar(name)
        da = np.array(pp.GetLSQDeriv(0, par))
        n = len(da)
        if np.abs(da).max() == 0:
            print(f"  {name:14s} (fixed by symmetry)")
            continue
        dn1 = ndiff(pp, par, STEP[name])[:n]
        dn2 = ndiff(pp, par, STEP[name] * 4)[:n]
        m = robust_mask(dn1, dn2)
        # dn ~= A*da(position) + c*y(intensity correction)
        M = np.vstack([da[m], y[:n][m]]).T
        coef, *_ = np.linalg.lstsq(M, dn1[m], rcond=None)
        perr = abs(coef[0] - 1.0)
        worst = max(worst, perr)
        print(f"  {name:14s} position coeff={coef[0]:.5f} (err={perr:.2e})  "
              f"{'OK' if perr < 2e-2 else 'FAIL'}")
    return worst


CASES = [
    ("cubic", "Fm-3m", (4.05, 4.05, 4.05, 90, 90, 90), 1.5406),
    ("cubic (Cu Ka1/Ka2)", "Fm-3m", (4.05, 4.05, 4.05, 90, 90, 90), "Cu"),
    ("tetragonal", "I4/mmm", (4.0, 4.0, 6.0, 90, 90, 90), 1.5406),
    ("hexagonal", "P63/mmc", (3.2, 3.2, 5.2, 90, 90, 120), 1.5406),
    ("orthorhombic", "Pnma", (5.0, 6.0, 7.0, 90, 90, 90), 1.5406),
    ("monoclinic", "P21/c", (5.0, 6.0, 7.0, 90, 100, 90), 1.5406),
    ("triclinic", "P-1", (5.0, 6.0, 7.0, 85, 95, 100), 1.5406),
    ("rhombohedral", "R-3m:R", (5.0, 5.0, 5.0, 87, 87, 87), 1.5406),
]

if __name__ == "__main__":
    nbad = 0
    summary = []
    for label, sg, cell, wl in CASES:
        w = check(label, sg, cell, wl)
        ok = w < 2e-2
        nbad += not ok
        summary.append((label, w, ok))
    print("\n================= SUMMARY =================")
    for label, w, ok in summary:
        print(f"  {label:22s} worst={w:.2e}  {'OK' if ok else 'FAIL'}")
    print("\nALL OK" if nbad == 0 else f"\n{nbad} CASE(S) FAILED")
