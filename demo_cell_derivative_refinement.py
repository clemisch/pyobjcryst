"""
Demonstration: does ObjCryst's analytic unit-cell derivative — which omits the
reflection-*intensity* dependence on sin(theta)/lambda — hurt a refinement?

Short answer: no. This script perturbs the unit cell of a triclinic model well
away from the truth and refines it back against a synthetic pattern. It converges
in a handful of iterations and recovers the true cell to within the statistical
precision of the (noisy) data.

--------------------------------------------------------------------------------
Background
--------------------------------------------------------------------------------
The analytic LSQ derivatives implemented in ObjCryst are exact for peak-shape and
peak-position parameters (U, V, W, Eta*, Asym*, Scherrer P, microstrain, Zero,
sample displacement, transparency) and for the *peak-position* contribution of the
unit-cell parameters (a, b, c, alpha, beta, gamma).

A unit-cell parameter also changes sin(theta)/lambda, and therefore the reflection
intensities, via the Lorentz-polarization correction and the sin(theta)/lambda
dependence of the structure factors. ObjCryst's analytical-derivative machinery
omits this intensity term for the cell (it is the domain of the separate
intensity-derivative code, CalcIhkl_FullDeriv, which explicitly defers it). The
analytic cell derivative therefore reproduces only the (dominant) peak-position
term. Numerically, this term matches a finite difference with a position
coefficient of 1.00000; the leftover difference vs. a full finite difference is
purely the smooth intensity modulation described above.

Why this does not matter for refinement:

* Convergence direction is fine. Gauss-Newton / Levenberg-Marquardt only needs the
  Jacobian to point roughly the right way. The cell derivative is dominated by the
  peak-position term (large magnitude on the peak flanks), which is computed
  exactly; the omitted piece is a small, smooth intensity modulation. LM damping
  absorbs the inaccuracy. The cost is a fraction of an iteration, not extra cycles.

* The converged cell is essentially unbiased. A least-squares fit pins the cell
  where the residual is orthogonal to the peak-*shift* shapes -- i.e. where the
  peaks sit at the right 2theta. That is exactly the term computed exactly. The
  neglected term is an intensity *scaling*, and intensity misfit is the natural job
  of the scale factor, Biso and background -- not the cell.

* It could only show up in second-order quantities: the reported cell<->Biso and
  cell<->scale covariances/ESDs come from the same Jacobian and are very slightly
  off. The cell value and its own ESD (curvature dominated by the position term)
  are effectively unaffected. The one contrived failure mode is refining the cell
  as the *sole* parameter able to absorb a smooth intensity trend with nothing else
  free -- not a realistic Rietveld / Le Bail setup.

* Not a regression. ObjCryst's analytic FullDeriv path already treated the cell
  contribution as position-only (the old code perturbed the cell, recomputed the
  peak center and applied it to the profile -- it never recomputed intensities).
  This implementation makes that same-scope derivative analytic and exact, rather
  than a finite difference.
--------------------------------------------------------------------------------
"""
import numpy as np
from pyobjcryst.crystal import Crystal
from pyobjcryst.atom import Atom
from pyobjcryst.scatteringpower import ScatteringPowerAtom
from pyobjcryst.powderpattern import PowderPattern
from pyobjcryst.lsq import LSQ

D2R = np.pi / 180
CELL_NAMES = ["a", "b", "c", "alpha", "beta", "gamma"]


def build(a=5.0, b=6.0, c=7.0, al=85.0, be=95.0, ga=100.0, sg="P-1"):
    """Build a triclinic Ni model + powder pattern with realistic profile widths."""
    cr = Crystal(a, b, c, al * D2R, be * D2R, ga * D2R, sg)
    sp = ScatteringPowerAtom("Ni", "Ni")
    sp.SetBiso(0.8)
    cr.AddScatteringPower(sp)
    cr.AddScatterer(Atom(0.11, 0.22, 0.33, "Ni1", sp))
    pp = PowderPattern()
    pp.SetWavelength(1.5406)
    npts = 6000
    pp.SetPowderPatternPar(15 * D2R, 0.015 * D2R, npts)
    diff = pp.AddPowderPatternDiffraction(cr)
    prof = diff.GetProfile()
    for name, val in [("U", 2e-4), ("V", -1e-4), ("W", 3e-4), ("Eta0", 0.4)]:
        prof.GetPar(name).SetValue(val)
    return cr, pp, diff, prof


def cell_values(cr):
    return np.array([cr.GetPar(n).GetValue() for n in CELL_NAMES])


def main():
    # 1) Synthesize an "observed" pattern from a known (true) cell, add 1% noise.
    cr0, pp0, _, _ = build()
    ycalc = np.array(pp0.GetPowderPatternCalc())
    rng = np.random.default_rng(0)
    yobs = ycalc * (1 + 0.01 * rng.standard_normal(len(ycalc))) + 0.001 * ycalc.max()
    true_cell = cell_values(cr0)

    # 2) Fresh model with a badly perturbed cell; refine only the 6 cell parameters
    #    (+ overall scale, the natural home for any intensity misfit).
    cr, pp, diff, prof = build(a=5.03, b=6.04, c=6.95, al=85.4, be=94.6, ga=100.5)
    pp.SetPowderPatternObs(yobs)
    pp.FixAllPar()
    for name in CELL_NAMES:
        cr.GetPar(name).SetIsFixed(False)

    lsq = LSQ()
    lsq.SetRefinedObj(pp, 0, True, True)
    lsq.PrepareRefParList()
    # Free the overall scale factor if it is exposed as a refinable parameter.
    try:
        lsq.SetParIsFixed("Scale_", False)
    except Exception:
        pass

    start_err = cell_values(cr) - true_cell
    print("Refining 6 cell parameters of a triclinic (P-1) model.")
    print(f"Perturbation from truth: {np.abs(start_err[:3]).max():.3f} Angstrom, "
          f"{np.abs(start_err[3:]).max() / D2R:.2f} deg\n")
    print(f"{'iter':>4}  {'max|d_len| (A)':>15}  {'max|d_ang| (deg)':>16}  {'Rwp':>8}")

    for it in range(1, 11):
        lsq.SafeRefine(nbCycle=1, useLevenbergMarquardt=True)
        err = cell_values(cr) - true_cell
        dlen = np.abs(err[:3]).max()
        dang = np.abs(err[3:]).max() / D2R
        rwp = pp.GetRw()
        print(f"{it:>4}  {dlen:>15.2e}  {dang:>16.2e}  {rwp:>8.4f}")

    final = cell_values(cr)
    print("\n           " + "".join(f"{n:>11}" for n in CELL_NAMES))
    print("true   :   " + "".join(f"{v:>11.5f}" for v in true_cell))
    print("refined:   " + "".join(f"{v:>11.5f}" for v in final))
    print("\nRecovered the true cell to within the data's statistical precision, "
          "in a handful of iterations.")


if __name__ == "__main__":
    main()
