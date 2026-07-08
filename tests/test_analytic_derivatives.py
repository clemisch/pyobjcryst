#!/usr/bin/env python
##############################################################################
#
# pyobjcryst
#
# See AUTHORS.txt for a list of people who contributed.
# See LICENSE.txt for license information.
#
##############################################################################
"""Tests for the analytic LSQ derivatives used in powder-pattern refinement.

ObjCryst computes analytic derivatives of the calculated powder pattern with
respect to:

  * peak-shape parameters  : U, V, W, Eta0, Eta1, Asym0, Asym1, Asym2
  * Scherrer broadening    : P
  * microstrain broadening : MicrostrainPPM
  * position corrections   : Zero, 2ThetaDispl, 2ThetaTransp,
                             2ThetaFlatDetDispRatio (+ per-phase variant)
  * unit cell              : a, b, c, alpha, beta, gamma

Peak-shape and position-correction parameters do not change sin(theta)/lambda,
so their analytic derivatives match a full finite difference of the pattern
essentially to machine precision. Unit-cell parameters also change
sin(theta)/lambda, hence the reflection intensities (via the
Lorentz-polarization correction and the structure factors). ObjCryst's
analytical-derivative machinery -- like most Le Bail / variable-projection
schemes, where the intensities are treated as linear parameters -- includes only
the peak-*position* contribution for the cell. The cell tests below therefore
decompose the numeric derivative into a position component (coefficient of the
analytic derivative, expected == 1) and an intensity component (proportional to
the pattern), and check the position coefficient is 1.

These tests mirror the standalone scripts ``validate_analytic_derivatives.py``
and ``demo_cell_derivative_refinement.py`` in the package root.
"""

import gc
import unittest

import numpy as np

from pyobjcryst.atom import Atom
from pyobjcryst.crystal import Crystal
from pyobjcryst.lsq import LSQ
from pyobjcryst.powderpattern import PowderPattern
from pyobjcryst.scatteringpower import ScatteringPowerAtom

D2R = np.pi / 180

PROFILE_PARS = [
    "U", "V", "W", "P", "MicrostrainPPM",
    "Eta0", "Eta1", "Asym0", "Asym1", "Asym2",
]
CORR_PARS = ["Zero", "2ThetaDispl", "2ThetaTransp", "2ThetaFlatDetDispRatio"]
PHASE_CORR_PARS = ["2ThetaFlatDetDispRatioPhase"]  # live on the diffraction phase
CELL_PARS = ["a", "b", "c", "alpha", "beta", "gamma"]

# Finite-difference steps, chosen small enough that the central difference is
# accurate but large enough to stay well above rounding noise.
STEP = {
    "U": 1e-8, "V": 1e-8, "W": 1e-8, "P": 1e-9, "MicrostrainPPM": 1e-2,
    "Eta0": 1e-5, "Eta1": 1e-5, "Asym0": 1e-5, "Asym1": 1e-6, "Asym2": 1e-6,
    "a": 1e-6, "b": 1e-6, "c": 1e-6, "alpha": 1e-6, "beta": 1e-6, "gamma": 1e-6,
    "Zero": 1e-6, "2ThetaDispl": 1e-6, "2ThetaTransp": 1e-6,
    "2ThetaFlatDetDispRatio": 1e-6, "2ThetaFlatDetDispRatioPhase": 1e-6,
}

# (label, space group, (a, b, c, alpha, beta, gamma), wavelength)
CASES = [
    ("cubic", "Fm-3m", (4.05, 4.05, 4.05, 90, 90, 90), 1.5406),
    ("cubic_alpha12", "Fm-3m", (4.05, 4.05, 4.05, 90, 90, 90), "Cu"),
    ("tetragonal", "I4/mmm", (4.0, 4.0, 6.0, 90, 90, 90), 1.5406),
    ("hexagonal", "P63/mmc", (3.2, 3.2, 5.2, 90, 90, 120), 1.5406),
    ("orthorhombic", "Pnma", (5.0, 6.0, 7.0, 90, 90, 90), 1.5406),
    ("monoclinic", "P21/c", (5.0, 6.0, 7.0, 90, 100, 90), 1.5406),
    ("triclinic", "P-1", (5.0, 6.0, 7.0, 85, 95, 100), 1.5406),
    ("rhombohedral", "R-3m:R", (5.0, 5.0, 5.0, 87, 87, 87), 1.5406),
]

# Relative-error tolerances (comfortably above the observed ~1e-3 worst case for
# overlapping low-symmetry peaks, well below a real error which would be O(1)).
TOL_DIRECT = 1e-2  # profile-shape & position-correction params vs finite diff
TOL_CELL = 2e-2    # deviation of the cell position coefficient from 1


def _build(sg, cell, wavelength):
    """Build a Ni crystal + powder pattern with non-trivial profile parameters."""
    a, b, c, al, be, ga = cell
    cr = Crystal(a, b, c, al * D2R, be * D2R, ga * D2R, sg)
    sp = ScatteringPowerAtom("Ni", "Ni")
    sp.SetBiso(0.5)
    cr.AddScatteringPower(sp)
    cr.AddScatterer(Atom(0.11, 0.22, 0.33, "Ni1", sp))

    pp = PowderPattern()
    pp.SetWavelength(wavelength)
    npts = 4000
    pp.SetPowderPatternPar(15 * D2R, 0.02 * D2R, npts)
    diff = pp.AddPowderPatternDiffraction(cr)
    prof = diff.GetProfile()
    for name, val in [
        ("U", 0.5e-4), ("V", -0.3e-4), ("W", 1.2e-4), ("P", 1e-5),
        ("MicrostrainPPM", 300.0), ("Eta0", 0.4), ("Eta1", 0.05),
        ("Asym0", 1.2), ("Asym1", 0.1), ("Asym2", 0.02),
    ]:
        prof.GetPar(name).SetValue(val)
    pp.GetPar("Zero").SetValue(0.002)
    pp.GetPar("2ThetaDispl").SetValue(0.001)
    pp.GetPar("2ThetaTransp").SetValue(0.0005)
    # Note: the flat-detector dispersion ratios are left at 0 here so that the
    # peak positions stay put and the finite-difference reference for the other
    # parameters is clean. At 0 their derivative is still non-zero (it is
    # sin(2*theta)/2), so they are still exercised by the loop below; a non-zero
    # value is checked separately in test_flatdet_derivative_at_nonzero_value.
    pp.SetPowderPatternObs(np.ones(npts))
    return cr, pp, diff, prof


def _ndiff(pp, par, step):
    """Central finite difference of the whole pattern w.r.t. one parameter."""
    v = par.GetValue()
    par.SetValue(v + step)
    yp = np.array(pp.GetPowderPatternCalc())
    par.SetValue(v - step)
    ym = np.array(pp.GetPowderPatternCalc())
    par.SetValue(v)
    pp.GetPowderPatternCalc()
    return (yp - ym) / (2 * step)


def _robust_mask(dn1, dn2):
    """Points where two step sizes agree.

    A reflection profile is stored on a finite window; when a peak shifts, a
    pixel can enter or leave that window, producing a discontinuity that
    corrupts the finite difference at a handful of points. Such points disagree
    between two step sizes and are excluded from the comparison.
    """
    return np.abs(dn1 - dn2) < 1e-3 * np.abs(dn1).max()


class _ObjCrystTestCase(unittest.TestCase):
    """Base class that forces a garbage collection after each test.

    pyobjcryst objects (Crystal, PowderPattern, ...) reference each other and
    register in ObjCryst's global registries; if they are left to be collected
    lazily they may be freed during an unrelated later test, which has been
    observed to segfault. Collecting them here, while nothing else is live,
    keeps the objects created by these tests contained to these tests.
    """

    def tearDown(self):
        gc.collect()


class TestAnalyticProfileDerivatives(_ObjCrystTestCase):
    """Analytic derivatives vs. finite differences of the calculated pattern."""

    def _check_direct(self, label, sg, cell, wavelength):
        cr, pp, diff, prof = _build(sg, cell, wavelength)
        for obj, names in ((prof, PROFILE_PARS), (pp, CORR_PARS),
                           (diff, PHASE_CORR_PARS)):
            for name in names:
                par = obj.GetPar(name)
                da = np.array(pp.GetLSQDeriv(0, par))
                n = len(da)
                dn1 = _ndiff(pp, par, STEP[name])[:n]
                dn2 = _ndiff(pp, par, STEP[name] * 4)[:n]
                m = _robust_mask(dn1, dn2)
                scale = np.abs(dn1[m]).max()
                self.assertGreater(
                    scale, 0.0,
                    msg=f"[{label}] {name}: parameter has no effect on pattern",
                )
                err = np.abs(da - dn1)[m].max() / scale
                self.assertLess(
                    err, TOL_DIRECT,
                    msg=f"[{label}] {name}: rel.err={err:.2e} exceeds "
                        f"{TOL_DIRECT:.0e}",
                )

    def test_profile_and_position_correction_derivatives(self):
        """U,V,W,P,microstrain,Eta*,Asym*,Zero,displacement,transparency,flat-det."""
        for label, sg, cell, wl in CASES:
            with self.subTest(case=label):
                self._check_direct(label, sg, cell, wl)

    def test_flatdet_dispersion_ratio_derivative_nonzero_at_zero(self):
        """Regression: the flat-detector dispersion ratio derivative must be
        non-zero even when the parameter is 0 (its usual starting value).

        Otherwise the refinement sees a zero-derivative column and immediately
        deactivates the parameter (the analytic derivative acts through the
        atan() flat-detector term of X2XCorrPhase, not through X2XCorr).
        """
        cr, pp, diff, prof = _build("Fm-3m", (4.05, 4.05, 4.05, 90, 90, 90), 1.5406)
        for obj, name in ((pp, "2ThetaFlatDetDispRatio"),
                          (diff, "2ThetaFlatDetDispRatioPhase")):
            par = obj.GetPar(name)
            par.SetValue(0.0)
            pp.GetPowderPatternCalc()
            da = np.array(pp.GetLSQDeriv(0, par))
            self.assertGreater(
                np.abs(da).max(), 0.0,
                msg=f"{name}: derivative is identically zero at value 0 -> the "
                    f"parameter would be deactivated during refinement",
            )

    def test_flatdet_derivative_at_nonzero_value(self):
        """Flat-detector dispersion ratio derivative at a realistic non-zero value.

        Uses a slightly looser tolerance than the other direct comparisons: a
        non-zero ratio shifts every peak, so a few finite-difference points near
        profile-window boundaries remain even after the two-step mask.
        """
        cr, pp, diff, prof = _build("P-1", (5.0, 6.0, 7.0, 85, 95, 100), 1.5406)
        pp.GetPar("2ThetaFlatDetDispRatio").SetValue(0.004)
        diff.GetPar("2ThetaFlatDetDispRatioPhase").SetValue(0.003)
        pp.GetPowderPatternCalc()
        for obj, name in ((pp, "2ThetaFlatDetDispRatio"),
                          (diff, "2ThetaFlatDetDispRatioPhase")):
            par = obj.GetPar(name)
            da = np.array(pp.GetLSQDeriv(0, par))
            n = len(da)
            dn1 = _ndiff(pp, par, STEP[name])[:n]
            dn2 = _ndiff(pp, par, STEP[name] * 4)[:n]
            m = _robust_mask(dn1, dn2)
            err = np.abs(da - dn1)[m].max() / np.abs(dn1[m]).max()
            self.assertLess(err, 2e-2, msg=f"{name}: rel.err={err:.2e}")


class TestAnalyticCellDerivatives(_ObjCrystTestCase):
    """The unit-cell derivative reproduces the peak-position term exactly."""

    def _check_cell(self, label, sg, cell, wavelength):
        cr, pp, diff, prof = _build(sg, cell, wavelength)
        y = np.array(pp.GetPowderPatternCalc())
        checked_any = False
        for name in CELL_PARS:
            par = cr.GetPar(name)
            da = np.array(pp.GetLSQDeriv(0, par))
            n = len(da)
            if np.abs(da).max() == 0:
                # Parameter is constrained (fixed) by the space-group symmetry;
                # analytic derivative must be exactly zero -- which it is here.
                continue
            checked_any = True
            dn1 = _ndiff(pp, par, STEP[name])[:n]
            dn2 = _ndiff(pp, par, STEP[name] * 4)[:n]
            m = _robust_mask(dn1, dn2)
            # dn ~= A * da (position term) + c * y (intensity-correction term)
            basis = np.vstack([da[m], y[:n][m]]).T
            coef, *_ = np.linalg.lstsq(basis, dn1[m], rcond=None)
            perr = abs(coef[0] - 1.0)
            self.assertLess(
                perr, TOL_CELL,
                msg=f"[{label}] {name}: position coeff={coef[0]:.5f} "
                    f"deviates from 1 by {perr:.2e}",
            )
        self.assertTrue(
            checked_any, msg=f"[{label}] no free cell parameters were tested"
        )

    def test_cell_position_coefficient_is_one(self):
        """a,b,c,alpha,beta,gamma across all crystal systems."""
        for label, sg, cell, wl in CASES:
            with self.subTest(case=label):
                self._check_cell(label, sg, cell, wl)

    def test_constrained_cell_parameters_have_zero_derivative(self):
        """Symmetry-fixed cell parameters must have an exactly-zero derivative."""
        # Cubic: only 'a' is free; b, c and all angles are constrained.
        cr, pp, diff, prof = _build("Fm-3m", (4.05, 4.05, 4.05, 90, 90, 90), 1.5406)
        pp.GetPowderPatternCalc()
        self.assertGreater(np.abs(np.array(pp.GetLSQDeriv(0, cr.GetPar("a")))).max(), 0)
        for name in ["b", "c", "alpha", "beta", "gamma"]:
            d = np.array(pp.GetLSQDeriv(0, cr.GetPar(name)))
            self.assertEqual(
                np.abs(d).max(), 0.0,
                msg=f"constrained parameter {name} has a non-zero derivative",
            )


class TestCellRefinementWithAnalyticDerivatives(_ObjCrystTestCase):
    """A perturbed cell is recovered by LSQ using the analytic derivatives."""

    def test_triclinic_cell_recovery(self):
        # 1) Synthesize a noisy "observed" pattern from a known triclinic cell.
        cr0, pp0, _, _ = _build("P-1", (5.0, 6.0, 7.0, 85, 95, 100), 1.5406)
        # sharper, more realistic peaks so the cell is well determined
        prof0 = pp0.GetPowderPatternComponent(0).GetProfile()
        for name, val in [("U", 2e-4), ("V", -1e-4), ("W", 3e-4)]:
            prof0.GetPar(name).SetValue(val)
        ycalc = np.array(pp0.GetPowderPatternCalc())
        rng = np.random.default_rng(0)
        yobs = ycalc * (1 + 0.01 * rng.standard_normal(len(ycalc)))
        yobs += 0.001 * ycalc.max()
        true_cell = np.array([cr0.GetPar(n).GetValue() for n in CELL_PARS])

        # 2) Fresh model with a badly perturbed cell; refine only the cell.
        cr, pp, diff, prof = _build(
            "P-1", (5.03, 6.04, 6.95, 85.4, 94.6, 100.5), 1.5406
        )
        for name, val in [("U", 2e-4), ("V", -1e-4), ("W", 3e-4)]:
            prof.GetPar(name).SetValue(val)
        pp.SetPowderPatternObs(yobs)
        pp.FixAllPar()
        for name in CELL_PARS:
            cr.GetPar(name).SetIsFixed(False)

        lsq = LSQ()
        lsq.SetRefinedObj(pp, 0, True, True)
        lsq.PrepareRefParList()

        rw_start = pp.GetRw()
        lsq.SafeRefine(nbCycle=8, useLevenbergMarquardt=True)
        rw_end = pp.GetRw()
        final_cell = np.array([cr.GetPar(n).GetValue() for n in CELL_PARS])

        # The fit must have improved and the cell must be recovered to well
        # within the perturbation (down to the statistical precision of the
        # noisy data).
        self.assertLess(rw_end, rw_start)
        len_err = np.abs(final_cell[:3] - true_cell[:3]).max()
        ang_err = np.abs(final_cell[3:] - true_cell[3:]).max()
        self.assertLess(len_err, 1e-2, msg=f"cell length error {len_err:.2e} A")
        self.assertLess(ang_err, 1e-2, msg=f"cell angle error {ang_err:.2e} rad")


if __name__ == "__main__":
    unittest.main()
