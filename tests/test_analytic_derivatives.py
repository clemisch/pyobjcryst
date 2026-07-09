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


def _lsq_deriv(pp, par):
    """Analytic LSQ derivative of the pattern w.r.t. ``par``.

    GetLSQDeriv does not compute a derivative for a *fixed* parameter (neither
    does the base-class numerical derivative it falls back to: RefinablePar.Mutate
    is a no-op on a fixed parameter). A real least-squares only ever asks for the
    derivatives of the parameters it has freed, so free the parameter for the
    computation -- then restore its original fixed state so this helper has no
    lasting side effect on the model.
    """
    was_fixed = par.IsFixed()
    par.SetIsFixed(False)
    try:
        return np.array(pp.GetLSQDeriv(0, par))
    finally:
        par.SetIsFixed(was_fixed)


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
                da = _lsq_deriv(pp, par)
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
            da = _lsq_deriv(pp, par)
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
            da = _lsq_deriv(pp, par)
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
            da = _lsq_deriv(pp, par)
            n = len(da)
            if np.abs(da).max() == 0:
                # Parameter is constrained by the space-group symmetry (even when
                # freed); its analytic derivative must be exactly zero.
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
        self.assertGreater(np.abs(_lsq_deriv(pp, cr.GetPar("a"))).max(), 0)
        # b, c and the angles are constrained by the cubic symmetry: their
        # derivative must be exactly zero even when the parameter is freed.
        for name in ["b", "c", "alpha", "beta", "gamma"]:
            d = _lsq_deriv(pp, cr.GetPar(name))
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


# --- Every-parameter sweep --------------------------------------------------
#
# The flat-detector bug (a parameter whose analytic derivative was silently 0,
# so the least-squares deactivated it) slipped through because the earlier tests
# only checked a hand-picked list of parameters -- the same list I had in mind
# when writing the implementation, so the test shared the implementation's blind
# spot. This test instead enumerates *every* refinable parameter of a
# representative model and checks each one, so any parameter whose derivative is
# silently wrong or zero is caught automatically -- independent of my
# assumptions about which parameters matter.
#
# GetLSQDeriv uses the analytic derivative only for the parameters implemented
# here (profile shape, unit cell, peak-position corrections) and falls back to a
# numerical derivative for everything else (structure factor: atom positions,
# Biso, occupancy; absorption; scale factor; ...). Both must agree with a finite
# difference of the pattern; the key invariant is that no parameter which
# actually affects the pattern may have an identically-zero analytic derivative.

# Sensible, in-range, non-limit values for every parameter of the model built in
# _build_full_model(). Values are kept away from parameter limits so the central
# finite difference is not clamped.
_SWEEP_VALUES = {
    # NB: the six unit-cell parameters are intentionally NOT set here - they are
    # already given sensible values (in radians for the angles) by
    # _build_full_model(). Setting an angle here via SetValue() would interpret
    # e.g. 85 as 85 radians and produce a degenerate cell.
    # atom (occupancy kept below its upper limit of 1)
    "x": 0.11, "y": 0.22, "z": 0.33, "Ni1occup": 0.8,
    # scattering power
    "Biso": 0.5, "ML Error": 0.2, "ML-Nb Ghost Atoms": 1.0, "Formal Charge": 0.0,
    # powder-pattern corrections & scale. The flat-detector dispersion ratios
    # are left at 0: their derivative there is still non-zero (so the sweep still
    # guards against the "silently deactivated" regression), while a non-zero
    # value would shift every peak and add finite-difference noise to the sharp
    # profile derivatives. Non-zero flat-det values are validated separately in
    # test_flatdet_derivative_at_nonzero_value.
    "Zero": 0.002, "2ThetaDispl": 0.001, "2ThetaTransp": 0.0005,
    "MuR": 0.3, "Scale_": 1.0,
    # diffraction phase
    "Global Biso": 0.3,
    # profile
    "U": 5e-5, "V": -3e-5, "W": 1.2e-4, "P": 1e-5, "MicrostrainPPM": 300.0,
    "Eta0": 0.4, "Eta1": 0.05, "Asym0": 1.2, "Asym1": 0.1, "Asym2": 0.02,
    # DIFC/DIFA are TOF-only; for the monochromatic model here they have no effect
    "DIFC": 48277.0, "DIFA": -6.7,
}


def _build_full_model():
    """Build a triclinic model exercising every kind of refinable parameter."""
    cr = Crystal(5.0, 6.0, 7.0, 85 * D2R, 95 * D2R, 100 * D2R, "P-1")
    sp = ScatteringPowerAtom("Ni", "Ni")
    sp.SetBiso(0.5)
    cr.AddScatteringPower(sp)
    atom = Atom(0.11, 0.22, 0.33, "Ni1", sp)
    cr.AddScatterer(atom)
    pp = PowderPattern()
    pp.SetWavelength(1.0)
    npts = 4000
    pp.SetPowderPatternPar(14 * D2R, 0.02 * D2R, npts)
    diff = pp.AddPowderPatternDiffraction(cr)
    prof = diff.GetProfile()
    pp.SetPowderPatternObs(np.ones(npts))
    return cr, sp, atom, pp, diff, prof


class TestAllParametersDerivative(_ObjCrystTestCase):
    """Enumerate every refinable parameter and check its LSQ derivative."""

    # Coarse tolerance: this test is a "nothing is silently broken" guard (it
    # must catch a zero or a factor-of-2 error); the tight (1e-8) validation of
    # the in-scope parameters lives in the other test classes.
    TOL = 5e-2

    def test_every_parameter_derivative(self):
        cr, sp, atom, pp, diff, prof = _build_full_model()

        # Set every parameter to its sensible value and free it, so that it is
        # both exercised and mutatable for the finite difference.
        objects = [cr, sp, atom, pp, diff, prof]
        for obj in objects:
            for i in range(obj.GetNbPar()):
                par = obj.GetPar(i)
                if not par.IsUsed():
                    continue
                if par.GetName() in _SWEEP_VALUES:
                    par.SetValue(_SWEEP_VALUES[par.GetName()])
                par.SetIsFixed(False)
        pp.GetPowderPatternCalc()
        y = np.array(pp.GetPowderPatternCalc())

        seen = set()
        n_checked = 0
        for obj in objects:
            is_cell = obj is cr  # the crystal's own parameters are the 6 cell ones
            for i in range(obj.GetNbPar()):
                par = obj.GetPar(i)
                name = par.GetName()
                if name in seen or not par.IsUsed():
                    continue
                seen.add(name)
                with self.subTest(parameter=name):
                    self._check_one(pp, par, name, y, is_cell)
                    n_checked += 1
        # Guard against the model silently losing parameters (which would make
        # this sweep vacuous).
        self.assertGreaterEqual(n_checked, 25)

    def _check_one(self, pp, par, name, y, is_cell):
        v0 = par.GetValue()
        # Use a tuned finite-difference step where we have one (profile widths in
        # particular have tiny values, for which a relative step would be pure
        # rounding noise); otherwise a relative step is fine.
        step = STEP.get(name, max(abs(v0) * 1e-5, 1e-7))
        da = _lsq_deriv(pp, par)
        n = len(da)
        dn1 = _ndiff(pp, par, step)[:n]
        dn2 = _ndiff(pp, par, step * 4)[:n]
        m = _robust_mask(dn1, dn2)
        scale_n = np.abs(dn1[m]).max() if m.any() else 0.0
        scale_a = np.abs(da).max()

        if scale_n == 0.0:
            # The parameter has no effect on the pattern in this configuration
            # (e.g. TOF DIFC/DIFA under monochromatic radiation, or Formal
            # Charge): the analytic derivative must then also be zero.
            self.assertEqual(
                scale_a, 0.0,
                msg=f"{name}: analytic derivative is non-zero ({scale_a:.2e}) "
                    f"but the parameter does not affect the pattern",
            )
            return

        # The parameter affects the pattern -> its derivative must not be a
        # silent zero. This is exactly the failure the flat-detector bug was.
        self.assertGreater(
            scale_a, 0.0,
            msg=f"{name}: analytic derivative is identically zero while the "
                f"parameter affects the pattern -> it would be deactivated "
                f"during refinement",
        )

        rel = np.abs(da - dn1)[m].max() / scale_n
        if rel < self.TOL:
            return
        # Unit-cell parameters: the analytic derivative deliberately keeps only
        # the peak-position term, so it can differ from a full finite difference
        # by the (smooth) intensity-correction term. Check the position
        # coefficient instead (see TestAnalyticCellDerivatives).
        if is_cell:
            coef, *_ = np.linalg.lstsq(
                np.vstack([da[m], y[:n][m]]).T, dn1[m], rcond=None
            )
            self.assertLess(
                abs(coef[0] - 1.0), self.TOL,
                msg=f"{name}: position coefficient {coef[0]:.4f} deviates from 1",
            )
            return
        self.fail(f"{name}: rel.err={rel:.2e} exceeds {self.TOL:.0e}")


if __name__ == "__main__":
    unittest.main()
