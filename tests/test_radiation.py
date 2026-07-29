#!/usr/bin/env python
##############################################################################
#
# File coded by:    Vincent Favre-Nicolin
#
# See AUTHORS.txt for a list of people who contributed.
# See LICENSE_DANSE.txt for license information.
#
##############################################################################
"""Tests for Radiation module."""

import unittest

from pyobjcryst.radiation import Radiation, RadiationType, WavelengthType


class TestRadiation(unittest.TestCase):

    def testRadiation(self):
        """Test Radiation creation."""
        Radiation()
        return

    def testWavelength(self):
        """Test setting & reading wavelength."""
        r = Radiation()
        r.SetWavelength(1.24)
        self.assertAlmostEqual(r.GetWavelength(), 1.24, places=3)
        return

    def testType(self):
        """Test setting & reading X-ray Tube wavelength."""
        r = Radiation()
        r.SetWavelengthType(WavelengthType.WAVELENGTH_ALPHA12)
        self.assertEqual(
            r.GetWavelengthType(), WavelengthType.WAVELENGTH_ALPHA12
        )
        r.SetRadiationType(RadiationType.RAD_NEUTRON)
        self.assertEqual(r.GetRadiationType(), RadiationType.RAD_NEUTRON)
        r.SetWavelength("Cu")
        self.assertAlmostEqual(r.GetWavelength(), 1.5418, places=4)
        self.assertEqual(
            r.GetWavelengthType(), WavelengthType.WAVELENGTH_ALPHA12
        )
        return

    def testLinearPolarRate(self):
        """Test getting and setting linear polarisation rate."""
        r = Radiation()
        # Default for X-ray should be 0
        self.assertAlmostEqual(r.GetLinearPolarRate(), 0.0)
        r.SetLinearPolarRate(0.95)
        self.assertAlmostEqual(r.GetLinearPolarRate(), 0.95)
        r.SetLinearPolarRate(0.0)
        self.assertAlmostEqual(r.GetLinearPolarRate(), 0.0)
        return


if __name__ == "__main__":
    unittest.main()
