import unittest
from app.asde.capital_curve import CapitalCurve

class TCapitalCurve(unittest.TestCase):
    def test_draw_curve_demo(self):
        CapitalCurve.draw_curve_demo()
        self.assertTrue(True)
