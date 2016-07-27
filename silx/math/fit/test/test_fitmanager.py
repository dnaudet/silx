# coding: utf-8
# /*##########################################################################
# Copyright (C) 2016 European Synchrotron Radiation Facility
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ############################################################################*/
"""
Tests for fitmanager module
"""

import unittest
import numpy
import os
import os.path

from silx.math.fit import fitmanager
from silx.math.fit import fittheories
from silx.math.fit.fittheory import FitTheory
from silx.math.fit.functions import sum_gauss, sum_stepdown, sum_stepup

from silx.testutils import temp_dir

custom_function_definition = """
import copy
from silx.math.fit.fittheory import FitTheory

CONFIG = {'d': 1.}

def myfun(x, a, b, c):
    "Model function"
    return (a * x**2 + b * x + c) / CONFIG['d']

def myesti(x, y, bg, yscaling):
    "Initial parameters for iterative fit (a, b, c) = (1, 1, 1)"
    return (1., 1., 1.), ((0, 0, 0), (0, 0, 0), (0, 0, 0))

def myconfig(d=1.):
    "This function cam modify CONFIG"
    CONFIG["d"] = d
    return CONFIG

def myderiv(x, parameters, index):
    "Custom derivative (does not work, causes singular matrix)"
    pars_plus = copy.copy(parameters)
    pars_plus[index] *= 1.0001

    pars_minus = parameters
    pars_minus[index] *= copy.copy(0.9999)

    delta_fun = myfun(x, *pars_plus) - myfun(x, *pars_minus)
    delta_par = parameters[index] * 0.0001 * 2

    return delta_fun / delta_par

THEORY = {
    'my fit theory':
        FitTheory(function=myfun,
                  parameters=('A', 'B', 'C'),
                  estimate=myesti,
                  configure=myconfig,
                  derivative=myderiv)
}

"""

old_custom_function_definition = """
CONFIG = {'d': 4.5}

def myfun(x, a, b, c):
    "Model function"
    return (a * x**2 + b * x + c) / CONFIG['d']

def myesti(x, y, bg, yscaling):
    "Initial parameters for iterative fit (a, b, c) = (1, 1, 1)"
    return (1., 1., 1.), ((0, 0, 0), (0, 0, 0), (0, 0, 0))

THEORY = ['my fit theory']
PARAMETERS = [('A', 'B', 'C')]
FUNCTION = [myfun]
ESTIMATE = [myesti]

"""


class TestFitmanager(unittest.TestCase):
    """
    Unit tests of multi-peak functions.
    """
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testFitManager(self):
        """Test fit manager on synthetic data using a gaussian function
        and a linear background"""
        # Create synthetic data with a sum of gaussian functions
        x = numpy.arange(1000).astype(numpy.float)

        p = [1000, 100., 250,
             255, 700., 45,
             1500, 800.5, 95]
        linear_bg = 2.65 * x + 13
        y = linear_bg + sum_gauss(x, *p)

        # Fitting
        fit = fitmanager.FitManager()
        fit.setdata(x=x, y=y)
        fit.loadtheories(fittheories)
        # Use one of the default fit functions
        fit.settheory('Gaussians')
        fit.setbackground('Linear')
        fit.estimate()
        fit.startfit()

        # first 2 parameters are related to the linear background
        self.assertEqual(fit.fit_results[0]["name"], "Constant")
        self.assertAlmostEqual(fit.fit_results[0]["fitresult"], 13)
        self.assertEqual(fit.fit_results[1]["name"], "Slope")
        self.assertAlmostEqual(fit.fit_results[1]["fitresult"], 2.65)

        for i, param in enumerate(fit.fit_results[2:]):
            param_number = i // 3 + 1
            if i % 3 == 0:
                self.assertEqual(param["name"],
                                 "Height%d" % param_number)
            elif i % 3 == 1:
                self.assertEqual(param["name"],
                                 "Position%d" % param_number)
            elif i % 3 == 2:
                self.assertEqual(param["name"],
                                 "FWHM%d" % param_number)
            self.assertAlmostEqual(param["fitresult"],
                                   p[i])

    def testLoadCustomFitFunction(self):
        """Test FitManager using a custom fit function defined in an external
        file and imported with FitManager.loadtheories"""
        # Create synthetic data with a sum of gaussian functions
        x = numpy.arange(100).astype(numpy.float)

        # a, b, c are the fit parameters
        # d is a known scaling parameter that is set using configure()
        a, b, c, d = 1.5, 2.5, 3.5, 4.5
        y = (a * x**2 + b * x + c) / d

        # Fitting
        fit = fitmanager.FitManager()
        fit.setdata(x=x, y=y)

        # Create a temporary function definition file, and import it
        with temp_dir() as tmpDir:
            tmpfile = os.path.join(tmpDir, 'customfun.py')
            # custom_function_definition
            fd = open(tmpfile, "w")
            fd.write(custom_function_definition)
            fd.close()
            fit.loadtheories(tmpfile)
            os.unlink(tmpfile)

        fit.settheory('my fit theory')
        # Test configure
        fit.configure(d=4.5)
        fit.estimate()
        fit.startfit()

        self.assertEqual(fit.fit_results[0]["name"],
                         "A1")
        self.assertAlmostEqual(fit.fit_results[0]["fitresult"],
                               1.5)
        self.assertEqual(fit.fit_results[1]["name"],
                         "B1")
        self.assertAlmostEqual(fit.fit_results[1]["fitresult"],
                               2.5)
        self.assertEqual(fit.fit_results[2]["name"],
                         "C1")
        self.assertAlmostEqual(fit.fit_results[2]["fitresult"],
                               3.5)

    def testLoadOldCustomFitFunction(self):
        """Test FitManager using a custom fit function defined in an external
        file and imported with FitManager.loadtheories (legacy PyMca format)"""
        # Create synthetic data with a sum of gaussian functions
        x = numpy.arange(100).astype(numpy.float)

        # a, b, c are the fit parameters
        # d is a known scaling parameter that is set using configure()
        a, b, c, d = 1.5, 2.5, 3.5, 4.5
        y = (a * x**2 + b * x + c) / d

        # Fitting
        fit = fitmanager.FitManager()
        fit.setdata(x=x, y=y)

        # Create a temporary function definition file, and import it
        with temp_dir() as tmpDir:
            tmpfile = os.path.join(tmpDir, 'oldcustomfun.py')
            # custom_function_definition
            fd = open(tmpfile, "w")
            fd.write(old_custom_function_definition)
            fd.close()
            fit.loadtheories(tmpfile)
            os.unlink(tmpfile)

        fit.settheory('my fit theory')
        # Test configure
        fit.estimate()
        fit.startfit()

        self.assertEqual(fit.fit_results[0]["name"],
                         "A1")
        self.assertAlmostEqual(fit.fit_results[0]["fitresult"],
                               1.5)
        self.assertEqual(fit.fit_results[1]["name"],
                         "B1")
        self.assertAlmostEqual(fit.fit_results[1]["fitresult"],
                               2.5)
        self.assertEqual(fit.fit_results[2]["name"],
                         "C1")
        self.assertAlmostEqual(fit.fit_results[2]["fitresult"],
                               3.5)

    def testAddTheory(self):
        """Test FitManager using a custom fit function imported with
        FitManager.addtheory"""
        # Create synthetic data with a sum of gaussian functions
        x = numpy.arange(100).astype(numpy.float)

        # a, b, c are the fit parameters
        # d is a known scaling parameter that is set using configure()
        a, b, c, d = -3.14, 1234.5, 10000, 4.5
        y = (a * x**2 + b * x + c) / d

        # Fitting
        fit = fitmanager.FitManager()
        fit.setdata(x=x, y=y)

        # Define and add the fit theory
        CONFIG = {'d': 1.}

        def myfun(x_, a_, b_, c_):
            """"Model function"""
            return (a_ * x_**2 + b_ * x_ + c_) / CONFIG['d']

        def myesti(x_, y_, bg, yscaling):
            """"Initial parameters for iterative fit:
                 (a, b, c) = (1, 1, 1)
            Constraints all set to 0 (FREE)"""
            return (1., 1., 1.), ((0, 0, 0), (0, 0, 0), (0, 0, 0))

        def myconfig(d_=1.):
            """This function can modify CONFIG"""
            CONFIG["d"] = d_
            return CONFIG

        def myderiv(x_, parameters, index):
            """Custom derivative"""
            pars_plus = numpy.array(parameters, copy=True)
            pars_plus[index] *= 1.001

            pars_minus = numpy.array(parameters, copy=True)
            pars_minus[index] *= 0.999

            delta_fun = myfun(x_, *pars_plus) - myfun(x_, *pars_minus)
            delta_par = parameters[index] * 0.001 * 2

            return delta_fun / delta_par

        fit.addtheory("polynomial",
                      FitTheory(function=myfun,
                                parameters=["A", "B", "C"],
                                estimate=myesti,
                                configure=myconfig,
                                derivative=myderiv))

        fit.settheory('polynomial')
        fit.configure(d_=4.5)
        fit.estimate()
        params1, sigmas, infodict = fit.startfit()

        self.assertEqual(fit.fit_results[0]["name"],
                         "A1")
        self.assertAlmostEqual(fit.fit_results[0]["fitresult"],
                               -3.14)
        self.assertEqual(fit.fit_results[1]["name"],
                         "B1")
        # params1[1] is the same as fit.fit_results[1]["fitresult"]
        self.assertAlmostEqual(params1[1],
                               1234.5)
        self.assertEqual(fit.fit_results[2]["name"],
                         "C1")
        self.assertAlmostEqual(params1[2],
                               10000)

        # change configuration scaling factor and check that the fit returns
        # different values
        fit.configure(d_=5.)
        fit.estimate()
        params2, sigmas, infodict = fit.startfit()
        for p1, p2 in zip(params1, params2):
            self.assertFalse(numpy.array_equal(p1, p2),
                             "Fit parameters are equal even though the " +
                             "configuration has been changed")

    def testStep(self):
        """Test fit manager on a step function with a more complex estimate
        function than the gaussian (convolution filter)"""
        for theory_name, theory_fun in (('Step Down', sum_stepdown),
                                        ('Step Up', sum_stepup)):
            # Create synthetic data with a sum of gaussian functions
            x = numpy.arange(1000).astype(numpy.float)

            # ('Height', 'Position', 'FWHM')
            p = [1000, 439, 250]

            linear_bg = 2.65 * x + 13
            y = theory_fun(x, *p) + linear_bg

            # Fitting
            fit = fitmanager.FitManager()
            fit.setdata(x=x, y=y)
            fit.loadtheories(fittheories)
            # Use one of the default fit functions
            fit.settheory(theory_name)
            fit.setbackground('Linear')
            fit.estimate()

            params, sigmas, infodict = fit.startfit()

            # # first 2 parameters are related to the linear background
            self.assertAlmostEqual(params[0], 13, places=5)
            self.assertAlmostEqual(params[1], 2.65, places=5)

            for i, param in enumerate(params[2:]):
                self.assertAlmostEqual(param, p[i], places=5)


test_cases = (TestFitmanager,)


def suite():
    loader = unittest.defaultTestLoader
    test_suite = unittest.TestSuite()
    for test_class in test_cases:
        tests = loader.loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    return test_suite

if __name__ == '__main__':
    unittest.main(defaultTest="suite")
