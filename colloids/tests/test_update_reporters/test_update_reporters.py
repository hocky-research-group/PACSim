import os
import pytest
from openmm import unit
from colloids.colloids_run import colloids_run
import numpy as np

class TestUpdateReporters(object):
    @pytest.fixture(autouse=True)
    def change_test_dir(self, request, monkeypatch):
        # Change the working directory to the directory of the test file.
        # See https://stackoverflow.com/questions/62044541/change-pytest-working-directory-to-test-case-directory
        monkeypatch.chdir(request.fspath.dirname)

    @pytest.fixture(autouse=True)
    def tear_down(self):
        yield
        assert os.path.isfile("final_frame.gsd")
        assert os.path.isfile("final_frame.xyz")
        assert os.path.isfile("state_data.csv")
        assert os.path.isfile("trajectory.gsd")
        assert os.path.isfile("update_reporter.csv")

        os.remove("final_frame.gsd")
        os.remove("final_frame.xyz")
        os.remove("state_data.csv")
        os.remove("trajectory.gsd")
        os.remove("update_reporter.csv")

    @pytest.mark.parametrize("yaml_file,expected_parameter_values",
                             [("debye_linearmonotonic.yaml", [5.1,
                                                            5.199999999999999,
                                                            5.299999999999999,
                                                            5.399999999999999,
                                                            5.499999999999998,
                                                            5.599999999999998,
                                                            5.6999999999999975,
                                                            5.799999999999997,
                                                            5.899999999999997,
                                                            5.9999999999999964]),
                              ("debye_linearunimodal.yaml", [5.333333333333333,
                                                            5.666666666666666,
                                                            5.999999999999999,
                                                            5.857142857142856,
                                                            5.714285714285713,
                                                            5.571428571428569,
                                                            5.428571428571426,
                                                            5.285714285714283,
                                                            5.14285714285714,
                                                            4.9999999999999964]),
                              ("debye_sinusoidal.yaml", [0.0,
                                                        2.7201055544468495,
                                                        4.564726253638138,
                                                        4.940158120464309,
                                                        3.725565802396744,
                                                        1.3118742685196438,
                                                        1.5240531055110833,
                                                        3.8694534077894454,
                                                        4.969443269616876,
                                                        4.4699833180027895])])
    def test_parameter_values(self, yaml_file, expected_parameter_values):
        simulation = colloids_run([yaml_file])
        f= np.loadtxt('update_reporter.csv', delimiter=",", dtype=float, unpack=True, skiprows=1)
        actual_parameter_values = f[1]
        assert actual_parameter_values == pytest.approx(expected_parameter_values,rel=1.0e-12, abs=1.0e-12)


if __name__ == '__main__':
    pytest.main([__file__])

