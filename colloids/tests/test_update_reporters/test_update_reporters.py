import os
import pytest
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
        pass
        #yield
        '''assert os.path.isfile("final_frame.gsd")
        assert os.path.isfile("final_frame.xyz")
        assert os.path.isfile("state_data.csv")
        assert os.path.isfile("trajectory.gsd")
        assert os.path.isfile("update_reporter.csv")

        os.remove("final_frame.gsd")
        os.remove("final_frame.xyz")
        os.remove("state_data.csv")
        os.remove("trajectory.gsd")
        os.remove("update_reporter.csv")'''

    @pytest.mark.parametrize("yaml_file,expected_parameter_values",
                             [("debye_ramp.yaml", [[5.0, 5.1,
                                                    5.199999999999999,
                                                    5.299999999999999,
                                                    5.399999999999999,
                                                    5.499999999999998,
                                                    5.599999999999998,
                                                    5.6999999999999975,
                                                    5.799999999999997,
                                                    5.899999999999997]]),
                              ("debye_triangle.yaml", [5.0, 5.333333333333333,
                                                    5.666666666666666,
                                                    5.333333333333333,
                                                    5.0,
                                                    4.666666666666667,
                                                    5.0,
                                                    5.333333333333333,
                                                    5.666666666666666,
                                                    5.333333333333333]),
                              ("debye_squared_sinusoidal.yaml", [5.0,
                                                            5.024471741852423,
                                                    5.095491502812527,
                                                    5.206107373853763,
                                                    5.345491502812527,
                                                    5.5,
                                                    5.654508497187473,
                                                    5.793892626146237,
                                                    5.904508497187473,
                                                    5.975528258147577,
                                                    6.0,
                                                    5.975528258147577,
                                                    5.904508497187474,
                                                    5.793892626146237,
                                                    5.654508497187473,
                                                    5.5,
                                                    5.345491502812527,
                                                    5.206107373853763,
                                                    5.095491502812527,
                                                    5.024471741852423,
                                                    5.0,
                                                    5.024471741852423,
                                                    5.095491502812526,
                                                    5.206107373853763,
                                                    5.345491502812526,
                                                    5.5,
                                                    5.654508497187473,
                                                    5.793892626146237,
                                                    5.904508497187473,
                                                    5.975528258147577,
                                                    6.0,
                                                    5.975528258147577,
                                                    5.904508497187474,
                                                    5.793892626146237,
                                                    5.654508497187474,
                                                    5.5,
                                                    5.345491502812527,
                                                    5.206107373853763,
                                                    5.095491502812527,
                                                    5.024471741852423,
                                                    5.0,
                                                    5.024471741852423,
                                                    5.095491502812526,
                                                    5.206107373853763,
                                                    5.345491502812526,
                                                    5.5,
                                                    5.654508497187473,
                                                    5.793892626146237,
                                                    5.904508497187473,
                                                    5.975528258147577,
                                                    6.0,
                                                    5.975528258147577,
                                                    5.904508497187474,
                                                    5.793892626146236,
                                                    5.654508497187474,
                                                    5.500000000000001,
                                                    5.345491502812527,
                                                    5.206107373853763,
                                                    5.095491502812527,
                                                    5.024471741852423,
                                                    5.0,
                                                    5.024471741852423,
                                                    5.095491502812526,
                                                    5.206107373853762,
                                                    5.345491502812526,
                                                    5.500000000000001,
                                                    5.654508497187473,
                                                    5.793892626146236,
                                                    5.904508497187473,
                                                    5.975528258147577,
                                                    6.0,
                                                    5.975528258147577,
                                                    5.904508497187474,
                                                    5.793892626146237,
                                                    5.654508497187474,
                                                    5.500000000000002,
                                                    5.345491502812527,
                                                    5.206107373853763,
                                                    5.095491502812527,
                                                    5.024471741852424,
                                                    5.0,
                                                    5.024471741852423,
                                                    5.095491502812526,
                                                    5.206107373853762,
                                                    5.345491502812526,
                                                    5.5,
                                                    5.654508497187473,
                                                    5.793892626146236,
                                                    5.904508497187473,
                                                    5.975528258147577,
                                                    6.0,
                                                    5.975528258147577,
                                                    5.904508497187474,
                                                    5.793892626146237,
                                                    5.654508497187474,
                                                    5.500000000000002,
                                                    5.345491502812527,
                                                    5.206107373853763,
                                                    5.095491502812527,
                                                    5.024471741852424])])
    def test_parameter_values(self, yaml_file, expected_parameter_values):
        simulation = colloids_run([yaml_file])
        f= np.loadtxt('update_reporter.csv', delimiter=",", dtype=float, unpack=True, skiprows=1)
        actual_parameter_values = f[1]
        return actual_parameter_values
        #assert actual_parameter_values == pytest.approx(expected_parameter_values,rel=1.0e-12, abs=1.0e-12)'''


if __name__ == '__main__':
    pytest.main([__file__])

