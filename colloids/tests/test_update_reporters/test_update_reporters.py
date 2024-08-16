from math import pi, sin
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
                             [("debye_ramp.yaml", [5.0 + i * 0.1 for i in range(11)]),
                              ("debye_triangle.yaml", ([5.0 + i * (1.0 / 3.0) for i in range(4)]
                                                       + [6.0 - i * (1.0 / 3.0) for i in range(1, 4)]
                                                       + [5.0 + i * (1.0 / 3.0) for i in range(1, 4)]
                                                       + [6.0 - (1.0 / 3.0)])),
                              ("debye_squared_sinusoidal.yaml", [5.0 + (sin(pi / (2.0 * 10) * i) ** 2)
                                                                 for i in range(101)])])
    def test_parameter_values(self, yaml_file, expected_parameter_values):
        colloids_run([yaml_file])
        f= np.loadtxt('update_reporter.csv', delimiter=",", dtype=float, skiprows=1)
        actual_parameter_values = f[:, 1]
        assert actual_parameter_values == pytest.approx(expected_parameter_values,rel=1.0e-12, abs=1.0e-12)


if __name__ == '__main__':
    pytest.main([__file__])

