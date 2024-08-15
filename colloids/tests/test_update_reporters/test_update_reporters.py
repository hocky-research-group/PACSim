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
                             [("debye_linearmonotonic.yaml", [5.0, 5.1,
                                                            5.199999999999999,
                                                            5.299999999999999,
                                                            5.399999999999999,
                                                            5.499999999999998,
                                                            5.599999999999998,
                                                            5.6999999999999975,
                                                            5.799999999999997,
                                                            5.899999999999997,
                                                            5.9999999999999964]),
                              ("debye_linearunimodal.yaml", [5.0, 5.333333333333333,
                                                            5.666666666666666,
                                                            5.999999999999999,
                                                            5.857142857142856,
                                                            5.714285714285713,
                                                            5.571428571428569,
                                                            5.428571428571426,
                                                            5.285714285714283,
                                                            5.14285714285714,
                                                            4.9999999999999964]),
                              ("debye_sinusoidal.yaml", [5.0,
                                                            4.877641290737884,
                                                            4.522542485937369,
                                                            3.9694631307311834,
                                                            3.2725424859373686,
                                                            2.5000000000000004,
                                                            1.7274575140626323,
                                                            1.0305368692688175,
                                                            0.4774575140626317,
                                                            0.12235870926211624,
                                                            7.498798913309288e-32,
                                                            0.1223587092621159,
                                                            0.47745751406263104,
                                                            1.0305368692688168,
                                                            1.727457514062631,
                                                            2.4999999999999996,
                                                            3.272542485937368,
                                                            3.9694631307311825,
                                                            4.522542485937368,
                                                            4.877641290737883,
                                                            5.0,
                                                            4.877641290737884,
                                                            4.522542485937369,
                                                            3.969463130731184,
                                                            3.27254248593737,
                                                            2.500000000000001,
                                                            1.7274575140626323,
                                                            1.030536869268818,
                                                            0.47745751406263204,
                                                            0.12235870926211646,
                                                            2.999519565323715e-31,
                                                            0.12235870926211569,
                                                            0.4774575140626307,
                                                            1.0305368692688162,
                                                            1.7274575140626303,
                                                            2.499999999999999,
                                                            3.272542485937367,
                                                            3.969463130731181,
                                                            4.522542485937367,
                                                            4.877641290737883,
                                                            5.0,
                                                            4.877641290737885,
                                                            4.522542485937369,
                                                            3.96946313073118,
                                                            3.272542485937371,
                                                            2.500000000000006,
                                                            1.7274575140626327,
                                                            1.0305368692688148,
                                                            0.47745751406263254,
                                                            0.12235870926211798,
                                                            6.748919021978358e-31,
                                                            0.1223587092621141,
                                                            0.4774575140626303,
                                                            1.030536869268819,
                                                            1.7274575140626294,
                                                            2.4999999999999942,
                                                            3.272542485937367,
                                                            3.969463130731185,
                                                            4.522542485937367,
                                                            4.877641290737882,
                                                            5.0,
                                                            4.877641290737883,
                                                            4.52254248593737,
                                                            3.969463130731188,
                                                            3.272542485937371,
                                                            2.4999999999999982,
                                                            1.7274575140626334,
                                                            1.0305368692688228,
                                                            0.4774575140626329,
                                                            0.12235870926211546,
                                                            1.199807826129486e-30,
                                                            0.1223587092621139,
                                                            0.47745751406263,
                                                            1.0305368692688186,
                                                            1.7274575140626294,
                                                            2.4999999999999933,
                                                            3.272542485937366,
                                                            3.969463130731184,
                                                            4.522542485937367,
                                                            4.877641290737882,
                                                            5.0,
                                                            4.877641290737883,
                                                            4.52254248593737,
                                                            3.9694631307311887,
                                                            3.2725424859373713,
                                                            2.499999999999999,
                                                            1.7274575140626347,
                                                            1.0305368692688233,
                                                            0.47745751406263326,
                                                            0.12235870926211563,
                                                            1.8746997283273223e-30,
                                                            0.12235870926211373,
                                                            0.47745751406262427,
                                                            1.0305368692688108,
                                                            1.7274575140626283,
                                                            2.500000000000001,
                                                            3.2725424859373744,
                                                            3.9694631307311914,
                                                            4.522542485937367,
                                                            4.877641290737879])])
    def test_parameter_valuesfi(self, yaml_file, expected_parameter_values):
        simulation = colloids_run([yaml_file])
        f= np.loadtxt('update_reporter.csv', delimiter=",", dtype=float, unpack=True, skiprows=1)
        actual_parameter_values = f[1]
        print(expected_parameter_values)
        assert actual_parameter_values == pytest.approx(expected_parameter_values,rel=1.0e-12, abs=1.0e-12)


if __name__ == '__main__':
    pytest.main([__file__])

