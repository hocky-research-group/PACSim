import filecmp
import os.path
import shutil
import subprocess
import pandas as pd
import pytest


class TestRunAndResume(object):
    @pytest.fixture
    def directory_name(self):
        return "test_run_and_resume"

    @pytest.fixture(autouse=True)
    def change_test_dir(self, request, monkeypatch):
        # Change the working directory to the directory of the test file.
        # See https://stackoverflow.com/questions/62044541/change-pytest-working-directory-to-test-case-directory
        monkeypatch.chdir(request.fspath.dirname)

    @pytest.fixture(autouse=True)
    def create_test_directory(self, directory_name):
        os.mkdir(directory_name)
        yield
        shutil.rmtree(directory_name)

    @pytest.fixture(autouse=True)
    def run(self, directory_name):
        subprocess.run("colloids-run run_test.yaml", shell=True, check=True, capture_output=True)
        assert os.path.isfile(directory_name + "/final_frame.gsd")
        assert os.path.isfile(directory_name + "/final_frame.xyz")
        assert os.path.isfile(directory_name + "/state_data.csv")
        assert os.path.isfile(directory_name + "/trajectory.gsd")
        assert os.path.isfile(directory_name + "/update_reporter.csv")
        # Remove speed column from state_data.csv which varies between runs.
        f = pd.read_csv(directory_name + "/state_data.csv", usecols=[0, 1, 2, 3])
        f.to_csv(directory_name + "/state_data.csv")

    @pytest.fixture(autouse=True)
    def resume(self, directory_name):
        subprocess.run("colloids-run resume_test.yaml", shell=True, check=True, capture_output=True)
        assert os.path.isfile(directory_name + "/checkpoint.chk")
        subprocess.run(f"colloids-resume resume_test.yaml {directory_name}/checkpoint.chk 50", shell=True,
                       check=True, capture_output=True)
        assert os.path.isfile(directory_name + "/final_frame_resume.gsd")
        assert os.path.isfile(directory_name + "/final_frame_resume.xyz")
        assert os.path.isfile(directory_name + "/state_data_resume.csv")
        assert os.path.isfile(directory_name + "/trajectory_resume.gsd")
        assert os.path.isfile(directory_name + "/update_reporter_resume.csv")
        # Remove speed column from state_data_resume.csv which varies between runs.
        f = pd.read_csv(directory_name + "/state_data_resume.csv", usecols=[0, 1, 2, 3])
        f.to_csv(directory_name + "/state_data_resume.csv")

    def test_run_and_resume(self, directory_name):
        assert filecmp.cmp(directory_name + "/final_frame.gsd", directory_name + "/final_frame_resume.gsd",
                           shallow=False)
        assert filecmp.cmp(directory_name + "/final_frame.xyz", directory_name + "/final_frame_resume.xyz",
                           shallow=False)
        assert filecmp.cmp(directory_name + "/state_data.csv", directory_name + "/state_data_resume.csv",
                           shallow=False)
        assert filecmp.cmp(directory_name + "/trajectory.gsd", directory_name + "/trajectory_resume.gsd",
                           shallow=False)
        assert filecmp.cmp(directory_name + "/update_reporter.csv", directory_name + "/update_reporter_resume.csv")


if __name__ == '__main__':
    pytest.main([__file__])
