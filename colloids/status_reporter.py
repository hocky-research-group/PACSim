from typing import Any
import openmm.app
import tqdm


class StatusReporter(object):
    """
    Reporter for an OpenMM simulation of colloids that shows a progress bar using the tqdm Python package.

    For general information on the tqdm Python package, see https://tqdm.github.io.

    :param report_interval:
        The interval (in time steps) at which the progress bar of the OpenMM simulation is updated.
        A sensible value is the total number of steps divided by 100.
        The value must be greater than zero.
    :type report_interval: int
    :param total_number_steps:
        The total number of steps of the OpenMM simulation.
        The value must be greater than zero.
    :type total_number_steps: int

    :raises ValueError:
        If the report interval is not greater than zero.
    :raises ValueError:
        If the total number of steps is not positive.
    """

    def __init__(self, report_interval: int, total_number_steps: int) -> None:
        """Constructor of the StatusReporter class."""
        if not report_interval > 0:
            raise ValueError("The report interval must be greater than zero.")
        if not total_number_steps >= 0:
            raise ValueError("The total number of steps must be positive.")
        self._report_interval = report_interval
        self._status = tqdm.tqdm(miniters=1, total=total_number_steps, unit="steps")

    # noinspection PyPep8Naming
    def describeNextReport(self, simulation: openmm.app.Simulation) -> tuple[int, bool, bool, bool, bool, bool]:
        """Get information about the next report this reporter will generate.

        This method is called by OpenMM once this reporter is added to the list of reporters of a simulation.

        :param simulation:
            The simulation to generate a report for.
        :type simulation: openmm.app.Simulation

        :returns:
            (Number of steps until next report,
            Whether the next report requires positions (False),
            Whether the next report requires velocities (False),
            Whether the next report requires forces (False),
            Whether the next report requires energies (False),
            Whether positions should be wrapped to lie in a single periodic box (False))
        :rtype: tuple[int, bool, bool, bool, bool, bool]
        """
        steps = self._report_interval - simulation.currentStep % self._report_interval
        return steps, False, False, False, False, False

    def report(self, *args: Any) -> None:
        """
        Generate a report by updating the progress bar.

        :param args:
            Variable length argument list that is included for compatibility with the OpenMM API.
        :type args: Any
        """
        self._status.update(self._report_interval)

    def __del__(self) -> None:
        """Destructor of the StatusReporter class."""
        try:
            self._status.close()
        except AttributeError:
            # If another error occured, the '_status' attribute might not exist.
            pass
