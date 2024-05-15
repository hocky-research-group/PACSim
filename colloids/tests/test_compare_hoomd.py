import openmm
from openmm import app
from openmm import unit
import pytest
from colloids import ColloidPotentialsAlgebraic, ColloidPotentialsParameters, ColloidPotentialsTabulated
from colloids.helper_functions import read_xyz_file
try:
    import hoomd
    import hoomd.md
    from colloids.colloid_potentials_tabulated_hoomd import ColloidPotentialsTabulatedHoomd
except ModuleNotFoundError:
    # Skip all tests from this module.
    pytestmark = pytest.mark.skipif(True, reason="hoomd is not installed")


class TestCompareHoomd(object):
    @pytest.fixture
    def parameters(self):
        params = {
            "radius_positive": 105.0 * (unit.nano * unit.meter),
            "radius_negative": 95.0 * (unit.nano * unit.meter),
            "surface_potential_positive": 44.0 * (unit.milli * unit.volt),
            "surface_potential_negative": -54.0 * (unit.milli * unit.volt),
            "colloid_potentials_parameters": ColloidPotentialsParameters(
                brush_density=0.09 / ((unit.nano * unit.meter) ** 2),
                brush_length=10.6 * (unit.nano * unit.meter),
                debye_length=5.726968 * (unit.nano * unit.meter),
                temperature=298.0 * unit.kelvin,
                dielectric_constant=80.0),
            "collision_rate": 0.01 / (unit.pico * unit.second),
            "timestep": 0.05 * (unit.pico * unit.second),
            "mass_positive": 1.0 * unit.amu,
            "side_length": 12328.05 * (unit.nano * unit.meter),
        }
        params["mass_negative"] = (
                (params["radius_negative"] / params["radius_positive"]) ** 3 * params["mass_positive"])
        return params

    @pytest.fixture(autouse=True)
    def change_test_dir(self, request, monkeypatch):
        # Change the working directory to the directory of the test file.
        # See https://stackoverflow.com/questions/62044541/change-pytest-working-directory-to-test-case-directory
        monkeypatch.chdir(request.fspath.dirname)

    @pytest.fixture(params=["first_frame", "final_frame"])
    def filename(self, request):
        return request.param

    @pytest.fixture(params=["algebraic", "tabulated"])
    def openmm_result(self, parameters, filename, request):
        types, positions, cell = read_xyz_file(f"{filename}.xyz")
        topology = app.topology.Topology()
        chain = topology.addChain()
        residue = topology.addResidue("res1", chain)
        for t, position in zip(types, positions):
            topology.addAtom(t, None, residue)
        topology.setPeriodicBoxVectors(cell)

        system = openmm.System()
        system.setDefaultPeriodicBoxVectors(openmm.Vec3(*cell[0]), openmm.Vec3(*cell[1]), openmm.Vec3(*cell[2]))
        platform = openmm.Platform.getPlatformByName("Reference")
        integrator = openmm.LangevinIntegrator(parameters["colloid_potentials_parameters"].temperature,
                                               parameters["collision_rate"], parameters["timestep"])
        if request.param == "algebraic":
            colloid_potentials = ColloidPotentialsAlgebraic(
                colloid_potentials_parameters=parameters["colloid_potentials_parameters"], use_log=False)
        else:
            assert request.param == "tabulated"
            colloid_potentials = ColloidPotentialsTabulated(
                radius_one=parameters["radius_positive"], radius_two=parameters["radius_negative"],
                surface_potential_one=parameters["surface_potential_positive"],
                surface_potential_two=parameters["surface_potential_negative"],
                colloid_potentials_parameters=parameters["colloid_potentials_parameters"], use_log=False)
        for t, position in zip(types, positions):
            if t == "P":
                system.addParticle(parameters["mass_positive"])
                colloid_potentials.add_particle(radius=parameters["radius_positive"],
                                                surface_potential=parameters["surface_potential_positive"])
            else:
                assert t == "N"
                system.addParticle(parameters["mass_negative"])
                colloid_potentials.add_particle(radius=parameters["radius_negative"],
                                                surface_potential=parameters["surface_potential_negative"])
        for force in colloid_potentials.yield_potentials():
            system.addForce(force)
        simulation = app.Simulation(topology, system, integrator, platform)
        simulation.context.setPositions(positions)
        simulation.context.setVelocitiesToTemperature(parameters["colloid_potentials_parameters"].temperature, 1)
        openmm_state = simulation.context.getState(getEnergy=True)
        return openmm_state.getPotentialEnergy()

    @pytest.fixture
    def hoomd_result(self, parameters, filename):
        hoomd.context.initialize("--mode=cpu --notice-level=0")
        snapshot = hoomd.data.gsd_snapshot(f"{filename}.gsd")
        hoomd.init.read_snapshot(snapshot)
        nl = hoomd.md.nlist.cell()
        ColloidPotentialsTabulatedHoomd(
            radius_one=parameters["radius_positive"].value_in_unit(unit.nano * unit.meter),
            radius_two=parameters["radius_negative"].value_in_unit(unit.nano * unit.meter),
            surface_potential_one=parameters["surface_potential_positive"].value_in_unit(unit.milli * unit.volt),
            surface_potential_two=parameters["surface_potential_negative"].value_in_unit(unit.milli * unit.volt),
            type_one="P", type_two="N",
            colloid_potentials_parameters=parameters["colloid_potentials_parameters"],
            neighbor_list=nl, shift=False)
        k_temperature = (parameters["colloid_potentials_parameters"].temperature * unit.BOLTZMANN_CONSTANT_kB
                         * unit.AVOGADRO_CONSTANT_NA).value_in_unit(unit.kilojoule_per_mole)
        hoomd.md.integrate.mode_standard(dt=parameters["timestep"].value_in_unit(unit.pico * unit.second))
        langevin = hoomd.md.integrate.langevin(group=hoomd.group.all(), kT=k_temperature, seed=1)
        langevin.set_gamma("P", parameters["mass_positive"].value_in_unit(unit.amu)
                           * parameters["collision_rate"].value_in_unit((unit.pico * unit.second) ** -1))
        langevin.set_gamma("N", parameters["mass_negative"].value_in_unit(unit.amu)
                           * parameters["collision_rate"].value_in_unit((unit.pico * unit.second) ** -1))
        # noinspection PyTypeChecker
        log = hoomd.analyze.log(filename=None, quantities=["potential_energy"], period=1, overwrite=True, phase=-1)
        hoomd.run(0)
        return log.query("potential_energy")

    def test_compare_hoomd(self, openmm_result, hoomd_result):
        assert openmm_result.value_in_unit(unit.kilojoule_per_mole) == pytest.approx(hoomd_result, rel=1e-5)


if __name__ == '__main__':
    pytest.main([__file__])
