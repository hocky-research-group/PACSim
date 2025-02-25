"""
Define the unit system of the colloids package.

Note that units like unit.nano * unit.meter should not be created often because it is slow and can lead to memory leaks.
"""
from openmm import unit

electric_potential_unit = unit.milli * unit.volt
energy_unit = unit.kilojoule_per_mole
length_unit = unit.nano * unit.meter
mass_unit = unit.amu
temperature_unit = unit.kelvin
time_unit = unit.pico * unit.second
