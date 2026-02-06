from openmm import unit
import gsd.hoomd
from gsd.hoomd import Frame
import numpy as np
from typing import Sequence, Union

# Pymatgen imports for structures
from pymatgen.core import Structure
from pymatgen.io.cif import CifParser
from pymatgen.core import Lattice, Structure, Molecule
from scipy.optimize import minimize_scalar

from colloids.colloids_create import ConfigurationGenerator


class LatticeBuilder():
    """
    Builder that handles structure creation from CIF or coordinates, supercell expansion, 
    ASE/pymatgen structure conversion, and resizing to match colloid radii.
    """

    def __init__(self, cif: str, cluster_specifications: list[str], lattice_vector_scaling_matrix: Union[int, Sequence[int]], 
                radii: dict[str, unit.Quantity], brush_length: unit.Quantity, lattice_scale_factor: float, 
                lattice_scale_start: float, radii_padding_factor: unit.Quantity):
        """Constructor of the LatticeBuilder class.

        :param cif: 
            The .cif file that specifies the desired lattice structure of the output configuration files.
        :type cif: str
        :param cluster_specifications:
            If not using this lattice builder funcitonality, this parameter represents the filenames of existing lammps-data format files
            that define the clusters from which to generate a configuration. If using the functionality in this class, the output will be a
            lammps-data format file that is then passed into the configuration generator. This parameter should thus be a unit-length list with a 
            string specifying the desired name of the lammps-data output with lattice information.
        :type cluster_specifications: list[str]
        :param lattice_vector_scaling_factor:
            A scaling matrix for transforming the lattice vectors.
            If only a single integer is given, the same scale factor is used in all directions.
            Every scale factor in the matrix should be positive.
        :type lattice_vector_scaling_factor: Union[int, list[int]]
        :param radii:
            The radii of the different types of colloidal particles that appear in the initial configuration file.
            The keys of the dictionary are the types of the colloidal particles and the values are the radii.
            The unit of the radii must be compatible with nanometers and the values must be greater than zero.
        :type radii: dict[str, unit.Quantity]
        :param brush_length:
            The thickness of the brush in the Alexander-de Gennes polymer brush model [i.e., L in eq. (1)].
            The unit of the brush_length must be compatible with nanometers and the value must be greater than zero.
        :type brush_length: unit.Quantity
        :param lattice_scale_factor:
            Scale-up increment factor.
            The lattice scale factor must be greater than zero.
        :type lattice_scale_factor: float
        :param lattice_scale_start:
            Starting scale factor.
            The lattice scale start factor must be greater than zero.
        :type lattice_scale_start: float
        :param radii_padding_factor: 
            Extra gap added to radii.
            The unit of the radii padding factor should be compatible with nanometers and the value must be greater than zero.
        :type radii_padding_factor: unit.Quantity

        
        :raises ValueError:

        """
        self._cif = cif
        self._cluster_file = cluster_specifications[0]
        self._radii = radii
        self._brush_length = brush_length
        self._lattice_vector_scaling_matrix = lattice_vector_scaling_matrix
        self._lattice_scale_factor = lattice_scale_factor
        self._lattice_scale_start = lattice_scale_start
        self._radii_padding_factor = radii_padding_factor

    @staticmethod
    def _load_lattice_from_cif(cif_file):
        """Load structure from CIF file using pymatgen."""
        parser = CifParser(cif_file)
        structure = parser.get_structures()[0]
        return structure

    #@staticmethod
    #def _set_manual_lattice(self, lattice, coords, species):
       # """Manually define a structure (instead of CIF)."""
       # self.structure = Structure(lattice, species, coords)


    @staticmethod
    def _set_colloid_labels_atomicnum(atomic_numbers):
        """Label atoms as '1' ... 'N' based on atomic number."""
        element_list = np.unique(atomic_numbers).tolist()
        type_map = {atomic_number: element_list.index(atomic_number)+1 for atomic_number in atomic_numbers}
        #print("\tElement map:",type_map)
        type_list = [ str(type_map[atomic_number]) for atomic_number in atomic_numbers ]
        return type_list

    @staticmethod
    def _set_colloid_labels_typemap(atomic_species):
        """Label atoms as '1' ... 'N' based on species using pymatgen atom names."""
        element_list = np.unique([str(x.name) for x in atomic_species]).tolist()
        #print("\tElement map:",self.type_map)
        try:
            type_list = [ self.type_map[species.name] for species in atomic_species ]
            return type_list
        except KeyError:
            print("Error: You must specify a type index for all elements in the element list:",element_list)
            #sys.exit(1)

    @staticmethod
    def _calculate_distances(positions):
        """Pairwise distance matrix."""
        try:
            from scipy.spatial import distance_matrix
            dists = distance_matrix(positions,positions)
        except ImportError:
            n = len(positions)
            dists = np.zeros((n, n))
            for i in range(n):
                dists[i,:] = np.linalg.norm(positions-positions[i],axis=1)
        return dists

    @staticmethod
    def _check_overlap(distances, radii):
        """Check if any particles overlap."""
        n = len(radii)
        for i in range(n):
            for j in range(i + 1, n):
                if distances[i, j] <= (radii[i] + radii[j]):
                    return True
        return False

    @staticmethod
    def _give_smallest_connection(distances, radii):
        """
        Return smallest effective distance (gap between particle surfaces).

        Effective distance = d - (r_i + r_j).
        """
        n = len(radii)
        min_dist = np.inf
        min_pair = None

        eff_matrix = np.zeros((n,n))

        for i in range(n):
            for j in range(i + 1, n):
                eff = distances[i, j] - (radii[i] + radii[j])
                eff_matrix[i,j] = eff
                if eff < min_dist:
                    min_dist = eff
                    min_pair = (i, j)

        return min_dist, min_pair

    def resize_to_match_radii(self, test_matrix=(3,3,3)):
        """
        Expand supercell until no overlaps remain given target radii.
        """

        structure1 = self._load_lattice_from_cif(self._cif)
        sc = structure1.make_supercell(test_matrix, in_place=False)
        positions = sc.cart_coords

        #if self.type_map is None:
            #print("Setting atom types based on elements in structure file")
        types = self._set_colloid_labels_atomicnum(sc.atomic_numbers)
        print("types", types)
        #else:
           # print("Setting atom types based on elements in type_map")
          #  types = self._set_colloid_labels_typemap(sc.species)

        # Effective radii in same units as positions
        radii = [self._radii[str(t)].value_in_unit(unit.nanometer) for t in types]
        radii = np.array(radii) + self._brush_length.value_in_unit(unit.nanometer) + self._radii_padding_factor

        dists = self._calculate_distances(positions)
        i = 0

        while self._check_overlap(dists, radii):
            scale = self._lattice_scale_start + i * self._lattice_scale_factor
            positions_exp = positions * scale
            dists = self._calculate_distances(positions_exp)
            i += 1

        #print("Optimal scale factor is",scale)

        # Check smallest connection after scaling
        #print("Computing closest pairs...")
        min_gap, pair = self._give_smallest_connection(dists, radii)
        #print("...closest distance is",min_gap)
        
        structure2 = self._load_lattice_from_cif(self._cif) #, self._lattice_vector_scaling_matrix)
        structure = structure2.make_supercell(self._lattice_vector_scaling_matrix, in_place=False)
        positions2 = structure.cart_coords
        #positions_exp = positions2*scale
        positions_exp = positions2*scale

        #print("Final system size:",positions2.shape)
        
        #Put center-of-gravity at 0,0,0
        positions_exp -= positions_exp.mean(axis=0)

        #if self.type_map is None:
        types = self._set_colloid_labels_atomicnum(structure.atomic_numbers)
        #else:
          #  types = self._set_colloid_labels_typemap(self.structure.species)

#        radii = [radii_dict[str(t)].value_in_unit(unit.nanometer) for t in types]
#        radii = np.array(radii) + brush_length + spacing
        
        # Store resized system
        positions = positions_exp
       # self.types = types
       # self.scale = scale
        box = np.max(positions, axis=0) + np.max(radii) #+ padding
       # self.box =  np.max(self.positions, axis=0) + np.max(radii) #+ padding
       # if self._padding_factor:
          #  box += box * self._padding_factor

        return positions, box, types
        
        '''
        N = len(positions)
        

        frame = gsd.hoomd.Frame()
        frame.configuration.step = 0
        frame.configuration.dimensions = 3
        frame.configuration.box = [box[0], box[1], box[2], 0.0, 0.0, 0.0]
    
        # Particles
        frame.particles.N = N
        frame.particles.types = sorted(set(types))
        frame.particles.typeid = np.array(
        [frame.particles.types.index(t) for t in types], dtype=np.int32
        )
        frame.particles.position = np.array(positions, dtype=np.float32)

        return frame'''


    def write_lammps_data(self, positions, box, types, atom_type_map=None, triclinic=True):
        """
        Write a LAMMPS .lmp data file in 'full' style (no bonds).
        
        Parameters
        ----------
        positions : ndarray (N,3)
            Atomic positions in Cartesian coordinates.
        types : list of str
            Particle type labels (e.g. ['A','B',...]).
        box : array_like (3,)
            Simulation box size along x,y,z (for orthogonal).
        atom_type_map : dict, optional
            Map from type labels (e.g. {'A':1, 'B':2}).
            If None, assigned in order of appearance.
        triclinic : bool
            If True, writes xy, xz, yz tilt factors (all set to 0).
        """
        
        filename = self._cluster_file #path to output file

        #positions, box, types = self._resize_to_match_radii(structure, matrix=(3, 3, 3), test_matrix=(3,3,3))
         
        positions = np.asarray(positions)
        n_atoms = len(positions)
        uniq_types = sorted(set(types))
     
        if atom_type_map is None:
            atom_type_map = {t: i+1 for i, t in enumerate(uniq_types)}
     
        n_types = len(atom_type_map)
     
        with open(filename, "w") as f:
             # Header counts
            f.write("(written by ASE)\n\n")
            f.write(f"{n_atoms} atoms\n")
            f.write("0 bonds\n")
            f.write(f"{n_types} atom types\n")
            f.write("0 bond types\n\n")
    
            # Box boundaries
            f.write(f"0.0 {box[0]:.6f} xlo xhi\n")
            f.write(f"0.0 {box[1]:.6f} ylo yhi\n")
            f.write(f"0.0 {box[2]:.6f} zlo zhi\n")
            if triclinic:
                f.write("0.0 0.0 0.0 xy xz yz\n")
            f.write("\n")
    
            # Atom section
            f.write("Atoms # full\n\n")
            for i, (pos, t) in enumerate(zip(positions, types), start=1):
                type_id = atom_type_map[t]
                f.write(f"{i:6d}   0   {type_id:d}   0.0   {pos[0]:.6f}   {pos[1]:.6f}   {pos[2]:.6f}\n")


    def generate_configuration(self, positions, box, types) -> Frame:

        N = len(positions)
        
        frame = gsd.hoomd.Frame()
        frame.configuration.step = 0
        frame.configuration.dimensions = 3
        frame.configuration.box = [box[0], box[1], box[2], 0.0, 0.0, 0.0]
    
        # Particles
        frame.particles.N = N
        frame.particles.types = sorted(set(types))
        frame.particles.typeid = np.array(
        [frame.particles.types.index(t) for t in types], dtype=np.int32
        )
        frame.particles.position = np.array(positions, dtype=np.float32)

        return frame


        '''

        """
        Generate the initial positions of the colloids in a gsd.hoomd.Frame instance.

        The generated frame should contain the following attributes:
        - frame.particles.N
        - frame.particles.position
        - frame.particles.types
        - frame.particles.typeid
        - frame.configuration.box

        The generated frame should not populate the following attributes:
        - frame.particles.type_shapes
        - frame.particles.diameter
        - frame.particles.charge
        - frame.particles.mass attributes

        :return:
            The initial configuration of the colloids.
        :rtype: gsd.hoomd.Frame
        """
    
    #, positions, types, box, radii_dict, charges_dict, masses_dict,
                #  mass_unit=unit.dalton, length_unit = unit.nanometer, electric_potential_unit= (unit.milli * unit.volt)) -> Frame:
     #
         
        N = len(positions)
        
         #with gsd.hoomd.open(name=filename, mode='w') as traj:
        frame = gsd.hoomd.Frame()
        frame.configuration.step = 0
        frame.configuration.dimensions = 3
        frame.configuration.box = [box[0], box[1], box[2], 0.0, 0.0, 0.0]
    
        # Particles
        frame.particles.N = N
        frame.particles.types = sorted(set(types))
        frame.particles.typeid = np.array(
        [frame.particles.types.index(t) for t in types], dtype=np.int32
        )
        frame.particles.position = np.array(positions, dtype=np.float32)
    
        # Mass, charge, diameter using units
        frame.particles.mass = np.array(
        [masses_dict[frame.particles.types[i]].value_in_unit(mass_unit)
        for i in frame.particles.typeid], dtype=np.float32
        )
        frame.particles.charge = np.array(
        [charges_dict[frame.particles.types[i]].value_in_unit(electric_potential_unit)
        for i in frame.particles.typeid], dtype=np.float32
        )
        frame.particles.diameter = np.array(
        [2.0 * radii_dict[frame.particles.types[i]].value_in_unit(length_unit)
        for i in frame.particles.typeid], dtype=np.float32
        )

        return frame
'''