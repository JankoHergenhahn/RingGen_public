import importlib
import numpy as np
from copy import deepcopy

from .Coordinates import Coordinates
from .Topology import TopFile
from .Tools import get_COM, get_normal_vector
from .TopologyGenerator import TopologyGenerator

class Defaults:
    """
    Handles import of default files.
    
    Attributes
    ---------
    atom_types : list of str
        contain information about non-bonded parameters for GROMACS files ,
        read from RingGen.DefaultFiles/default_atom_types.txt ,
    coordinate_fragments : dict
        contains coordinate building blocks, 
        read from any .xyz files in RingGen.DefaultFiles/ ,
        organized as {name: RingGen.Coordinates}
    topology_fragments
        contains topology building blocks, 
        read from RingGen.DefaultFiles/default.top ,
        organized as {name: RingGen.Coordinates}
    
    """
    def __init__(self):
        self.atom_types = self._load_default_atom_types()
        self.coordinate_fragments = self._load_default_coordinate_fragments()
        self.topology_fragments = self._load_default_topology_fragments()
           
    def add_topology_fragment(self, topology):
        """
        Adds topology object directly to topology fragments.
        """
        self.topology_fragments.add_moleculetype(topology)
        
    def add_fragment_from_file(self, file):
        """
        Used to generate topology and coordinate fratgment from file.

        Parameters
        ----------
        file : str
            File name of coordinates, must be .xyz file type.

        Returns
        -------
        None.
        """
        #Get name
        if "/" in file:
            name = file.split("/")[-1].split(".")[0]
        else:
            name = file.split(".")[0]
        
        #Process Coordinates
        coordinates = Coordinates().read_xyz(file)
        with open(file) as f:
            H_connection_points = [int(x) for x in f.readlines()[1].split()]
        frag_coords = self._process_coordinates(coordinates, 
                                                H_connection_points)
        self.coordinate_fragments.update({name: frag_coords})
        
        #Process Topology
        connecting_atoms = self._get_connecting_atoms(coordinates, 
                                                      H_connection_points)
        generator = TopologyGenerator().generate_topology(file)
        top = generator.get_topology()
        
        for i in H_connection_points:
            top.atoms[i][2] = 2
        top.reduce([1])
        top.con_atoms = connecting_atoms
        top.set_name(name)
        self.add_topology_fragment(top)
        
    def add_fragment_from_structure(self, structure, H_connection_points):
        """
        Used to generate topology and coordinate fratgment from structure.

        Parameters
        ----------
        structure : RingGen.Structure
            Structure object that must contain a topology and a coordinate 
            object. 
        H_connection_points : list of int
            Indices of hydrogen atoms that are removed at the fragment
            "connecting points", so that the fragment topology and coordinates
            can be used to construct larger structures.
        
        Returns
        -------
        None.
        """
        name = structure.name
        connecting_atoms = self._get_connecting_atoms(structure.coordinates, 
                                                      H_connection_points)
        frag_coords = self._process_coordinates(structure.coordinates,
                                                H_connection_points)
        self.coordinate_fragments.update({name: frag_coords})
        
        #Topology
        top = deepcopy(structure.topology)
        res_ids = list(set([x[2] for x in top.atoms]))
        next_res_id = max(res_ids) + 1
        for i in H_connection_points:
            top.atoms[i][2] = next_res_id
        top.reduce(res_ids)
        top.con_atoms = connecting_atoms
        top.set_name(name)

        self.add_topology_fragment(top)
        
        return self
    
  #Local methods
    @staticmethod
    def _load_default_atom_types():
        """
        Loads default atom types.
        """
        default_atom_types = []
        with importlib.resources.open_text('RingGen.DefaultFiles',
                                           'default_atom_types.txt') as f:
            for line in f:
                default_atom_types.append(line.strip('\n'))
        return default_atom_types
    
    @staticmethod
    def _load_default_topology_fragments():
        """
        Loads default topology fragments.
        """
        with importlib.resources.path("RingGen.DefaultFiles", 
                                      "default.top") as file:
            fragments = TopFile().read_file(file)
        fragments = Defaults._edit_topologies(fragments)
        return fragments
    
    @staticmethod
    def _edit_topologies(fragments):
        """
        Processes topologies based on their names.
        
        Names of linker topologies have the structure "LINKER_FragA_FragB".
            The connecting atoms for the liner fragment are determined from 
            the bonded parameters.
        Names of topologies for normal building blocks have the structure
            "FragA:i:j:k..." where i,j,k are atom indices indicating the 
            connecting atoms of the fragment
        """
        keys_copy = [x for x in fragments.moleculetypes.keys()]
        delete_names = []
        for top_name in keys_copy:
            #Linker topology fragment
            if top_name.startswith("LINKER_"):
                top_temp = fragments.moleculetypes[top_name]
    
                connection = [x-1 for x in top_temp.bonds[0][:2]]
                fragments.moleculetypes[top_name].connection = connection 
                
                nb_atoms = len(top_temp.atoms)
                connectivity = np.zeros((nb_atoms, nb_atoms))
                for dihedral in top_temp.dihedrals:
                    if dihedral[4] == 1:
                        connectivity[dihedral[0]-1, dihedral[1]-1] = 1
                        connectivity[dihedral[1]-1, dihedral[0]-1] = 1
                        
                        connectivity[dihedral[1]-1, dihedral[2]-1] = 1
                        connectivity[dihedral[2]-1, dihedral[1]-1] = 1
                        
                        connectivity[dihedral[2]-1, dihedral[3]-1] = 1
                        connectivity[dihedral[3]-1, dihedral[2]-1] = 1
                
                fragments.moleculetypes[top_name].connectivity = connectivity
            #Normal building block
            if ":" in top_name:
                delete_names.append(top_name)
                cols = top_name.split(":")
                top_temp = deepcopy(fragments.moleculetypes[top_name])
                
                con_atoms = []
                for i in range(1, len(cols)):
                    con_atoms.append(int(cols[i]))
                top_temp.con_atoms = con_atoms
    
                fragments.moleculetypes.update({cols[0]: top_temp})
                    
        for name in delete_names:
            del fragments.moleculetypes[name]
        return fragments
  
    @staticmethod
    def _load_default_coordinate_fragments():
        """
        Loads default coordinate fragments.
        """
        default_coordinate_fragments = {}
        #Create hydrogen fragment manually
        H_coords = Coordinates().set(["H"], np.zeros((1,3)))
        H_coords.distance = 1
        H_coords.connecting_atoms = []
        default_coordinate_fragments.update({"H": H_coords})
        
        #Load other files from .xyz files
        folder = "RingGen.DefaultFiles"
        for file in importlib.resources.contents("RingGen.DefaultFiles"):
            base_name = file.split(".")[0]
            file_type = file.split(".")[-1]
            if file_type == "xyz":
                with importlib.resources.path(folder, file) as file:
                    coordinates = Coordinates().read_xyz(file)
                    # second line contains connecting points for structure
                    # generation
                    with open(file) as f:
                        H_connection_points = \
                            [int(x) for x in f.readlines()[1].split()]
                    fragment = Defaults._process_coordinates(
                            coordinates, H_connection_points)
                    default_coordinate_fragments.update({base_name: fragment})
        return default_coordinate_fragments
            
    @staticmethod
    def _process_coordinates(coordinates, H_connection_points):
        """
        Ensures that coordinate fragments have a consistent orientation 
        relative to their connecting points. 
        Hydrogens in connecting positions are removed so that the coordinates
        can be used asa fragment to build larger sstructures.
        """
        coords = deepcopy(coordinates)
        connecting_atoms = Defaults._get_connecting_atoms(coords, 
                                                          H_connection_points)
        coords.remove_coordinates(H_connection_points)
        
        if len(H_connection_points) == 0:
            coords.distance = 0
        elif len(H_connection_points) == 1:
            coords.translate(-1 * coords.coordinates[connecting_atoms[0]])
            vec = get_COM(coords.coordinates)
            vec /= np.linalg.norm(vec)
            coords.rotate(vec + [1,0,0])
            plane_vec = get_normal_vector(coords.coordinates)
            angle = np.arccos(np.dot(plane_vec, [0,1,0]))
            coords.rotate([1,0,0], -angle)
            coords.distance = np.max(coords.coordinates[:,0])
            coords.distance += 1
        else:
            coords.translate(-1 * coords.coordinates[connecting_atoms[0]])
            vec = deepcopy(coords.coordinates[connecting_atoms[1]])
            vec /= np.linalg.norm(vec)
            coords.rotate(vec + [1,0,0], np.pi)
            plane_vec = get_normal_vector(coords.coordinates)
            angle = np.arccos(np.dot(plane_vec, [0,1,0]))
            coords.rotate([1,0,0], -angle)
            coords.distance = np.linalg.norm(
                coords.coordinates[connecting_atoms[1]] - 
                coords.coordinates[connecting_atoms[0]])
            coords.distance += 1.5
        
        coords.connecting_atoms = connecting_atoms        
        return coords
    
    @staticmethod
    def _get_connecting_atoms(coordinates, H_connection_points):
        """
        Finds atoms connected to the hydrogen atoms specified in 
        H_connection_points (list of int).
        """
        coords = deepcopy(coordinates)
        coords.create_connectivity()
        
        connecting_atoms_coordinates = []
        for i in H_connection_points:
            connecting_atoms_coordinates.append(
                coords.coordinates[np.where(coords.connectivity[i])[0][0]])
        coords.remove_coordinates(H_connection_points)
        connecting_atoms = []
        for c_ca in connecting_atoms_coordinates:
            for i, c in enumerate(coords.coordinates):
                if np.linalg.norm(c-c_ca) < 0.1:
                    connecting_atoms.append(i)
                    
        return connecting_atoms
    
    
    
        
        