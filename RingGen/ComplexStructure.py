import numpy as np
from copy import deepcopy

from .Coordinates import Coordinates
from .Structure import Structure
from .Topology import Topology
try:
    from .Minimizer import Minimizer
except:
    from .Minimizer_SD import Minimizer  

class ComplexStructure:
    """
    Object to handle systems that contain more than one molecule.
    
    Attributes
    ----------
    structures : list of RingGen.Structures
    """
    def __init__(self, structures = []):
        self.structures = structures
        
    def add_structure(self,structure):
        """
        Checks structure has certain attributes and adds it to the system.
        
        Returns self: Allows method chaining.
        """
        structure = deepcopy(structure)
        if structure.coordinates.connectivity.size == 0:
            structure.coordinates.create_connectivity()
        if structure.connectivity.size == 0:
            structure.create_connectivity()
        if structure.elements == []:
            structure.create_elements()
        self.structures.append(structure)
        return self       
    
    def get_combined_structure(self):
        """
        Generates structure in which coordinates and topologies of all 
        structures have been combined.

        Returns
        -------
        combined_structures : RingGen.Structure
            Coordinates and atoms of topologies appear in same order as given 
            in the self.structures list.
        """
        combined_structures = Structure()

        #combine coordinates
        elements = [x for y in self.structures 
                    for x in y.coordinates.elements]
        coordinates = np.vstack([x.coordinates.coordinates 
                                 for x in self.structures])        
        combined_structures.coordinates = \
            Coordinates().set(elements, coordinates).create_connectivity()
            
        #combine topologies
        combined_structures.topology = self._combine_topologies()
        
        return combined_structures
        
    def minimize(self, 
                 included_structures = [], 
                 steps = 1000, 
                 constrained_atoms = []):
        """
        MM-Optimizes the geometry described by all coordinates using the 
        force field in the topology objects.
        
        Parameters
        ----------
        included_structures : list of int, optional
            Select which structures are included in optimization if specified. 
            The default is [], which selects all structures.
        steps : int, optional
            Maximum number of iterations in optimization. The default is 1000.
        constrained_atoms : list of int, optional
            Selected atoms will be frozen during optimization. 
            The default is [].

        Returns self: Allows method chaining.
        """
        #Preparation
        if included_structures == []:
            included_structures = np.arange(len(self.structures))
        temp_coordinates = Coordinates()
        temp_coordinates.set_coordinates(
            np.vstack([deepcopy(x.coordinates.coordinates) 
                       for i,x in enumerate(self.structures) 
                       if i in included_structures]))
        temp_coordinates.set_elements(
            [x for i,y in enumerate(self.structures) 
             for x in y.coordinates.elements 
             if i in included_structures])
        top_list = [deepcopy(x.topology) 
                    for i,x in enumerate(self.structures) 
                    if i in included_structures]
        temp_topology = self._combine_topologies(top_list)
        temp_topology.set_name("Temp")
        
        #Minimization
        my_minimizer = Minimizer(temp_topology, 
                                 temp_coordinates, 
                                 constrained_atoms = constrained_atoms)
        my_minimizer.minimize(steps = steps)
       
        #Output
        atom_index_cum = 0
        for i,x in enumerate(self.structures):
            coord_length = len(self.structures[i].coordinates.elements)
            coords = my_minimizer.get_coordinates()
            coords = coords[atom_index_cum:atom_index_cum+coord_length]
            self.structures[i].coordinates.set_coordinates(coords)
            atom_index_cum += coord_length
            
        return self
    
    def _combine_topologies(self, topologies_list = []):
        """
        Combines topologies listed in topologies_list.
        
        Parameters
        ----------
        topologies_list : list of RingGen.Topologies, optional
            If not given, will include topologies of all structures.

        Returns
        -------
        total_top : RingGen.Topology
            Combined topology with atoms ordered accoriding to topologies_list.
        """
        #Select all topolgies if topologies_list is unspecified
        if topologies_list == []:
            topologies_list = [deepcopy(x.topology) for x in self.structures]
        
        #concatenate atoms and bonded parameters
        topology_lengths = [len(t.atoms) for t in topologies_list]
        topology_lengths_cum = [sum(topology_lengths[:i]) 
                                for i,_ in enumerate(topology_lengths)]
        total_top = Topology()
        prop_length_dictionary = {"atoms":1, 
                                  "bonds":2, 
                                  "pairs":2, 
                                  "angles":3, 
                                  "dihedrals":4}
        for i,top in enumerate(topologies_list):
            for prop in ["atoms","bonds","pairs","angles","dihedrals"]:
                prop_temp = top.__dict__[prop]
                for j,_ in enumerate(prop_temp):
                    for k in range(prop_length_dictionary[prop]):
                        prop_temp[j][k] += topology_lengths_cum[i]
                total_top.__dict__[prop].extend(prop_temp)
        
        # Adjust residue ids
        final_residues = [t.atoms[-1][2] for t in topologies_list]
        final_residues_cum = [sum(final_residues[:i]) 
                              for i,_ in enumerate(final_residues)]
        residue_nr_corrections = []
        for i,topology_length in enumerate(topology_lengths):
            residue_nr_corrections.extend(
                topology_length * [final_residues_cum[i]])
            
        for i,atom in enumerate(total_top.atoms):
            total_top.atoms[i][2] += residue_nr_corrections[i]

        return total_top
    
    def combine_coordinates(self, coordinates_list = []):
        """
        Combines coordinates listed in coordinates_list.
        
        Parameters
        ----------
        coordinates_list : list of RingGen.Coordinates, optional
            If not given, will include coordinates of all structures.

        Returns
        -------
        total_coord : RingGen.Coordinates
            Combined coordinates with atoms ordered accoriding to 
            coordinates_list.
        """
        #Select all coordinates if coordinates_list is unspecified
        if coordinates_list == []:
            coordinates_list = [deepcopy(x.coordinates) 
                                for x in self.structures]
        
        elements = [x for y in coordinates_list for x in y.elements]
        coordinates = np.vstack([x.coordinates for x in coordinates_list])        
        return Coordinates().set(elements, coordinates).create_connectivity()   
    
    def write_full_xyz(self, output_name='out.xyz', comment=""):
        """
        Write .xyz format coordinate file with atoms from all structures.
        """
        self.combine_coordinates().write_xyz(output_name, comment)
        
    def write_full_pdb(self, output_name='out.xyz', comment=""):
        """
        Write .pdb format coordinate file with atoms from all structures.
        """
        temp_top = self._combine_topologies()
        temp_top.set_name("Complex")
        self.combine_coordinates().write_pdb(output_name, topology = temp_top)