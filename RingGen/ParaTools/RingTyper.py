import numpy as np
import itertools

class RingTyper:
    """
    Tool to find rings (closed loops in connectivity graph) in a molecule.
    """
    def __init__(self, elements, connectivity, 
                 bonds = None, bond_orders = None):
        self.connectivity = connectivity
        self.elements = elements
        self.rings = self.get_rings()
        self.ring_ids = self.get_ring_id_dict()
        
        #Ring types can only be determined if bond orders are provided
        if bonds is not None and bond_orders is not None:
            self.ring_types = self.get_ring_types(bonds, bond_orders)

    def get_rings(self):
        """
        Finds rings in structure based on connectivity. Rings up to a length 
        of 6 atoms are detected.
        
        
        Returns
        -------
        ring_paths : list of lists of int
            Each list entry is a list of atoms that form a ring.

        """
        reduced_connectivity, atom_key = \
            self._reduce_connectivity(self.connectivity, self.elements)
        ring_paths = self._get_ring_paths(reduced_connectivity)
        ring_paths.sort()
        #remove dublicates:
        ring_paths = list(ring_paths 
                          for ring_paths,_ in itertools.groupby(ring_paths)) 
        ring_paths = self._correct_ring_atom_numbers(ring_paths, atom_key)
        return ring_paths

    def get_ring_id_dict(self):
        """
        Creates a dictionary which lists all the rings an atom belongs to.

        Returns
        -------
        ring_atom_ring_id_dict : dict
            Dictionary with each atom index being a key with the associated
            value being lists of ring ids (int). Order of rings is the same as
            in self.rings.
        """
        ring_atom_ring_id_dict = [ [] for x in range(len(self.elements)) ]

        for i, ring in enumerate(self.rings):
            for j, atom in enumerate(self.elements):
                if j in ring:
                    ring_atom_ring_id_dict[j].append(i)

        return ring_atom_ring_id_dict

    def get_ring_types(self, bonds, bond_orders):
        """
        Determines aromaticity level of ring based on atoms, number of 
        connections and bond orders.
        """
        planar_atoms = ["C_3", "N_3", "N_2", "O_2", "S_2", "P_2", "P_3"]
        aliphatic_atoms = ["C_4", "N_3", "O_2", "S_2"]
        atom_types = []
        for i, element in enumerate(self.elements):
            nb_bonds = len(np.where(self.connectivity[i] == 1)[0])
            atom_types.append(f"{element}_{nb_bonds}")
            
        
        
        double_bonded_atoms = [[] for x in range(len(self.elements))]
        single_bonded_atoms = [[] for x in range(len(self.elements))]
        for i, bond_order in enumerate(bond_orders):
            bond = bonds[i]
            if bond_order == 2:
                double_bonded_atoms[bond[0]].append(bond[1])
                double_bonded_atoms[bond[1]].append(bond[0])
            if bond_order == 1:
                single_bonded_atoms[bond[0]].append(bond[1])
                single_bonded_atoms[bond[1]].append(bond[0])
        ring_types = []
        for ring in self.rings:
            planar = all([atom_types[x] in planar_atoms for x in ring])
            aliphatic = all([atom_types[x] in aliphatic_atoms for x in ring])
            outside_double_bonds = \
                any([all([(x not in ring and self.elements[x] != "C") 
                          for x in double_bonded_atoms[atom]]) 
                     for atom in ring])
            adjacent_single_bonds = \
                any([(sum([(x in ring) for x in single_bonded_atoms[atom]])==2 
                      and atom_types[atom] != "C_3")
                     for atom in ring])
            if not planar and aliphatic:
                ring_types.append("AR5")
            elif not planar:
                ring_types.append("AR4")
            elif planar and outside_double_bonds:
                ring_types.append("AR3")
            elif planar and adjacent_single_bonds or len(ring) == 4:
                ring_types.append("AR2")
            else:
                 ring_types.append("AR1")

        return ring_types

    @staticmethod
    def _reduce_connectivity(connectivity, elements):
        """
        Finds hydrogens, removes columns and rows from connectivity matrix at
        those indices.
        Want to do ring search without hydrogens for easier computation.
        """
        hydrogens = []
        atom_key = {}
        non_hydrogen_counter = 0
        for i,atom in enumerate(elements):
            if atom == "H":
                hydrogens.append(i)
            else:
                atom_key.update({non_hydrogen_counter:i})
                non_hydrogen_counter+=1
        con_mod = np.delete(np.delete(connectivity,
                                      hydrogens,
                                      axis=0),
                            hydrogens,
                            axis=1)
        return con_mod, atom_key

    @staticmethod
    def _get_ring_paths(reduced_connectivity):
        """
        Get all paths starting at all atoms, check if they form rings
        """
        path_depth = 6
        ring_paths = []
        for i,con in enumerate(reduced_connectivity):
            paths = [[i]]
            for j in range(path_depth):
                paths = RingTyper._take_step_in_path_search(
                    paths,reduced_connectivity)
                if j >= 1:
                    ring_paths.extend(
                        RingTyper._check_if_path_is_ring(i,
                                                         paths,
                                                         reduced_connectivity))
        return ring_paths

    @staticmethod
    def _take_step_in_path_search(paths, reduced_connectivity):
        """
        Extends paths by adding an atom that is connected to 
        the last atom in path.
        """
        new_paths = []
        for j,path in enumerate(paths):
            connections = np.where(reduced_connectivity[path[-1]] == 1)[0]
            for con in connections:
                if con not in path:
                    new_paths.append(path+[con])
        return new_paths

    @staticmethod
    def _check_if_path_is_ring(i, paths, reduced_connectivity):
        """
        Goes through paths, if last atom connected to first atom in path,
        add to ring_paths.
        """
        ring_paths = []
        for path in paths:
            atoms_connected_to_last_atom_in_path = \
                np.where(reduced_connectivity[path[-1]] == 1)[0]
            if i in atoms_connected_to_last_atom_in_path:
                ring_paths.append(list(np.sort(path)))
        return ring_paths

    @staticmethod
    def _correct_ring_atom_numbers(ring_paths, atom_key):
        """
        Corrects atom indices for actual structure that contains hydrogens.
        """
        for ring in ring_paths:
            for i,atom in enumerate(ring):
                ring[i] = atom_key[atom]
        return ring_paths